"""FastAPI service for the GENZONE N-CMAPSS digital twin (AI Manthan): runs Layers 1-4 from the trained bundle and
serves the dashboard.

    python -m backend            # http://localhost:8000  (dashboard at /, API under /api)
"""
from __future__ import annotations

import io
import mimetypes
import os
import threading
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .pipeline.bundle import ModelBundle
from .pipeline.service import engine_payload, parse_upload, truth_columns
from .pipeline.store import ALERT_STATUS, EVENT_TYPES, FINDINGS, Store

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC = os.path.join(ROOT, "public")
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(ROOT, "backend", "models"))
DB_PATH = os.environ.get("GENZONE_DB", os.path.join(ROOT, "backend", "data", "genzone.db"))

mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("model/gltf+json", ".gltf")

app = FastAPI(title="GENZONE N-CMAPSS Digital Twin", version="0.3.0")


@app.middleware("http")
async def revalidate_dashboard_files(request, call_next):
    """Make browsers re-check dashboard files (ETag) on every load instead of reusing a stale cached copy."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


class State:
    def __init__(self):
        self.bundle: ModelBundle | None = None
        self.error: str | None = None
        self.uploads: dict[str, pd.DataFrame] = {}
        self.store: Store | None = None
        self.synced = False
        self.lock = threading.RLock()


STATE = State()


def get_bundle() -> ModelBundle:
    with STATE.lock:
        if STATE.bundle is None and STATE.error is None:
            try:
                STATE.bundle = ModelBundle.load(MODEL_DIR)
            except Exception as exc:          # surfaced to the dashboard instead of crashing
                STATE.error = f"{type(exc).__name__}: {exc}"
    if STATE.bundle is None:
        raise HTTPException(503, f"No model bundle loaded ({STATE.error}). Train with "
                                 "training/ncmapss_colab_training.ipynb and unzip the bundle into backend/models/.")
    return STATE.bundle


def set_bundle(bundle: ModelBundle | None):
    """Used by tests and tools to inject a bundle."""
    with STATE.lock:
        STATE.bundle, STATE.error = bundle, None
        STATE.uploads.clear()
        STATE.synced = False
    demo_results.cache_clear()
    physics_report.cache_clear()


def set_store(store: Store | None):
    """Used by tests to inject a store (e.g. in-memory)."""
    with STATE.lock:
        STATE.store, STATE.synced = store, False


def get_store() -> Store:
    with STATE.lock:
        if STATE.store is None:
            STATE.store = Store(DB_PATH)
        return STATE.store


def ensure_synced() -> Store:
    """Register the demo fleet (flights, alerts, summaries) in the database once per process."""
    store = get_store()
    with STATE.lock:
        if not STATE.synced:
            if not STATE.uploads:                       # uploads from an earlier run are gone from memory
                store.drop_source("upload")
            b, out = get_bundle(), demo_results()
            for unit, g in out.groupby("unit"):
                store.sync_engine(b, g, str(int(unit)), "demo")
            STATE.synced = True
    return store


@lru_cache(maxsize=1)
def demo_results() -> pd.DataFrame:
    b = get_bundle()
    if b.demo is None:
        raise HTTPException(404, "Bundle has no demo_cycles.csv")
    return b.run(truth_columns(b.demo))


def engine_frame(engine_id: str) -> tuple[pd.DataFrame, str]:
    if engine_id in STATE.uploads:
        return STATE.uploads[engine_id], "upload"
    out = demo_results()
    try:
        g = out[out["unit"] == int(engine_id)]
    except ValueError:
        g = out.iloc[0:0]
    if g.empty:
        raise HTTPException(404, f"Unknown engine {engine_id}")
    return g, "demo"


@app.get("/api/health")
def health():
    try:
        b = get_bundle()
        return {"status": "ok", "bundle_loaded": True, "model": b.info()}
    except HTTPException as exc:
        return {"status": "ok", "bundle_loaded": False, "detail": exc.detail}


@app.get("/api/model")
def model_info():
    return get_bundle().info()


@app.get("/api/engines")
def engines():
    out = demo_results()
    res = []
    for unit, g in out.groupby("unit"):
        res.append({"engine_id": str(int(unit)), "dataset": str(g["dataset"].iloc[0]) if "dataset" in g else None,
                    "unit_local": int(g["unit_local"].iloc[0]) if "unit_local" in g else None,
                    "n_cycles": int(len(g))})
    return res


@app.get("/api/engine/{engine_id}")
def engine(engine_id: str):
    g, source = engine_frame(engine_id)
    return engine_payload(get_bundle(), g, engine_id, source)


@app.get("/api/engine/{engine_id}/explain/{cycle}")
def explain(engine_id: str, cycle: int, target: str = "rul"):
    g, _ = engine_frame(engine_id)
    try:
        return get_bundle().explain(g, cycle, target)
    except KeyError:
        raise HTTPException(404, f"Engine {engine_id} has no cycle {cycle}")
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Run the full pipeline on an uploaded CSV (raw 1 Hz N-CMAPSS columns, or per-flight aggregates)."""
    b = get_bundle()
    try:
        cyc = parse_upload(pd.read_csv(io.BytesIO(await file.read())))
        out = b.run(cyc.drop(columns=[c for c in cyc.columns if c.startswith("hi_")] + ["rul", "hs"], errors="ignore"))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    result = {}
    for unit, g in out.groupby("unit"):
        eid = f"up{len(STATE.uploads) + 1}-{int(unit)}"
        STATE.uploads[eid] = g
        get_store().sync_engine(b, g, eid, "upload")
        result[eid] = engine_payload(b, g, eid, "upload")
    return {"engines": result}


class EventIn(BaseModel):
    event_type: str
    finding: str
    performed_by: str
    cycle: int | None = None
    component: str | None = None
    action: str | None = None
    observed_health: float | None = None
    alert_id: int | None = None
    notes: str | None = None


class AlertIn(BaseModel):
    status: str
    user: str
    note: str | None = None


@app.get("/api/fleet")
def fleet(as_of: float = 1.0):
    """Fleet triage list: ranked by health band, then planning RUL (lower bound of the RUL interval).
    as_of in (0, 1] replays the fleet at that share of each engine's recorded flights."""
    if not 0.0 < as_of <= 1.0:
        raise HTTPException(422, "as_of must be in (0, 1]")
    store, b = ensure_synced(), get_bundle()
    return {"as_of": as_of, "engines": store.fleet(as_of, b.manifest.get("reliability"),
                                                   b.manifest.get("band_edges", [0.75, 0.55, 0.30])), "enums": {"event_types": EVENT_TYPES, "findings": FINDINGS,
                                                          "alert_status": ALERT_STATUS}}


@app.get("/api/engine/{engine_id}/alerts")
def engine_alerts(engine_id: str, status: str | None = None):
    engine_frame(engine_id)
    return ensure_synced().alerts(engine_id, status)


@app.post("/api/alerts/{alert_id}")
def handle_alert(alert_id: int, body: AlertIn):
    try:
        row = ensure_synced().handle_alert(alert_id, body.status, body.user, body.note)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    if row is None:
        raise HTTPException(404, f"Unknown alert {alert_id}")
    return row


@app.get("/api/engine/{engine_id}/maintenance")
def engine_maintenance(engine_id: str):
    engine_frame(engine_id)
    return ensure_synced().events(engine_id)


@app.post("/api/engine/{engine_id}/maintenance", status_code=201)
def add_maintenance(engine_id: str, body: EventIn):
    engine_frame(engine_id)
    try:
        return ensure_synced().add_event(engine_id, body.model_dump())
    except KeyError:
        raise HTTPException(404, f"Unknown engine {engine_id}")
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@lru_cache(maxsize=1)
def physics_report() -> dict:
    from .physics_check import run_checks
    return run_checks(demo_results())


@app.get("/api/physics/check")
def physics_check():
    """Deep physics check of the pipeline on the bundled test engines (conservation laws, second law, monitors)."""
    return physics_report()


@app.get("/api/learning/summary")
def learning_summary():
    return ensure_synced().learning_summary()


@app.get("/api/learning/export.csv")
def learning_export():
    return Response(ensure_synced().export_csv(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=maintenance_findings.csv"})


if os.path.isdir(PUBLIC):
    app.mount("/", StaticFiles(directory=PUBLIC, html=True), name="dashboard")
