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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from .pipeline.bundle import ModelBundle
from .pipeline.service import engine_payload, parse_upload, truth_columns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC = os.path.join(ROOT, "public")
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(ROOT, "backend", "models"))

mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("model/gltf+json", ".gltf")

app = FastAPI(title="GENZONE N-CMAPSS Digital Twin", version="0.3.0")


class State:
    def __init__(self):
        self.bundle: ModelBundle | None = None
        self.error: str | None = None
        self.uploads: dict[str, pd.DataFrame] = {}
        self.upload_samples: dict[str, pd.DataFrame | None] = {}
        self.lock = threading.Lock()


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
        STATE.upload_samples.clear()
    demo_results.cache_clear()


@lru_cache(maxsize=1)
def demo_results() -> pd.DataFrame:
    b = get_bundle()
    if b.demo is None:
        raise HTTPException(404, "Bundle has no demo_cycles.csv")
    return b.run(truth_columns(b.demo), b.demo_samples)


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
    g, source = engine_frame(engine_id)
    b = get_bundle()
    samples = STATE.upload_samples.get(engine_id) if source == "upload" else b.samples_for(engine_id)
    try:
        return b.explain(g, cycle, target, samples)
    except KeyError:
        raise HTTPException(404, f"Engine {engine_id} has no cycle {cycle}")
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Run the full pipeline on an uploaded CSV (raw 1 Hz N-CMAPSS columns, or per-cycle aggregates - the latter
    cannot feed the 0.1 Hz CNN, so the tree-only blend is used)."""
    b = get_bundle()
    try:
        cyc, samples = parse_upload(pd.read_csv(io.BytesIO(await file.read())))
        out = b.run(cyc.drop(columns=[c for c in cyc.columns if c.startswith("hi_")] + ["rul", "hs"], errors="ignore"),
                    samples)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    result = {}
    for unit, g in out.groupby("unit"):
        eid = f"up{len(STATE.uploads) + 1}-{int(unit)}"
        STATE.uploads[eid] = g
        STATE.upload_samples[eid] = None if samples is None else samples[samples["unit"] == unit]
        result[eid] = engine_payload(b, g, eid, "upload")
    return {"engines": result}


if os.path.isdir(PUBLIC):
    app.mount("/", StaticFiles(directory=PUBLIC, html=True), name="dashboard")
