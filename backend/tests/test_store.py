"""Maintenance workflow persistence + API (stub bundle, in-memory SQLite)."""
import pytest
from fastapi.testclient import TestClient

from backend import app as appmod
from backend.pipeline.store import Store
from backend.tests.stub_bundle import stub_bundle


@pytest.fixture()
def client():
    appmod.set_bundle(stub_bundle())
    appmod.set_store(Store(":memory:"))
    yield TestClient(appmod.app)
    appmod.set_bundle(None)
    appmod.set_store(None)


def test_fleet_is_ranked_and_complete(client):
    eng = client.get("/api/fleet").json()["engines"]
    assert eng and [e["priority"] for e in eng] == list(range(1, len(eng) + 1))
    key = [(-e["urgency"], e["planning_rul"]) for e in eng]
    assert key == sorted(key)                                     # most urgent first, then lowest planning RUL
    for e in eng:                                                 # a short planning RUL escalates a "Normal" engine
        assert e["urgency"] >= e["band"] and (e["planning_rul"] > 15 or e["urgency"] >= 2)
    for e in eng:
        assert 0 <= e["planning_rul"] <= e["rul_q95"] + 1e-9 and e["planning_rul"] <= e["rul"] + 1e-9
        assert e["localisation"]


def test_payload_has_advisory_and_only_trusted_forecasts(client):
    eid = client.get("/api/engines").json()[0]["engine_id"]
    p = client.get(f"/api/engine/{eid}").json()
    assert p["advisory"]["planning_rul"] <= p["advisory"]["rul"] + 1e-9 and "Advisory" in p["advisory"]["guidance"]
    assert all(f["trust"] in ("confident", "indicative") for f in p["forecast_display"])


def test_alert_workflow_and_audit(client):
    eid = client.get("/api/fleet").json()["engines"][0]["engine_id"]
    alerts = client.get(f"/api/engine/{eid}/alerts").json()
    assert alerts and all(a["status"] == "open" for a in alerts)
    r = client.post(f"/api/alerts/{alerts[0]['id']}", json={"status": "acknowledged", "user": "eng1", "note": "seen"})
    assert r.status_code == 200 and r.json()["handled_by"] == "eng1" and r.json()["handled_utc"]
    assert client.post(f"/api/alerts/{alerts[0]['id']}", json={"status": "bogus", "user": "x"}).status_code == 422
    assert client.post("/api/alerts/999999", json={"status": "dismissed", "user": "x"}).status_code == 404


def test_maintenance_event_roundtrip_and_validation(client):
    eid = client.get("/api/fleet").json()["engines"][0]["engine_id"]
    alert = client.get(f"/api/engine/{eid}/alerts").json()[0]
    ev = {"event_type": "borescope", "finding": "confirmed", "performed_by": "A. Rao", "cycle": alert["cycle"],
          "component": "hpc", "observed_health": 0.6, "alert_id": alert["id"], "notes": "blade erosion"}
    r = client.post(f"/api/engine/{eid}/maintenance", json=ev)
    assert r.status_code == 201 and r.json()["performed_utc"]
    assert client.get(f"/api/engine/{eid}/maintenance").json()[0]["notes"] == "blade erosion"
    assert [a for a in client.get(f"/api/engine/{eid}/alerts").json() if a["id"] == alert["id"]][0]["status"] == "acknowledged"
    for bad in ({"event_type": "nope"}, {"finding": "maybe"}, {"component": "wing"}, {"observed_health": 1.5},
                {"performed_by": " "}):
        assert client.post(f"/api/engine/{eid}/maintenance", json={**ev, **bad}).status_code == 422
    assert client.post("/api/engine/424242/maintenance", json=ev).status_code == 404
    fleet = {e["engine_id"]: e for e in client.get("/api/fleet").json()["engines"]}
    assert fleet[eid]["n_events"] == 1 and fleet[eid]["last_maintenance_utc"]


def test_learning_export_joins_prediction_shown(client):
    eid = client.get("/api/fleet").json()["engines"][0]["engine_id"]
    cyc = client.get(f"/api/engine/{eid}").json()["columns"]["cycle"][-1]
    client.post(f"/api/engine/{eid}/maintenance", json={"event_type": "borescope", "finding": "confirmed", "performed_by": "e",
                                                         "cycle": cyc, "component": "hpc", "observed_health": 0.5})
    client.post(f"/api/engine/{eid}/maintenance", json={"event_type": "sensor_check", "finding": "not_found", "performed_by": "e"})
    s = client.get("/api/learning/summary").json()
    assert s["events"] == 2 and s["flag_precision"] == 0.5 and s["n_health_checks"] == 1 and s["health_estimate_mae"] is not None
    csv = client.get("/api/learning/export.csv").text.splitlines()
    assert len(csv) == 3 and "predicted_rul" in csv[0]
    assert csv[1].split(",")[csv[0].split(",").index("predicted_rul")] != ""


def test_upload_registers_engine(client):
    from backend.tests.stub_bundle import raw_1hz_frame
    import io
    buf = io.StringIO(); raw_1hz_frame().to_csv(buf, index=False)
    r = client.post("/api/analyze", files={"file": ("u.csv", buf.getvalue(), "text/csv")})
    assert r.status_code == 200
    ids = {e["engine_id"] for e in client.get("/api/fleet").json()["engines"]}
    assert set(r.json()["engines"]) <= ids


def test_fleet_replay_as_of(client):
    full = {e["engine_id"]: e for e in client.get("/api/fleet").json()["engines"]}
    early = client.get("/api/fleet?as_of=0.3").json()["engines"]
    assert len(early) == len(full)
    assert all(e["last_cycle"] <= full[e["engine_id"]]["last_cycle"] for e in early)
    assert [e["priority"] for e in early] == list(range(1, len(early) + 1))
    assert client.get("/api/fleet?as_of=0").status_code == 422 and client.get("/api/fleet?as_of=1.5").status_code == 422


def test_old_schema_is_migrated_and_engineer_records_kept(tmp_path):
    import sqlite3
    path = str(tmp_path / "old.db")
    db = sqlite3.connect(path)
    db.executescript("CREATE TABLE engines (engine_id TEXT PRIMARY KEY, source TEXT);"
                     "CREATE TABLE maintenance_events (id INTEGER PRIMARY KEY AUTOINCREMENT, engine_id TEXT NOT NULL, "
                     "cycle INTEGER, component TEXT, event_type TEXT NOT NULL, finding TEXT NOT NULL, action TEXT, "
                     "observed_health REAL, alert_id INTEGER, performed_by TEXT NOT NULL, performed_utc TEXT NOT NULL, notes TEXT);"
                     "INSERT INTO maintenance_events (engine_id, event_type, finding, performed_by, performed_utc) "
                     "VALUES ('1', 'borescope', 'confirmed', 'x', '2026-01-01');")
    db.commit(); db.close()
    s = Store(path)
    assert len(s.events("1")) == 1                                       # engineer input survives the upgrade
    cols = [r[1] for r in s.db.execute("PRAGMA table_info(engines)")]
    assert "urgency" in cols                                             # derived table rebuilt with the new schema


def test_stale_uploads_are_dropped_on_restart():
    from backend.tests.stub_bundle import raw_1hz_frame
    import io
    store = Store(":memory:")
    appmod.set_bundle(stub_bundle()); appmod.set_store(store)
    c = TestClient(appmod.app)
    buf = io.StringIO(); raw_1hz_frame().to_csv(buf, index=False)
    up = list(c.post("/api/analyze", files={"file": ("u.csv", buf.getvalue(), "text/csv")}).json()["engines"])
    assert set(up) <= {e["engine_id"] for e in c.get("/api/fleet").json()["engines"]}
    appmod.set_bundle(stub_bundle())                     # simulates a restart: uploads gone, same database
    appmod.set_store(store)
    ids = {e["engine_id"] for e in c.get("/api/fleet").json()["engines"]}
    assert not set(up) & ids
    appmod.set_bundle(None); appmod.set_store(None)


def test_sample_csvs_upload_and_analyse():
    """The real N-CMAPSS sample files in samples/ go through /api/analyze (stub model, real data format)."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2] / "samples"
    files = sorted(root.glob("test_flights_*.csv"))
    assert files, "sample CSVs missing"
    appmod.set_bundle(stub_bundle()); appmod.set_store(Store(":memory:"))
    c = TestClient(appmod.app)
    for f in files:
        r = c.post("/api/analyze", files={"file": (f.name, f.read_bytes(), "text/csv")})
        assert r.status_code == 200, (f.name, r.text[:200])
        eng = r.json()["engines"]
        assert len(eng) == 1
        p = next(iter(eng.values()))
        assert p["source"] == "upload" and p["n_cycles"] >= 40 and "truth_rul" not in p["columns"]
    appmod.set_bundle(None); appmod.set_store(None)
