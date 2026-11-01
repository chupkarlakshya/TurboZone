"""Backend tests with a STUB bundle (fixed formulas, hand-set EKF, no trained models - see stub_bundle.py)."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import app as appmod
from backend.pipeline import ncmapss as nc
from backend.pipeline.bundle import ModelBundle
from backend.pipeline.service import engine_payload, truth_columns
from backend.tests.stub_bundle import raw_1hz_frame, stub_bundle


@pytest.fixture()
def bundle():
    return stub_bundle()


def test_run_produces_all_layers(bundle):
    out = bundle.run(truth_columns(bundle.demo))
    for t in bundle.targets:
        assert out[t].notna().all()
        assert (out[t] >= 0).all() and (t == "rul" or (out[t] <= 1).all())
    assert {"res_T48", "ref_T48", "ekf_0", "ekf_0_sigma", "kf_HPT_eff_mod", "kf_hi_hpt", "innov_T48", "nis_T48",
            "gated_T48", "nis_total", "ekf_min_eig", "rul_q05", "base_xgboost", "contrib_catboost"} <= set(out.columns)
    assert (out["rul_q05"] <= out["rul_q95"]).all() and (out["ekf_min_eig"] > -1e-12).all()   # covariance stays PSD
    assert np.allclose(out["hi_hpt"], out["kf_hi_hpt"])                  # health = EKF estimate, not an ML target
    assert "truth_hi_hpt" in out and "truth_rul" in out                 # labels kept apart from predictions
    contrib = out[[c for c in out.columns if c.startswith("contrib_")]].sum(axis=1)
    assert np.allclose(contrib + bundle.stack.intercept, out["rul_raw"])   # Ridge contributions add up


def test_payload_is_json_and_has_dashboard_and_spec_fields(bundle):
    out = bundle.run(truth_columns(bundle.demo))
    p = engine_payload(bundle, out[out.unit == 1001], "1001", "demo")
    s = json.dumps(p)                                           # strict JSON (no NaN)
    assert "NaN" not in s
    cols = p["columns"]
    for k in ("alt_m", "x_T48", "ideal_T48", "res_T48", "kf_hi_hpt", "hi_hpt", "rul", "ohi", "v_T40", "eta_turb_est",
              "base_randomforest", "contrib_lightgbm", "rul_spread"):
        assert k in cols and len(cols[k]) == p["n_cycles"], k
    assert p["status"] in {"healthy", "warning", "degraded"} and p["model_version"] and p["prediction_timestamp"]
    assert any(a["type"] == "rul" for a in p["alerts"])
    assert set(p["constraint_checks"]) >= {"health_raw_within_0_1", "monotonic_degradation"}
    assert p["model"]["base_models"] == ["randomforest", "xgboost", "lightgbm", "catboost"]


def test_explain_rul_and_health(bundle):
    out = bundle.run(truth_columns(bundle.demo))
    e = out[out.unit == 1001]
    ex = bundle.explain(e, 35, "rul")
    assert ex["prediction"] == pytest.approx(float(e.loc[e.cycle == 35, "rul"].iloc[0]))   # same as the main run
    assert len(ex["occlusion"]["sensors"]) == len(nc.XS_VARS)
    assert all(np.isfinite(r["value"]) for r in ex["occlusion"]["sensors"])
    assert ex["tree_shap"] is None                              # stubs are not trees -> reported as unavailable
    assert [m["name"] for m in ex["contributions"]["models"]] == bundle.stack.names
    hx = bundle.explain(e, 35, "hi_hpt")
    assert hx["source"].startswith("EKF") and hx["contributions"] is None
    assert hx["prediction"] == pytest.approx(float(e.loc[e.cycle == 35, "kf_hi_hpt"].iloc[0]))
    with pytest.raises(ValueError):
        bundle.explain(e, 35, "nope")


def test_api_with_bundle_and_uploads(bundle):
    appmod.set_bundle(bundle)
    c = TestClient(appmod.app)
    h = c.get("/api/health").json()
    assert h["bundle_loaded"] and h["model"]["dataset"].startswith("STUB")
    eng = c.get("/api/engines").json()
    assert [e["engine_id"] for e in eng] == ["1001", "1002"] and eng[0]["n_cycles"] == 40
    p = c.get("/api/engine/1001").json()
    assert p["n_cycles"] == 40 and p["source"] == "demo"
    assert c.get("/api/engine/9999").status_code == 404
    assert c.get("/api/engine/1001/explain/20?target=hi_hpt").json()["target"] == "hi_hpt"
    assert c.get("/api/engine/1001/explain/999").status_code == 404
    assert c.get("/api/engine/1001/explain/20?target=zzz").status_code == 422
    raw = raw_1hz_frame()
    r = c.post("/api/analyze", files={"file": ("f.csv", raw.to_csv(index=False), "text/csv")})
    assert r.status_code == 200, r.text
    eid = next(iter(r.json()["engines"]))
    assert r.json()["engines"][eid]["n_cycles"] == 12
    assert c.get(f"/api/engine/{eid}/explain/12").status_code == 200
    bad = c.post("/api/analyze", files={"file": ("f.csv", "a,b\n1,2\n", "text/csv")})
    assert bad.status_code == 422


def test_api_without_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(appmod, "MODEL_DIR", str(tmp_path))
    appmod.set_bundle(None)
    c = TestClient(appmod.app)
    assert c.get("/api/health").json()["bundle_loaded"] is False
    r = c.get("/api/engines")
    assert r.status_code == 503 and "unzip the bundle" in r.json()["detail"]
    appmod.set_bundle(None)


def test_bundle_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        ModelBundle.load(str(tmp_path))
    (tmp_path / "manifest.json").write_text(json.dumps({"version": 3}))
    with pytest.raises(ValueError, match="retrain"):                      # old (pre-pipeline.md) bundles are refused
        ModelBundle.load(str(tmp_path))


def test_health_forecast_columns_intervals_payload_and_info(bundle):
    out = bundle.run(truth_columns(bundle.demo))
    for c in nc.COMPONENTS:
        for h in (10, 25):
            cur = out[f"kf_hi_{c}"].to_numpy()
            assert np.allclose(out[f"fc_{c}_h{h}"], np.clip(cur - 0.002 * h, 0, 1))        # current + learned change
            assert np.allclose(out[f"fc_{c}_h{h}_lo"], np.clip(out[f"fc_{c}_h{h}"] - 0.05, 0, 1))
            assert (out[f"fc_{c}_h{h}_hi"] >= out[f"fc_{c}_h{h}"]).all()
    p = engine_payload(bundle, out[out.unit == 1001], "1001", "demo")
    json.dumps(p)
    assert "fc_hpt_h25" in p["columns"] and len(p["columns"]["fc_hpt_h25"]) == p["n_cycles"]
    assert p["model"]["forecast"]["horizons"] == [10, 25] and p["model"]["forecast"]["useful"]["hpt"]["10"] is True
    ex = bundle.explain(out[out.unit == 1001], 35, "rul")                                   # occlusion skips forecasts
    assert np.isfinite(ex["prediction"])


def test_bundle_without_forecast_still_works():
    b = stub_bundle()
    b.forecast, b.forecast_models = {}, {}
    out = b.run(truth_columns(b.demo))
    assert not [c for c in out.columns if c.startswith("fc_")] and b.info()["forecast"] is None


class _Recorder:
    """Stub model that remembers which columns it was asked to predict on."""
    seen = []

    def predict(self, x):
        _Recorder.seen.append(list(x.columns))
        return np.full(len(x), 30.0)


def test_models_use_only_the_manifest_model_features(bundle):
    full = bundle.features
    subset = [f for f in full if nc.feature_group(f) not in ("ekf", "innov")]
    assert 0 < len(subset) < len(full)
    bundle.manifest["model_features"] = subset
    bundle.model_features = subset
    bundle.models = {n: _Recorder() for n in bundle.stack.names}
    _Recorder.seen = []
    out = bundle.run(truth_columns(bundle.demo))
    assert _Recorder.seen and all(cols == subset for cols in _Recorder.seen)       # never saw EKF columns
    assert np.isfinite(out["rul"]).all()
    json.dumps(engine_payload(bundle, out[out.unit == 1001], "1001", "demo"))
    assert stub_bundle().model_features == stub_bundle().features                  # default: all features


def test_component_rul_columns_cap_interval_payload_and_info(bundle):
    out = bundle.run(truth_columns(bundle.demo))
    assert np.allclose(out["crul_hpt"], 17.0) and np.allclose(out["crul_hpt_lo"], 13.0) and np.allclose(out["crul_hpt_hi"], 23.0)
    assert np.allclose(out["crul_lpt"], 100.0) and "crul_lpt_lo" not in out          # capped; no interval -> no bounds
    assert "crul_fan" not in out                                                      # no model for that component
    p = engine_payload(bundle, out[out.unit == 1001], "1001", "demo")
    json.dumps(p)
    assert len(p["columns"]["crul_hpt"]) == p["n_cycles"]
    cr = p["model"]["component_rul"]
    assert cr["level"] == 0.30 and cr["show_below"] == 0.9 and cr["useful"]["hpt"] is True
    ex = bundle.explain(out[out.unit == 1001], 35, "rul")                              # occlusion path skips extensions
    assert np.isfinite(ex["prediction"])


def test_bundle_without_component_rul_still_works():
    b = stub_bundle()
    b.comp_rul, b.comp_rul_models = {}, {}
    out = b.run(truth_columns(b.demo))
    assert not [c for c in out.columns if c.startswith("crul_")] and b.info()["component_rul"] is None
