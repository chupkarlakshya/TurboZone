"""Backend tests with a STUB bundle (fixed formulas, no trained models - see stub_bundle.py)."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import app as appmod
from backend.pipeline import ncmapss as nc
from backend.pipeline.bundle import ModelBundle, OnnxCNN
from backend.pipeline.service import engine_payload, truth_columns
from backend.tests.stub_bundle import TARGETS, raw_1hz_frame, stub_bundle


@pytest.fixture()
def bundle():
    return stub_bundle()


def test_run_produces_all_layers(bundle):
    out = bundle.run(truth_columns(bundle.demo), bundle.demo_samples)
    for t in TARGETS:
        assert out[t].notna().all()
        assert (out[t] >= 0).all() and (t == "rul" or (out[t] <= 1).all())
    assert {"res_T48", "ref_T48", "kf_HPT_eff_mod", "kf_hi_hpt", "nis_T48", "rul_q05"} <= set(out.columns)
    assert (out["rul_q05"] <= out["rul_q95"]).all()
    assert "truth_hi_hpt" in out and "hi_hpt" in out          # labels kept apart from predictions


def test_cnn_unavailable_falls_back_to_tree_only_blend():
    b = stub_bundle(with_cnn=False)
    out = b.run(truth_columns(b.demo), b.demo_samples)
    assert np.allclose(out["rul"], out["rul_trees_only"])
    b = stub_bundle()
    no_samples = b.run(truth_columns(b.demo))                   # e.g. a per-cycle upload: CNN cannot run
    assert np.allclose(no_samples["rul"], no_samples["rul_trees_only"])
    with_cnn = b.run(truth_columns(b.demo), b.demo_samples)
    assert not np.allclose(with_cnn["rul"], with_cnn["rul_trees_only"])
    part = b.demo_samples[b.demo_samples["cycle"] > 5]          # cycles without samples -> tree-only for those
    mixed = b.run(truth_columns(b.demo), part)
    early = mixed["cycle"] <= 5
    assert np.allclose(mixed.loc[early, "rul_raw"], mixed.loc[early, "rul_trees_only"])
    assert np.allclose(mixed.loc[~early, "rul"], with_cnn.loc[~early, "rul"])


def test_smoothing_is_causal_and_explain_matches_run():
    b = stub_bundle(smoothing={"rul": 0.3, "hi_hpt": 0.5})
    out = b.run(truth_columns(b.demo), b.demo_samples)
    assert np.allclose(out["rul"], np.clip(out["rul_raw"], 0, None))     # exact -1/cycle trend passes unchanged
    assert not np.allclose(out["hi_hpt"], np.clip(out["hi_hpt_raw"], 0, 1))
    e = out[out.unit == 1001]
    ex = b.explain(e, 20, "rul", b.samples_for(1001))
    assert ex["prediction"] == pytest.approx(float(e.loc[e.cycle == 20, "rul"].iloc[0]))


def test_payload_is_json_and_has_dashboard_fields(bundle):
    out = bundle.run(truth_columns(bundle.demo))
    p = engine_payload(bundle, out[out.unit == 1001], "1001", "demo")
    assert p["model"]["cnn_used"] is True
    s = json.dumps(p)                                           # strict JSON (no NaN)
    assert "NaN" not in s
    cols = p["columns"]
    for k in ("alt_m", "x_T48", "ideal_T48", "res_T48", "kf_hi_hpt", "hi_hpt", "rul", "ohi", "v_T40", "eta_turb_est"):
        assert k in cols and len(cols[k]) == p["n_cycles"], k
    assert any(a["type"] == "rul" for a in p["alerts"])
    assert set(p["constraint_checks"]) >= {"health_raw_within_0_1", "monotonic_degradation"}


def test_explain_occlusion_covers_every_sensor(bundle):
    out = bundle.run(truth_columns(bundle.demo), bundle.demo_samples)
    e = out[out.unit == 1001]
    ex = bundle.explain(e, 35, "rul", bundle.samples_for(1001))
    assert ex["target"] == "rul" and len(ex["occlusion"]["sensors"]) == len(nc.XS_VARS)
    assert all(np.isfinite(r["value"]) for r in ex["occlusion"]["sensors"])
    assert any(r["sensor"] == "T48" and r["value"] != 0 for r in ex["occlusion"]["sensors"])   # CNN input occluded
    assert ex["tree_shap"] is None                              # stubs are not trees -> reported as unavailable
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
    (tmp_path / "manifest.json").write_text(json.dumps({"version": 2}))
    with pytest.raises(ValueError, match="version"):
        ModelBundle.load(str(tmp_path))


def test_onnx_cnn_wrapper_with_untrained_network(tmp_path):
    """Random-init (untrained) network, only to check the ONNX + 0.1 Hz encoder plumbing end to end."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    enc = nc.SampleEncoder()
    enc.mu, enc.sd = np.zeros(len(enc.channels)), np.ones(len(enc.channels))
    n_in, n_out, win = len(enc.channels), len(TARGETS), 5

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Sequential(torch.nn.ConstantPad1d((1, 1), 0.0), torch.nn.Conv1d(n_in, 1, 3))
            self.head = torch.nn.Linear(win, n_out)

        def forward(self, x):
            return self.head(self.conv(x.transpose(1, 2)).flatten(1))

    path = str(tmp_path / "cnn.onnx")
    kw = dict(input_names=["window"], output_names=["targets"],
              dynamic_axes={"window": {0: "batch"}, "targets": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(Net().eval(), torch.zeros(1, win, n_in), path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(Net().eval(), torch.zeros(1, win, n_in), path, **kw)
    b = stub_bundle()
    df = nc.run_layers_1_to_3(truth_columns(b.demo), b.ref, b.kf, b.d_ref)
    b.cnn = OnnxCNN(path, win, enc, TARGETS)
    cc = b.cnn_cycle(b.demo_samples, df)
    assert cc.shape == (len(df), n_out) and np.isfinite(cc).all()
    out = b.run(truth_columns(b.demo), b.demo_samples)
    assert np.isfinite(out["rul"]).all()


def test_constant_target_component_that_never_degrades():
    b = stub_bundle()
    cm = {"coef": [], "intercept": 1.0}
    b.manifest["targets"]["hi_fan"] = {"constant": 1.0, "base_models": [], "meta": cm, "trees_only_meta": cm}
    b.trees["hi_fan"] = []
    out = b.run(truth_columns(b.demo), b.demo_samples)
    assert (out["hi_fan"] == 1.0).all() and (out["hi_fan_spread"] == 0).all()
    ex = b.explain(out[out.unit == 1001], 20, "hi_fan")
    assert ex["tree_shap"] is None and all(abs(r["value"]) < 1e-12 for r in ex["occlusion"]["sensors"])
    json.dumps(engine_payload(b, out[out.unit == 1001], "1001", "demo"))
