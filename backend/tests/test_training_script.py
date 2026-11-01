"""Plumbing test of the training script's Layer 1 and engine-split cells (file search, schema validation,
flight-end aggregation, Drive-style cache, train/validation/test split).

It runs the REAL cell source from training/ncmapss_colab_training.py on a tiny synthetic .h5 file. No model is fitted and
no result is produced - only data handling is checked, so the first Colab run is less likely to fail at the start.
"""
import gc
import glob
import hashlib
import json
import os
import re
import shutil
import time
import zlib

import numpy as np
import pandas as pd
import pytest

from backend.pipeline import ncmapss as nc
from backend.pipeline import stacking as st

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "training", "ncmapss_colab_training.py")
FNAME = "N-CMAPSS_DS02-006.h5"


def _read():
    with open(SCRIPT, encoding="utf-8") as fh:
        return fh.read()


def _cells():
    """[(header, body)] for every '# %%' cell, in order."""
    return [(c.partition("\n")[0].strip(), c.partition("\n")[2]) for c in re.split(r"^# %%", _read(), flags=re.M)[1:]]


def _cell_with(marker):
    for _, body in _cells():
        if marker in body:
            return body
    raise AssertionError(f"cell with {marker!r} not found")


def _frame(n_units, unit_offset, n_cycles=6, n_sec=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for c in range(1, n_cycles + 1):
            for _ in range(n_sec):
                w = [rng.uniform(5000, 35000), rng.uniform(0.2, 0.8), rng.uniform(40, 90), rng.uniform(470, 530)]
                x = [600 + rng.normal(), 1500 + rng.normal(), 1800, 1300, 20, 14, 18, 25, 300, 280, 22, 2000,
                     8500 + rng.normal(), 2.0 + 0.01 * rng.normal()]
                t = list(-0.001 * c * np.arange(1, 11) / 10)
                rows.append(w + x + t + [unit_offset + u, c, 1 + c % 3, 1 if c < 3 else 0, n_cycles - c])
    cols = nc.W_VARS + nc.XS_VARS + nc.T_VARS + ["unit", "cycle", "Fc", "hs", "RUL"]
    return pd.DataFrame(rows, columns=cols)


def _write(path, dev, test):
    import h5py
    groups = {"W": nc.W_VARS, "X_s": nc.XS_VARS, "T": nc.T_VARS, "A": ["unit", "cycle", "Fc", "hs"]}
    with h5py.File(path, "w") as f:
        for split, df in (("dev", dev), ("test", test)):
            for g, names in groups.items():
                f[f"{g}_{split}"] = df[names].to_numpy()
            f[f"Y_{split}"] = df[["RUL"]].to_numpy()
            f[f"X_v_{split}"] = np.column_stack([df["T30"] + 900, df["Ps30"] * 1.05, df["P50"] * 2])
        for g, names in groups.items():
            f[f"{g}_var"] = np.array(names, dtype="S")
        f["X_v_var"] = np.array(["T40", "P30", "P45"], dtype="S")


def _namespace(data_dir, cache_dir):
    return dict(os=os, glob=glob, json=json, hashlib=hashlib, zlib=zlib, time=time, gc=gc, shutil=shutil, np=np, pd=pd,
                nc=nc, st=st, SEED=42, VAL_FRACTION=0.2, N_FOLDS=5, HEALTHY_SAMPLES_PER_FILE=1000, DATA_DIR=str(data_dir),
                DRIVE_DIR=None, DRIVE_OK=False, CACHE_DIR=str(cache_dir), CACHE_AGGREGATES=True, DELETE_AFTER_USE=False,
                DATASET_HANDLE="x/y", FILES=[FNAME], REMOTE={}, EXPECTED_BYTES={}, T_START=time.time(), TIMINGS={},
                stage=lambda name: None)


def test_layer1_and_split_then_reuse_the_cache(tmp_path):
    pytest.importorskip("h5py")
    pytest.importorskip("pyarrow")
    data, cache = tmp_path / "data", tmp_path / "cache"
    data.mkdir()
    cache.mkdir()
    _write(data / FNAME, _frame(3, 1), _frame(2, 10, seed=1))
    layer1, split = _cell_with('stage("Layer 1: download + aggregate")'), _cell_with('stage("Engine split")')

    first = _namespace(data, cache)
    exec(compile(layer1, "layer1", "exec"), first)
    cyc = first["cyc"]
    assert first["FINGERPRINTS"][FNAME]["bytes"] == os.path.getsize(data / FNAME)
    assert set(first["SCHEMA"][FNAME]["splits"]) == {"dev", "test"}
    assert first["SCHEMA"][FNAME]["inspect"]["datasets"]["W_dev"]["shape"][1] == 4
    assert cyc["unit"].nunique() == 5 and len(cyc) == 5 * 6 and set(cyc.split) == {"dev", "test"}
    assert (cyc["unit"] // 1000 == 1).all() and (cyc["unit"] % 1000 == cyc["unit_local"]).all()   # composite id
    assert {"mn_T48", "mx_T48", "ls_T48", "sd_T48", "m_T48"} <= set(cyc.columns)
    assert first["healthy_w"][0].shape[1] == 4 and set(first["healthy_u"][0] // 1000) == {1}
    assert any(f.endswith(".meta.json") for f in os.listdir(cache))

    exec(compile(split, "split", "exec"), first)
    parts = first["cyc"].drop_duplicates("unit").set_index("unit")["partition"]
    assert set(parts[parts == "test"].index % 1000) == {10, 11}            # official test engines stay test
    assert (parts == "val").sum() == 1 and (parts == "train").sum() == 2
    for tr, te in first["folds"]:
        g = first["groups"][first["tr_idx"]]
        assert not set(g[tr]) & set(g[te])

    os.remove(data / FNAME)                                    # second run must not need the data file at all
    second = _namespace(data, cache)
    exec(compile(layer1, "layer1", "exec"), second)
    pd.testing.assert_frame_equal(first["cyc"].drop(columns="partition"), second["cyc"])
    assert np.allclose(first["healthy_w"][0], second["healthy_w"][0])


def test_script_cells_are_well_formed():
    headers = [h for h, _ in _cells()]
    assert headers.count("[pipeline:ncmapss]") == 1 and headers.count("[pipeline:stacking]") == 1
    assert headers.count("") >= 15 and headers.count("[markdown]") >= 15
    text = _read()
    assert re.search(r"^BUILD_INFO = \{.*\}", text, flags=re.M)         # build_notebook.py stamps this line
    compile(text, SCRIPT, "exec")


def test_forecast_cell_runs_with_real_lightgbm_and_round_trips_through_a_saved_booster(tmp_path):
    """Plumbing only: runs the notebook's real forecasting cell on tiny synthetic data (no result is kept) and checks
    that a model saved with booster_.save_model and reloaded as a Booster - what the backend does - predicts the same."""
    pytest.importorskip("lightgbm")
    from backend.pipeline.bundle import load_model
    rng = np.random.default_rng(0)
    rows = []
    for u in range(1, 13):
        n = 70
        decay = np.linspace(1.0, rng.uniform(0.3, 0.9), n)
        for i in range(n):
            rows.append({"unit": u, "cycle": i + 1, **{f"hi_{c}": float(decay[i]) for c in nc.COMPONENTS},
                         **{f"kf_hi_{c}": float(decay[i] + 0.01 * rng.normal()) for c in nc.COMPONENTS}})
    cyc = pd.DataFrame(rows)
    FEATURES = ["f0", "f1", "f2", "f3"]
    X = pd.DataFrame(rng.normal(size=(len(cyc), 4)), columns=FEATURES)
    X["f0"] = cyc["kf_hi_hpt"]
    groups = cyc["unit"].to_numpy()
    tr_idx, va_idx, te_idx = (np.flatnonzero(np.isin(groups, u)) for u in (range(1, 9), range(9, 11), range(11, 13)))
    ns = dict(np=np, pd=pd, time=time, nc=nc, st=st, stage=lambda name: None, RUN_FORECAST=True,
              FORECAST_HORIZONS=(5, 10), FORECAST_DEGRADED_BELOW=0.9, X=X, cyc=cyc, FEATURES=FEATURES, groups=groups,
              tr_idx=tr_idx, va_idx=va_idx, te_idx=te_idx, ROLL=3, INTERVAL_LEVEL=0.9, RELIABLE_MIN_CYCLES=5,
              PARAMS={"lightgbm": {"n_estimators": 40, "num_leaves": 7}}, TIMINGS={}, metrics={},
              RELIABILITY={c: {"status": "ok"} for c in nc.COMPONENTS})
    exec(compile(_cell_with('stage("Forecasting")'), "forecast", "exec"), ns)
    fc = ns["FORECAST"]
    assert set(fc["models"]) == {(c, h) for c in nc.COMPONENTS for h in (5, 10)}
    assert fc["features"] == FEATURES + [f"kf_hi_{c}" for c in nc.COMPONENTS]
    for (c, h), pred in fc["pred_te"].items():
        assert len(pred) == len(te_idx) and np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
        assert fc["intervals"][c][str(h)]["low"] <= fc["intervals"][c][str(h)]["high"]
        assert isinstance(fc["useful"][c][str(h)], bool)
    met = fc["metrics"]["hpt"]["5"]["test"]["degraded"]
    assert met["n"] > 0 and met["skill_vs_persistence"] is not None and "forecast" in ns["metrics"]
    # the training rows without a label h flights ahead (the last h flights of each engine) must have been excluded
    assert ns["metrics"]["forecast"]["hpt"]["10"]["test"]["all"]["n"] == len(te_idx) - 2 * 10

    model = fc["models"][("hpt", 5)]
    path = str(tmp_path / "forecast_hpt_h5.txt")
    model.booster_.save_model(path)
    booster = load_model("lightgbm", path)
    xf = np.column_stack([X.to_numpy(dtype=np.float64)] + [cyc[f"kf_hi_{c}"].to_numpy() for c in nc.COMPONENTS])
    cur = cyc["kf_hi_hpt"].to_numpy()
    backend_style = np.clip(cur[te_idx] + booster.predict(xf[te_idx]), 0, 1)       # exactly what ModelBundle._forecast does
    assert np.allclose(backend_style, fc["pred_te"][("hpt", 5)], atol=1e-9)


def test_rul_pipeline_cell_fits_both_feature_sets_with_real_models_and_ships_the_chosen_one(monkeypatch):
    """Plumbing only: runs the notebook's real section-10 cell (tuning, out-of-fold, Ridge stack for the full and the
    no-EKF feature sets, then the choice) with real learners on tiny random data; nothing is kept."""
    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")
    pytest.importorskip("catboost")
    small = {"randomforest": dict(n_estimators=20, max_features=0.5, min_samples_leaf=2),       # tiny models: speed only
             "xgboost": dict(n_estimators=40, max_depth=3, learning_rate=0.1),
             "lightgbm": dict(n_estimators=40, num_leaves=7, learning_rate=0.1),
             "catboost": dict(iterations=40, depth=3, learning_rate=0.1)}
    for k_, v_ in small.items():
        monkeypatch.setitem(st.SPEC_PARAMS, k_, v_)
    rng = np.random.default_rng(0)
    t, s_ = nc.default_feature_columns(2)
    names = nc.CausalFeaturizer(t, s_).feature_names
    n, n_units = 120, 12
    groups = np.repeat(np.arange(1, n_units + 1), n // n_units)
    X = pd.DataFrame(rng.normal(size=(n, len(names))), columns=names)
    y = (X["res_T48"] * 5 + 40 + rng.normal(size=n)).to_numpy()
    tr, va, te = (np.flatnonzero(np.isin(groups, u)) for u in (range(1, 9), range(9, 11), range(11, 13)))
    folds = st.engine_folds(groups[tr], 3)
    for setting, expect in ((False, "no_ekf"),):          # one run only (real models are slow); the rule itself is unit-tested
        ns = dict(np=np, pd=pd, time=time, nc=nc, st=st, stage=lambda name: None, FEATURES=names, Xtr=X.iloc[tr],
                  Xva=X.iloc[va], Xte=X.iloc[te], ytr=y[tr], yva=y[va], yte=y[te], folds=folds, TUNE=False,
                  TIMINGS={}, EKF_IN_RUL=setting, EKF_SELECT_MARGIN=0.02, EKF_SELECT_MIN_VAL_ENGINES=5,
                  val_units=np.array([9, 10]))
        exec(compile(_cell_with('stage("Base models + Ridge stack")'), "sec10", "exec"), ns)
        assert ns["CHOICE"] == expect and set(ns["RUL_RUNS"]) == {"full", "no_ekf"}
        assert ns["RUL_FEATURES"] == ns["FEATURE_SETS"][expect]
        assert len(ns["FEATURE_SETS"]["no_ekf"]) < len(ns["FEATURE_SETS"]["full"])
        assert not any(nc.feature_group(f) in ("ekf", "innov") for f in ns["FEATURE_SETS"]["no_ekf"])
        assert all(np.isfinite(r["pred_te"]).all() for r in ns["RUL_RUNS"].values())
        assert ns["models"] is ns["RUL_RUNS"][expect]["models"] and ns["stack"] is ns["RUL_RUNS"][expect]["stack"]
        assert ns["models"]["xgboost"].n_features_in_ == len(ns["RUL_FEATURES"])      # models really saw only that set


def test_component_rul_cell_censors_trains_and_round_trips_with_real_lightgbm(tmp_path):
    """Plumbing only: runs the notebook's real component-RUL cell with real LightGBM on tiny random data (nothing is
    kept) and checks censoring, the useful flag, and that a saved booster reloaded like the backend predicts the same."""
    pytest.importorskip("lightgbm")
    from backend.pipeline.bundle import load_model
    rng = np.random.default_rng(1)
    rows = []
    for u in range(1, 15):
        n = 60
        floor = 0.1 if u % 3 else 0.6                       # every third engine's HPT never reaches 0.30 (censored)
        hpt = np.linspace(1.0, floor, n)
        for i in range(n):
            rows.append({"unit": u, "cycle": i + 1, **{f"hi_{c}": 1.0 for c in nc.COMPONENTS if c != "hpt"},
                         "hi_hpt": float(hpt[i]), **{f"kf_hi_{c}": 1.0 for c in nc.COMPONENTS if c != "hpt"},
                         "kf_hi_hpt": float(hpt[i] + 0.01 * rng.normal())})
    cyc = pd.DataFrame(rows)
    groups = cyc["unit"].to_numpy()
    names = ["f0", "f1", "f2"]
    X = pd.DataFrame(rng.normal(size=(len(cyc), 3)), columns=names)
    X["f0"] = cyc["kf_hi_hpt"]
    tr, va, te = (np.flatnonzero(np.isin(groups, u)) for u in (range(1, 10), range(10, 13), range(13, 15)))
    ns = dict(np=np, pd=pd, time=time, nc=nc, st=st, stage=lambda name: None, RUN_COMPONENT_RUL=True, COMP_FAIL_HI=0.30,
              COMP_RUL_CAP=100.0, X=X, cyc=cyc, FEATURES=names, groups=groups, tr_idx=tr, va_idx=va, te_idx=te,
              pred_va=np.full(len(va), 30.0), pred_te=np.full(len(te), 30.0), ROLL=3, INTERVAL_LEVEL=0.9,
              RELIABLE_MIN_CYCLES=5, PARAMS={"lightgbm": {"n_estimators": 40, "num_leaves": 7}}, TIMINGS={}, metrics={},
              RELIABILITY={c: {"status": "ok"} for c in nc.COMPONENTS})
    exec(compile(_cell_with('stage("Component RUL")'), "crul", "exec"), ns)
    cr = ns["COMPONENT_RUL"]
    assert set(cr["models"]) == {"hpt"}                    # other components never degrade -> no model
    assert cr["features"] == names + [f"kf_hi_{c}" for c in nc.COMPONENTS]
    eng = ns["metrics"]["component_rul"]["hpt"]["engines_reaching_level"]
    assert eng == {"train": 6, "validation": 2, "test": 2}  # engines 3, 6, 9, 12 are censored and excluded
    assert len(cr["pred_te"]["hpt"]) == len(te) and ((cr["pred_te"]["hpt"] >= 0) & (cr["pred_te"]["hpt"] <= 100)).all()
    assert isinstance(cr["useful"]["hpt"], bool) and cr["useful"]["fan"] is False
    model = cr["models"]["hpt"]
    path = str(tmp_path / "component_rul_hpt.txt")
    model.booster_.save_model(path)
    xf = np.column_stack([X.to_numpy(dtype=np.float64)] + [cyc[f"kf_hi_{c}"].to_numpy() for c in nc.COMPONENTS])
    backend_style = np.clip(load_model("lightgbm", path).predict(xf[te]), 0.0, 100.0)
    assert np.allclose(backend_style, cr["pred_te"]["hpt"], atol=1e-9)
