"""Tests for the stacking module with STUB learners (fixed formulas; nothing is learned from synthetic data)."""
import numpy as np
import pandas as pd
import pytest

from backend.pipeline import stacking as st


class Recorder:
    """Stub learner: remembers which rows it was 'trained' on and predicts a fixed function of column 'a'."""
    seen = []

    def __init__(self, name, params=None):
        self.name, self.params = name, params or {}

    def fit(self, x, y, **kw):
        Recorder.seen.append((self.name, set(x.index)))
        return self

    def predict(self, x):
        return np.asarray(x["a"], dtype=float) * self.params.get("k", 1.0)


def test_split_engines_is_by_engine_stratified_and_reproducible():
    units = np.repeat(np.arange(1, 21), 5)
    strata = {u: ("DS01" if u <= 10 else "DS02") for u in range(1, 21)}
    tr, va = st.split_engines(units, strata, val_fraction=0.2, seed=7)
    assert not set(tr) & set(va) and len(tr) + len(va) == 20
    assert {strata[u] for u in va} == {"DS01", "DS02"}                  # every subset in validation
    tr2, va2 = st.split_engines(units, strata, val_fraction=0.2, seed=7)
    assert list(va) == list(va2)


def test_out_of_fold_rows_never_seen_by_their_model():
    df = pd.DataFrame({"a": np.arange(40, dtype=float), "unit": np.repeat(np.arange(8), 5)})
    folds = st.engine_folds(df["unit"].to_numpy(), n_folds=4)
    for tr, te in folds:
        assert not set(df.unit.iloc[tr]) & set(df.unit.iloc[te])
    Recorder.seen = []
    oof = st.oof_predictions(["m1", "m2"], {"m2": {"k": 2.0}}, df[["a"]], np.zeros(40), folds, factory=Recorder)
    assert np.allclose(oof[:, 0], df.a) and np.allclose(oof[:, 1], 2 * df.a)
    for (name, seen), (tr, te) in zip(Recorder.seen, folds * 2):
        assert not seen & set(te)


def test_tune_model_picks_the_best_validation_grid_point():
    x = pd.DataFrame({"a": np.linspace(0, 1, 10)})
    best, table = st.tune_model("lin", x, None, x, 3.0 * x["a"], grid=[{"k": 1.0}, {"k": 3.0}, {"k": 5.0}],
                                factory=Recorder)
    assert best == {"k": 3.0} and len(table) == 3


def test_stack_model_predict_contributions_and_serialisation():
    m = st.StackModel(["a", "b"], mean=[10.0, 20.0], scale=[2.0, 4.0], coef=[3.0, -1.0], intercept=50.0, alpha=1.0)
    base = np.array([[12.0, 24.0]])
    assert m.predict(base)[0] == pytest.approx(50 + 3 * 1 - 1 * 1)
    assert np.allclose(m.contributions(base).sum(axis=1) + m.intercept, m.predict(base))
    assert np.allclose(m.effective_weights(), [1.5, -0.25])
    m2 = st.StackModel.from_dict(m.to_dict())
    assert m2.predict(base)[0] == pytest.approx(m.predict(base)[0])


def test_metrics_and_empirical_interval():
    assert st.rmse([1, 3], [0, 0]) == pytest.approx(np.sqrt(5)) and st.mae([1, -3], [0, 0]) == 2
    s = st.nasa_score([10.0, -13.0], [0.0, 0.0])                       # late (pred > true) penalised more
    assert s[0] == pytest.approx(np.e - 1) and s[1] == pytest.approx(np.e - 1)
    assert st.nasa_score([5.0], [0.0])[0] > st.nasa_score([-5.0], [0.0])[0]
    r = st.report([1, 2, 3], [1, 2, 4])
    assert r["n"] == 3 and r["r2"] < 1
    lo, hi = st.empirical_interval(np.arange(-50, 51, dtype=float), level=0.9)
    assert lo == pytest.approx(-45) and hi == pytest.approx(45)
    rows = st.error_by_group([1, 2, 3, 4], [1, 2, 0, 0], ["x", "x", "y", "y"])
    assert rows[0]["rmse"] == 0 and rows[1]["bias"] == pytest.approx(3.5)
    assert list(st.rul_phase([5, 30, 90])) == ["end_of_life (RUL<=20)", "mid (20<RUL<=60)", "early (RUL>60)"]


def test_spec_starting_configurations():
    assert st.SPEC_PARAMS["randomforest"] == dict(n_estimators=400, max_features=0.7, min_samples_leaf=2)
    assert st.SPEC_PARAMS["xgboost"]["n_estimators"] == 800 and st.SPEC_PARAMS["catboost"]["depth"] == 7
    assert st.BASE_NAMES == ["randomforest", "xgboost", "lightgbm", "catboost"]


def test_future_values_match_by_cycle_and_respect_engine_ends():
    units = np.array([1, 1, 1, 1, 2, 2, 2])
    cycles = np.array([1, 2, 3, 4, 1, 2, 4])                  # engine 2 has a gap (cycle 3 missing)
    v = np.array([10.0, 20, 30, 40, 1, 2, 4])
    f = st.future_values(units, cycles, v, 2)
    assert np.allclose(f[:2], [30, 40]) and np.isnan(f[2:4]).all()   # last 2 flights of engine 1 have no future
    assert np.isnan(f[4]) and np.isnan(f[5])                          # gap: cycle+2 not the row 2 later / past the end
    assert np.isnan(st.future_values(units, cycles, v, 10)).all() and len(st.future_values([], [], [], 3)) == 0


def test_trend_projection_is_causal_and_clipped():
    units = np.array([1] * 6 + [2] * 3)
    cycles = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3])
    v = np.array([1.0, 0.98, 0.96, 0.94, 0.92, 0.90, 0.5, 0.5, 0.5])
    t = st.trend_projection(units, cycles, v, 10, window=5)
    assert t[5] == pytest.approx(0.90 - 0.02 * 10)                    # slope -0.02 per flight
    assert t[0] == 1.0 and np.allclose(t[6:], 0.5)                    # no history -> no change; flat -> flat
    assert st.trend_projection(units, cycles, v, 100)[5] == 0.0       # clipped at 0
    cut = st.trend_projection(units[:4], cycles[:4], v[:4], 10, window=5)
    assert np.allclose(cut, st.trend_projection(units, cycles, v, 10, window=5)[:4])   # causal


def test_forecast_metrics_select_degraded_rows_on_the_present():
    truth = np.array([1.0, 1.0, 0.6, 0.4])
    cur_true = np.array([1.0, 1.0, 0.8, 0.5])
    pred, persist, trend = np.array([1.0, 1.0, 0.6, 0.4]), np.array([1.0, 1.0, 0.8, 0.5]), np.array([1.0, 1.0, 0.7, 0.45])
    m = st.forecast_metrics(pred, truth, persist, trend, cur_true, degraded_below=0.9)
    assert m["all"]["n"] == 4 and m["degraded"]["n"] == 2
    assert m["degraded"]["forecast_rmse"] == pytest.approx(0.0) and m["degraded"]["skill_vs_persistence"] == pytest.approx(1.0)
    assert m["degraded"]["trend_rmse"] < m["degraded"]["persistence_rmse"]
    assert st.forecast_metrics(pred, truth, persist, trend, np.ones(4))["degraded"]["forecast_rmse"] is None


def test_banded_interval_and_apply():
    rng = np.random.default_rng(0)
    pred = np.concatenate([np.full(100, 10.0), np.full(100, 40.0), np.full(10, 90.0)])
    resid = np.concatenate([rng.normal(0, 1, 100), rng.normal(0, 5, 100), rng.normal(0, 30, 10)])
    iv = st.banded_interval(pred, resid, level=0.9, min_n=30)
    b = iv["bands"]
    assert [x["source"] for x in b] == ["band", "band", "pooled"]          # only 10 residuals above 60 -> pooled
    assert (b[1]["high"] - b[1]["low"]) > (b[0]["high"] - b[0]["low"])    # wider where the error is larger
    lo, hi = st.apply_interval(np.array([5.0, 40.0, 100.0]), iv)
    assert lo[0] == max(0.0, 5 + b[0]["low"]) and hi[1] == pytest.approx(40 + b[1]["high"])
    assert hi[2] == pytest.approx(100 + iv["high"])
    old = st.apply_interval(np.array([3.0]), {"low": -5.0, "high": 2.0})    # older bundles: one interval
    assert old[0][0] == 0.0 and old[1][0] == 5.0


def test_cv_stack_predictions_are_engine_held_out_and_fixed_alpha():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(6), 20)
    y = rng.uniform(0, 100, len(groups))
    oof = y[:, None] + rng.normal(0, 3, size=(len(y), 4))
    folds = st.engine_folds(groups, 3)
    cv = st.cv_stack_predictions(oof, y, folds, ["a", "b", "c", "d"], alpha=10.0)
    assert np.isfinite(cv).all() and st.rmse(cv, y) < 5
    assert st.fit_stack(oof, y, ["a", "b", "c", "d"], alpha=7.0).alpha == 7.0


def test_boosted_models_early_stop_on_validation():
    pytest.importorskip("lightgbm")
    pytest.importorskip("catboost")
    rng = np.random.default_rng(2)
    x = pd.DataFrame(rng.normal(size=(300, 5)), columns=list("abcde"))
    xv = pd.DataFrame(rng.normal(size=(100, 5)), columns=list("abcde"))
    for name, key in (("lightgbm", "n_estimators"), ("catboost", "iterations"), ("xgboost", "n_estimators")):
        best, _ = st.tune_model(name, x, x.a * 3, xv, xv.a * 3, grid=[{}])
        assert 1 <= best[key] <= st.SPEC_PARAMS[name][key]                 # number of trees frozen from validation


def test_choose_feature_set_rules():
    val = {"full": 5.0, "no_ekf": 4.0}
    assert st.choose_feature_set(True, val, 1)[0] == "full" and st.choose_feature_set(False, val, 1)[0] == "no_ekf"
    c, why = st.choose_feature_set("auto", val, 1)                       # one validation engine cannot choose
    assert c == "full" and "too few" in why
    assert st.choose_feature_set("auto", val, 10)[0] == "no_ekf"          # clearly better, enough engines
    assert st.choose_feature_set("auto", {"full": 5.0, "no_ekf": 4.95}, 10)[0] == "full"   # within the 2% margin
    assert st.choose_feature_set("auto", {"full": 4.0, "no_ekf": 5.0}, 10)[0] == "full"    # full better
    assert st.choose_feature_set("auto", {"full": 5.0, "no_ekf": 4.95}, 10, margin=0.0)[0] == "no_ekf"
    with pytest.raises(ValueError):
        st.choose_feature_set("sometimes", val, 10)


def test_component_rul_labels_censoring_and_crossing():
    units = np.array([1] * 5 + [2] * 3 + [3] * 2)
    cycles = np.array([1, 2, 3, 4, 5, 1, 2, 3, 1, 2])
    health = np.array([1.0, 0.8, 0.5, 0.29, 0.1,      # engine 1 reaches 0.30 at cycle 4
                       1.0, 0.9, 0.8,                 # engine 2 never does (censored)
                       0.2, 0.1])                     # engine 3 is already below at the start
    lab = st.component_rul_labels(units, cycles, health, 0.30)
    assert np.allclose(lab[:5], [3, 2, 1, 0, 0])                       # 0 at and after the crossing
    assert np.isnan(lab[5:8]).all()
    assert np.allclose(lab[8:], [0, 0])


def test_trend_time_to_level_causal_and_capped():
    units = np.array([1] * 6 + [2] * 4)
    cycles = np.arange(1, 11)
    v = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.9, 0.9, 0.9, 0.9])
    t = st.trend_time_to_level(units, cycles, v, 0.30, window=5, cap=100.0)
    assert t[5] == pytest.approx((0.5 - 0.3) / 0.1)                    # slope -0.1 per flight
    assert np.allclose(t[6:], 100.0) and t[0] == 100.0                 # flat / no history -> cap
    assert st.trend_time_to_level(units, cycles, np.where(units == 1, 0.2, 0.9), 0.30)[0] == 0.0
    cut = st.trend_time_to_level(units[:4], cycles[:4], v[:4], 0.30, window=5)
    assert np.allclose(cut, st.trend_time_to_level(units, cycles, v, 0.30, window=5)[:4])        # causal


def test_component_rul_metrics_skill_against_both_baselines():
    truth = np.array([10.0, 5.0, 0.0])
    m = st.component_rul_metrics(truth, truth, truth + 10, truth + 5, np.array([1, 1, 2]))
    assert m["n"] == 3 and m["n_engines"] == 2 and m["model_rmse"] == 0
    assert m["skill_vs_engine"] == pytest.approx(1.0) and m["skill_vs_trend"] == pytest.approx(1.0)
    empty = st.component_rul_metrics(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
    assert empty["n"] == 0 and empty["model_rmse"] is None and empty["skill_vs_engine"] is None
