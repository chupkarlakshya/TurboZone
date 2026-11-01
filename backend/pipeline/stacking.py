"""Four tree regressors + Ridge stacking for RUL, exactly as pipeline.md sections 7-10 describe.

* Base models: Random Forest, XGBoost, LightGBM, CatBoost with the spec's starting configurations; small grids tuned
  on VALIDATION engines only; XGBoost uses early stopping on the validation engines.
* Stacking: engine-grouped out-of-fold predictions on the TRAINING engines -> StandardScaler fitted on those OOF
  predictions only -> Ridge (alpha chosen on validation engines) -> base models refitted on the full training partition.
* Metrics: RMSE, MAE, R2 and the PHM08/C-MAPSS NASA score (late predictions penalised more: exp(d/10) - 1 for
  d = pred - true > 0, exp(-d/13) - 1 for d < 0).
* Uncertainty: an EMPIRICAL interval from validation-engine residual quantiles (section 9), clearly labelled.

Shared by the Colab notebook (inlined) and the backend (StackModel only needs numpy at inference).
"""
from __future__ import annotations

import numpy as np

DEFAULT_SEED = 42
BASE_NAMES = ["randomforest", "xgboost", "lightgbm", "catboost"]

SPEC_PARAMS = {        # pipeline.md 7.1-7.4 starting configurations (search starting points, not calibrated optima)
    "randomforest": dict(n_estimators=400, max_features=0.7, min_samples_leaf=2),
    "xgboost": dict(n_estimators=800, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=5.0),
    "lightgbm": dict(n_estimators=800, learning_rate=0.03, num_leaves=31, max_depth=-1, subsample=0.8,
                     colsample_bytree=0.8, reg_lambda=5.0),
    "catboost": dict(iterations=800, depth=7, learning_rate=0.03, l2_leaf_reg=5.0),
}
TUNE_GRIDS = {         # small validation-engine grids around the starting points
    "randomforest": [dict(max_features=f, min_samples_leaf=m) for f in (0.5, 0.7) for m in (2, 5)],
    "xgboost": [dict(max_depth=d) for d in (4, 6, 8)],
    "lightgbm": [dict(num_leaves=n) for n in (15, 31, 63)],
    "catboost": [dict(depth=d) for d in (5, 7)],
}
EARLY_STOPPING_ROUNDS = 50


def make_model(name: str, params: dict | None = None, seed: int = DEFAULT_SEED, n_jobs: int = -1):
    p = {**SPEC_PARAMS[name], **(params or {})}
    if name == "randomforest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**p, n_jobs=n_jobs, random_state=seed)
    if name == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(**p, objective="reg:squarederror", tree_method="hist", n_jobs=n_jobs, random_state=seed)
    if name == "lightgbm":
        import lightgbm as lgb
        # subsample only takes effect with subsample_freq > 0 (LightGBM semantics, see pipeline.md 7.3)
        return lgb.LGBMRegressor(**p, subsample_freq=1, n_jobs=n_jobs, random_state=seed, verbosity=-1)
    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**p, loss_function="RMSE", random_seed=seed, verbose=False, thread_count=n_jobs,
                                 allow_writing_files=False)
    raise ValueError(f"unknown model {name}")


# ----------------------------------------------------------------------------------------------- metrics
def rmse(pred, true) -> float:
    d = np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    return float(np.sqrt(np.mean(d ** 2))) if len(d) else float("nan")


def mae(pred, true) -> float:
    d = np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    return float(np.mean(np.abs(d))) if len(d) else float("nan")


def r2(pred, true) -> float:
    t = np.asarray(true, dtype=np.float64)
    tot = float(np.sum((t - t.mean()) ** 2)) if len(t) else 0.0
    return float(1.0 - np.sum((np.asarray(pred) - t) ** 2) / tot) if tot > 0 else float("nan")


def nasa_score(pred, true) -> np.ndarray:
    d = np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    return np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)


def report(pred, true) -> dict:
    s = nasa_score(pred, true)
    return {"rmse": rmse(pred, true), "mae": mae(pred, true), "r2": r2(pred, true),
            "nasa_score_sum": float(s.sum()), "nasa_score_mean": float(s.mean()) if len(s) else float("nan"),
            "n": int(len(np.asarray(true)))}


# ----------------------------------------------------------------------------------------------- engine splits
def split_engines(units, strata=None, val_fraction: float = 0.2, seed: int = DEFAULT_SEED):
    """Split complete engine trajectories into (train_units, val_units) with a fixed seed. With `strata`
    ({unit: subset}), the split is done per subset so every subset is represented in validation (if it has >= 2)."""
    uniq = np.unique(np.asarray(units))
    rng = np.random.default_rng(seed)
    strata = strata or {u: 0 for u in uniq}
    val = []
    for s in sorted({strata[u] for u in uniq}, key=str):
        members = np.array(sorted(u for u in uniq if strata[u] == s))
        if len(members) < 2:
            continue
        k = max(1, int(round(val_fraction * len(members))))
        val += list(rng.choice(members, size=k, replace=False))
    val = np.array(sorted(val))
    return np.array(sorted(set(uniq) - set(val))), val


def engine_folds(groups, n_folds: int = 5):
    """GroupKFold by engine over the rows of the training partition (no shuffling of correlated samples)."""
    from sklearn.model_selection import GroupKFold
    groups = np.asarray(groups)
    k = min(n_folds, len(np.unique(groups)))
    return list(GroupKFold(k).split(np.zeros(len(groups)), groups=groups))


# ----------------------------------------------------------------------------------------------- tuning + base models
def tune_model(name, x_tr, y_tr, x_val, y_val, grid=None, factory=make_model):
    """Pick the grid point with the lowest validation-engine RMSE. The boosted models (XGBoost, LightGBM, CatBoost)
    early-stop on the validation engines (pipeline.md 7.2-7.4: tune the number of iterations on validation engines);
    the chosen number of trees is frozen into the returned params, so later refits (out-of-fold, final, ablation) use
    it without early stopping. Returns (best_params, table)."""
    grid = grid if grid is not None else TUNE_GRIDS[name]
    table, best, best_score = [], None, np.inf
    for cand in grid:
        params = dict(cand)
        if name == "xgboost":
            m = factory(name, {**params, "early_stopping_rounds": EARLY_STOPPING_ROUNDS})
            m.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)
            params["n_estimators"] = int(m.best_iteration) + 1
        elif name == "lightgbm":
            import lightgbm as lgb
            m = factory(name, params)
            m.fit(x_tr, y_tr, eval_set=[(x_val, y_val)],
                  callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
            params["n_estimators"] = max(1, int(m.best_iteration_ or m.n_estimators))
        elif name == "catboost":
            m = factory(name, {**params, "early_stopping_rounds": EARLY_STOPPING_ROUNDS})
            m.fit(x_tr, y_tr, eval_set=(x_val, y_val))
            params["iterations"] = int(m.get_best_iteration()) + 1
        else:
            m = factory(name, params)
            m.fit(x_tr, y_tr)
        score = rmse(m.predict(x_val), y_val)
        table.append({"model": name, "params": params, "val_rmse": score})
        if score < best_score:
            best, best_score = params, score
    return best, table


def oof_predictions(names, params, x, y, folds, factory=make_model) -> np.ndarray:
    """(n, len(names)) out-of-fold predictions: every row is predicted by models that never saw its engine."""
    out = np.full((len(y), len(names)), np.nan)
    for k, name in enumerate(names):
        for tr, te in folds:
            m = factory(name, params.get(name))
            m.fit(x.iloc[tr] if hasattr(x, "iloc") else x[tr], np.asarray(y)[tr])
            out[te, k] = m.predict(x.iloc[te] if hasattr(x, "iloc") else x[te])
    return out


def fit_base_models(names, params, x, y, factory=make_model) -> dict:
    """Refit each base model on the full training partition."""
    models = {}
    for name in names:
        m = factory(name, params.get(name))
        m.fit(x, np.asarray(y))
        models[name] = m
    return models


def predict_bases(models: dict, names, x) -> np.ndarray:
    return np.column_stack([np.asarray(models[n].predict(x), dtype=np.float64).reshape(-1) for n in names])


# ----------------------------------------------------------------------------------------------- Ridge meta-learner
class StackModel:
    """RUL = intercept + sum_i coef_i * (p_i - mean_i) / scale_i  (scaler fitted on out-of-fold predictions only)."""

    def __init__(self, names, mean, scale, coef, intercept, alpha=None, diagnostics=None):
        self.names = list(names)
        self.mean, self.scale = np.asarray(mean, dtype=np.float64), np.asarray(scale, dtype=np.float64)
        self.coef, self.intercept = np.asarray(coef, dtype=np.float64), float(intercept)
        self.alpha, self.diagnostics = alpha, diagnostics or {}

    def transform(self, base: np.ndarray) -> np.ndarray:
        return (np.asarray(base, dtype=np.float64) - self.mean) / self.scale

    def predict(self, base: np.ndarray) -> np.ndarray:
        return self.transform(base) @ self.coef + self.intercept

    def contributions(self, base: np.ndarray) -> np.ndarray:
        """coefficient_i * scaled_base_prediction_i per row (pipeline.md 11.2); the intercept is separate."""
        return self.transform(base) * self.coef

    def effective_weights(self) -> np.ndarray:
        """d RUL / d p_i (used to weight per-model SHAP values)."""
        return self.coef / self.scale

    def to_dict(self):
        return {"names": self.names, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coef": self.coef.tolist(), "intercept": self.intercept, "alpha": self.alpha,
                "diagnostics": self.diagnostics}

    @classmethod
    def from_dict(cls, d):
        return cls(d["names"], d["mean"], d["scale"], d["coef"], d["intercept"], d.get("alpha"), d.get("diagnostics"))


def fit_stack(oof: np.ndarray, y, names, val_base=None, y_val=None,
              alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0), alpha: float | None = None) -> StackModel:
    """Scaler on the OOF matrix, Ridge on scaled OOF predictions; alpha by validation-engine RMSE (or fixed via
    `alpha`). Coefficients are not constrained to be positive or to sum to one (8.3)."""
    from sklearn.linear_model import Ridge

    y = np.asarray(y, dtype=np.float64)
    mean = oof.mean(axis=0)
    scale = oof.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    z = (oof - mean) / scale
    table = {}
    if alpha is not None:
        pass
    elif val_base is not None and y_val is not None and len(y_val):
        zv = (np.asarray(val_base) - mean) / scale
        for a in alphas:
            table[a] = rmse(Ridge(alpha=a).fit(z, y).predict(zv), y_val)
        alpha = min(table, key=table.get)
    else:
        alpha = 1.0
    m = Ridge(alpha=alpha).fit(z, y)
    resid = y - m.predict(z)
    diag = {"alpha_val_rmse": {str(k): v for k, v in table.items()},
            "base_prediction_correlation": np.corrcoef(oof.T).tolist() if oof.shape[1] > 1 else [[1.0]],
            "oof_rmse_per_base": {n: rmse(oof[:, i], y) for i, n in enumerate(names)},
            "meta_residual_mean": float(resid.mean()), "meta_residual_std": float(resid.std())}
    return StackModel(names, mean, scale, m.coef_, m.intercept_, alpha, diag)


def cv_stack_predictions(oof: np.ndarray, y, folds, names, alpha: float) -> np.ndarray:
    """Engine-held-out predictions of the WHOLE stack on the training engines: for each fold, the scaler + Ridge are
    refitted on the other folds' out-of-fold predictions and applied to this fold. Used as extra (slightly pessimistic,
    because each base model saw fewer engines) residuals for the empirical interval."""
    y = np.asarray(y, dtype=np.float64)
    out = np.full(len(y), np.nan)
    for tr, te in folds:
        out[te] = fit_stack(oof[tr], y[tr], names, alpha=alpha).predict(oof[te])
    return out


# ----------------------------------------------------------------------------------------------- uncertainty
RUL_BAND_EDGES = (20.0, 60.0)      # predicted-RUL ranges of the interval: <= 20, (20, 60], > 60


def banded_interval(pred, residuals, level: float = 0.90, edges=RUL_BAND_EDGES, min_n: int = 30) -> dict:
    """Empirical interval PER PREDICTED-RUL RANGE (pipeline.md 9: error by degradation group): quantiles of
    residuals (true - predicted) of the rows whose prediction falls in each range. Ranges with fewer than `min_n`
    residuals use the pooled quantiles. Bands are defined on the prediction because the true RUL is unknown online."""
    pred, r = np.asarray(pred, dtype=np.float64), np.asarray(residuals, dtype=np.float64)
    low, high = empirical_interval(r, level)
    bounds = [-np.inf, *edges, np.inf]
    bands = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        m = (pred > a) & (pred <= b)
        lo, hi, src = (*empirical_interval(r[m], level), "band") if m.sum() >= min_n else (low, high, "pooled")
        bands.append({"pred_above": None if np.isinf(a) else float(a), "pred_up_to": None if np.isinf(b) else float(b),
                      "low": lo, "high": hi, "n": int(m.sum()), "source": src})
    return {"level": level, "low": low, "high": high, "bands": bands}


def apply_interval(pred, interval: dict):
    """(lower, upper) RUL bounds for each prediction, clipped at 0; uses the band of each prediction if present."""
    pred = np.asarray(pred, dtype=np.float64)
    lo, hi = np.full(len(pred), interval["low"]), np.full(len(pred), interval["high"])
    for band in interval.get("bands") or []:
        a = -np.inf if band["pred_above"] is None else band["pred_above"]
        b = np.inf if band["pred_up_to"] is None else band["pred_up_to"]
        m = (pred > a) & (pred <= b)
        lo[m], hi[m] = band["low"], band["high"]
    return np.clip(pred + lo, 0.0, None), np.clip(pred + hi, 0.0, None)


def empirical_interval(residuals, level: float = 0.90):
    """(low, high) offsets from validation residual quantiles (true - predicted): interval = prediction + offsets.
    An empirical prediction interval, not a calibrated probability (pipeline.md 9)."""
    r = np.asarray(residuals, dtype=np.float64)
    a = (1.0 - level) / 2.0
    return float(np.quantile(r, a)), float(np.quantile(r, 1.0 - a))


def error_by_group(pred, true, labels) -> list[dict]:
    """RMSE / MAE per group (engine, subset, flight class, degradation phase...)."""
    pred, true, labels = np.asarray(pred), np.asarray(true), np.asarray(labels)
    rows = []
    for g in sorted(np.unique(labels), key=str):
        m = labels == g
        rows.append({"group": g.item() if hasattr(g, "item") else g, "n": int(m.sum()),
                     "rmse": rmse(pred[m], true[m]), "mae": mae(pred[m], true[m]),
                     "bias": float(np.mean(pred[m] - true[m]))})
    return rows


def rul_phase(rul) -> np.ndarray:
    """Degradation phase label from the true RUL (for reporting only)."""
    r = np.asarray(rul, dtype=np.float64)
    return np.where(r <= 20, "end_of_life (RUL<=20)", np.where(r <= 60, "mid (20<RUL<=60)", "early (RUL>60)"))


# ----------------------------------------------------------------------------------------------- forecasting
# EXTENSION beyond pipeline.md (which defines RUL as the only ML target): direct multi-horizon forecasts of component
# health. Target = true health h flights later MINUS the current EKF health estimate (so a model that learns nothing
# reproduces the "no change" baseline). Future labels come from the hidden parameters T and are used as targets only.
def future_values(units, cycles, values, h: int) -> np.ndarray:
    """Value of the same engine h flights later (matched by cycle number); NaN where the engine ends first.
    Rows must be sorted by (unit, cycle)."""
    units, cycles = np.asarray(units), np.asarray(cycles, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    j = np.minimum(np.arange(n) + int(h), n - 1)
    ok = (np.arange(n) + int(h) < n) & (units[j] == units) & (cycles[j] - cycles == h)
    out[ok] = values[j[ok]]
    return out


def trend_projection(units, cycles, values, h: int, window: int = 5) -> np.ndarray:
    """Causal straight-line baseline: per engine, least-squares slope (per flight) of the last `window` values,
    projected h flights ahead from the current value and clipped to [0, 1]."""
    units, values = np.asarray(units), np.asarray(values, dtype=np.float64)
    out = values.copy()
    for u in np.unique(units):
        idx = np.flatnonzero(units == u)
        v = values[idx]
        for i in range(len(idx)):
            win = v[max(0, i - window + 1): i + 1]
            if len(win) >= 2:
                t = np.arange(len(win)) - (len(win) - 1) / 2.0
                slope = (t * (win - win.mean())).sum() / (t ** 2).sum()
                out[idx[i]] = v[i] + slope * h
    return np.clip(out, 0.0, 1.0)


def forecast_metrics(pred, truth, persistence, trend, current_truth, degraded_below: float = 0.9) -> dict:
    """RMSE/MAE of the forecast and of both baselines on all rows and on DEGRADED rows (current true health below
    `degraded_below`; selected on the present, never on the future outcome). skill = 1 - RMSE/RMSE_persistence."""
    pred, truth = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    refs = {"forecast": pred, "persistence": np.asarray(persistence, dtype=np.float64),
            "trend": np.asarray(trend, dtype=np.float64)}
    cur = np.asarray(current_truth, dtype=np.float64)
    out = {}
    for part, m in (("all", np.ones(len(truth), dtype=bool)), ("degraded", cur < degraded_below)):
        row = {"n": int(m.sum())}
        for name, p in refs.items():
            row[f"{name}_rmse"] = rmse(p[m], truth[m]) if m.any() else None
            row[f"{name}_mae"] = mae(p[m], truth[m]) if m.any() else None
        base = row["persistence_rmse"]
        row["skill_vs_persistence"] = (1.0 - row["forecast_rmse"] / base) if base else None
        base = row["trend_rmse"]
        row["skill_vs_trend"] = (1.0 - row["forecast_rmse"] / base) if base else None
        out[part] = row
    return out


# ----------------------------------------------------------------------------------------------- feature-set selection
def choose_feature_set(setting, val_rmse: dict, n_val_engines: int, margin: float = 0.02, min_engines: int = 5):
    """Which RUL feature set ships: "full" (the spec's list, EKF features included) or "no_ekf".

    setting True / False forces the full / no-EKF set. "auto": the VALIDATION engines decide (pipeline.md 7.5) - the
    spec's full set is kept unless "no_ekf" is better by more than `margin` (relative), and the choice is only made at
    all if there are at least `min_engines` validation engines (one engine cannot choose reliably). Returns
    (choice, reason). Test engines never enter this decision."""
    if setting is True:
        return "full", "forced by config (EKF_IN_RUL = True)"
    if setting is False:
        return "no_ekf", "forced by config (EKF_IN_RUL = False)"
    if setting != "auto":
        raise ValueError(f"EKF_IN_RUL must be 'auto', True or False, got {setting!r}")
    if n_val_engines < min_engines:
        return "full", (f"only {n_val_engines} validation engine(s) (< {min_engines}): too few to choose reliably, "
                        "keeping the spec's full feature list")
    if val_rmse["no_ekf"] < (1.0 - margin) * val_rmse["full"]:
        return "no_ekf", (f"validation RMSE {val_rmse['no_ekf']:.2f} (without EKF features) beats "
                          f"{val_rmse['full']:.2f} (full) by more than {margin:.0%}")
    return "full", (f"without EKF features is not better by more than {margin:.0%} on validation "
                    f"({val_rmse['no_ekf']:.2f} vs {val_rmse['full']:.2f}): keeping the spec's full feature list")


# ----------------------------------------------------------------------------------------------- component RUL
# EXTENSION beyond pipeline.md: N-CMAPSS only has an engine-level RUL, so a component RUL is DERIVED offline from the
# hidden parameters T (labels only): flights until the component's TRUE health first falls to a failure level.
def component_rul_labels(units, cycles, health, level: float) -> np.ndarray:
    """Per row: flights until the engine's TRUE component health first falls to <= `level` (0 at and after the
    crossing). NaN for every row of an engine whose component never reaches the level (censored). Rows must be sorted
    by (unit, cycle)."""
    units, cycles = np.asarray(units), np.asarray(cycles, dtype=np.float64)
    health = np.asarray(health, dtype=np.float64)
    out = np.full(len(health), np.nan)
    for u in np.unique(units):
        idx = np.flatnonzero(units == u)
        below = np.flatnonzero(health[idx] <= level)
        if len(below):
            out[idx] = np.maximum(cycles[idx][below[0]] - cycles[idx], 0.0)
    return out


def trend_time_to_level(units, cycles, values, level: float, window: int = 5, cap: float = 100.0) -> np.ndarray:
    """Causal baseline: per engine, flights until a straight line fitted to the last `window` values reaches `level`
    (0 if already there; `cap` if the trend is flat or rising). Flights are assumed consecutive."""
    units, values = np.asarray(units), np.asarray(values, dtype=np.float64)
    out = np.full(len(values), float(cap))
    for u in np.unique(units):
        idx = np.flatnonzero(units == u)
        v = values[idx]
        for i in range(len(idx)):
            if v[i] <= level:
                out[idx[i]] = 0.0
                continue
            win = v[max(0, i - window + 1): i + 1]
            if len(win) >= 2:
                t = np.arange(len(win)) - (len(win) - 1) / 2.0
                slope = (t * (win - win.mean())).sum() / (t ** 2).sum()
                if slope < -1e-9:
                    out[idx[i]] = min((v[i] - level) / (-slope), cap)
    return out


def component_rul_metrics(pred, truth, engine_pred, trend, units) -> dict:
    """RMSE/MAE of the component-RUL model and of two baselines (the engine RUL used as the component RUL, and the
    EKF-health trend), on rows with a label. skill = 1 - RMSE / RMSE_baseline."""
    pred, truth = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    out = {"n": int(len(truth)), "n_engines": int(len(np.unique(units))) if len(truth) else 0}
    for name, p in (("model", pred), ("engine_rul", np.asarray(engine_pred, dtype=np.float64)),
                    ("trend", np.asarray(trend, dtype=np.float64))):
        out[f"{name}_rmse"] = rmse(p, truth) if len(truth) else None
        out[f"{name}_mae"] = mae(p, truth) if len(truth) else None
    for name in ("engine_rul", "trend"):
        base = out[f"{name}_rmse"]
        out[f"skill_vs_{name.replace('_rul', '')}"] = (1.0 - out["model_rmse"] / base) if base else None
    return out
