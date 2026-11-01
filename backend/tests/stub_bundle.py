"""A STUB (version 4) model bundle for tests and UI checks. Nothing here is trained: the four "models" are fixed
formulas, the healthy baseline and the EKF use hand-set parameters, the Ridge stack uses hand-set weights, and the
engine data is synthetic. Never use it for results."""
import numpy as np
import pandas as pd

from backend.pipeline import ncmapss as nc
from backend.pipeline import stacking as st
from backend.pipeline.bundle import ModelBundle

NOMINAL = np.array([600, 1500, 1800, 1300, 20, 14, 18, 25, 300, 280, 22, 2000, 8500, 2.0])


class Lin:
    """predict(X) = b + w * X[feature] (a fixed formula, not a fitted model)."""

    def __init__(self, feature, w, b):
        self.f, self.w, self.b = feature, w, b

    def predict(self, x):
        return self.b + self.w * x[self.f].to_numpy(dtype=float)


def manual_ekf(seed=0):
    """EKF with hand-set parameters: one state per health parameter, operating-point independent influence."""
    rng = np.random.default_rng(seed)
    e = nc.GasPathEKF()
    s, p, k = len(nc.XS_VARS), len(nc.T_VARS), 4
    e.b, e.H_params = np.zeros((k, s)), np.zeros((p, k, s))
    e.H_params[:, 0, :] = rng.normal(size=(p, s))
    e.A = np.eye(p)
    e.H = e.H_params.copy()
    e.groups, e.state_names = [[j] for j in range(p)], list(nc.T_VARS)
    e.R, e.Q0, e.q_scale = np.full(s, 1e-6), np.full(p, 1e-6), 1.0
    e.x0, e.P0 = np.zeros(p), np.full(p, 1e-3)
    return e


def manual_reference():
    ref = nc.HealthyReference()
    ref.coef = np.zeros((len(nc.poly_terms(4, 3)), len(nc.XS_VARS)))
    ref.coef[0] = NOMINAL                                              # healthy value = nominal (phi_0 mean is 1)
    return ref


def demo_frame(n_units=2, n_cycles=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, n_cycles + 1):
            frac = c / n_cycles
            r = {"unit": 1000 + u, "unit_local": u, "dataset": "STUB", "cycle": c, "fc": 1 + c % 3, "hs": int(frac < 0.6),
                 "rul": float(n_cycles - c), "duration_s": 3000 + 100 * rng.normal(), "alt_mean": 25000 + 3000 * rng.normal(),
                 "Mach_mean": 0.6, "TRA_mean": 70.0, "T2_mean": 495.0, "alt_max": 32000.0, "Mach_max": 0.8}
            drift = np.zeros(14)
            drift[[2, 3, 13]] = [0.02, 0.015, 0.03] * np.array(frac ** 2)    # T48, T50, Wf rise late in life
            m = NOMINAL * (1 + drift + 1e-3 * rng.normal(size=14))
            for j, s in enumerate(nc.XS_VARS):
                r[f"m_{s}"], r[f"sd_{s}"] = m[j], 0.01 * m[j]
                r[f"mn_{s}"], r[f"mx_{s}"], r[f"ls_{s}"] = 0.98 * m[j], 1.02 * m[j], m[j]
            for pn in nc.PROXIES:
                r[f"p_{pn}"] = 0.85 - 0.05 * frac ** 2 if pn.startswith("eta") else 1.0 + 0.01 * rng.normal()
            for j in range(len(nc.poly_terms(4, 3))):
                r[f"phi_{j}"] = 1.0 if j == 0 else 0.0
            r["v_T40"], r["v_P30"], r["v_P45"] = 2600 * (1 + drift[2]), 310.0, 60.0
            for comp in nc.COMPONENTS:
                r[f"hi_{comp}"] = float(np.clip(1 - (0.9 if comp == "hpt" else 0.2) * frac ** 2, 0, 1))
            rows.append(r)
    return pd.DataFrame(rows)


class Const:
    """Stub component-RUL model: a constant output (a formula, not a fitted model)."""

    def __init__(self, v):
        self.v = v

    def predict(self, x):
        return np.full(len(x), self.v)


class Delta:
    """Stub forecaster: predicts a fixed change in health over h flights (a formula, not a fitted model)."""

    def __init__(self, h):
        self.h = h

    def predict(self, x):
        return np.full(len(x), -0.002 * self.h)


def stub_bundle(n_units=2, n_cycles=40, seed=0):
    ekf = manual_ekf(seed)
    feat = nc.CausalFeaturizer(*nc.default_feature_columns(len(ekf.state_names)), roll=5)
    names = list(st.BASE_NAMES)
    models = {n: Lin("cycle", -1.0, 42.0 + d) for n, d in zip(names, (0, 2, -2, 1))}
    stack = st.StackModel(names, mean=np.zeros(4), scale=np.ones(4), coef=[0.25] * 4, intercept=0.0, alpha=1.0)
    manifest = {"version": 4, "dataset": "STUB (synthetic, UI/test only)", "run": "stub", "files": [],
                "components": nc.COMPONENTS, "d_ref": {c: 0.03 for c in nc.COMPONENTS}, "features": feat.feature_names,
                "interval": {"level": 0.9, "low": -5.0, "high": 5.0, "method": "empirical (stub)"},
                "band_edges": [0.75, 0.55, 0.30], "metrics": {"note": "stub"},
                "build": {"commit": "stub", "trained_utc": "stub"}}
    horizons = (10, 25)
    manifest["forecast"] = {
        "model": "stub", "horizons": list(horizons), "target": "stub",
        "features": feat.feature_names + [f"kf_hi_{c}" for c in nc.COMPONENTS],
        "intervals": {c: {str(h): {"low": -0.05, "high": 0.05} for h in horizons} for c in nc.COMPONENTS},
        "useful": {c: {str(h): c == "hpt" for h in horizons} for c in nc.COMPONENTS}}
    fmodels = {(c, h): Delta(h) for c in nc.COMPONENTS for h in horizons}
    manifest["component_rul"] = {
        "model": "stub", "level": 0.30, "cap": 100.0, "show_below": 0.9, "target": "stub",
        "features": feat.feature_names + [f"kf_hi_{c}" for c in nc.COMPONENTS],
        "intervals": {"hpt": {"low": -4.0, "high": 6.0}}, "useful": {c: c == "hpt" for c in nc.COMPONENTS}}
    cmodels = {c: Const(17.0 if c == "hpt" else 250.0) for c in ("hpt", "lpt")}      # lpt output exceeds the cap
    return ModelBundle(manifest, manual_reference(), ekf, feat, models, stack, demo_frame(n_units, n_cycles, seed),
                       fmodels, cmodels)


def raw_1hz_frame(n_cycles=12, sec=30, seed=1):
    """Synthetic 1 Hz upload in N-CMAPSS column layout."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(1, n_cycles + 1):
        for _ in range(sec):
            rows.append([rng.uniform(10000, 35000), rng.uniform(0.3, 0.8), rng.uniform(50, 90), rng.uniform(470, 520)]
                        + list(NOMINAL * (1 + 1e-3 * rng.normal(size=14))) + [7, c, 2])
    return pd.DataFrame(rows, columns=nc.W_VARS + nc.XS_VARS + ["unit", "cycle", "Fc"])
