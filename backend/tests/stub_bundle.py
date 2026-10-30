"""A STUB model bundle for tests and UI checks. Nothing here is trained: models are fixed formulas, the healthy
reference and Kalman filter use hand-set parameters, and the engine data is synthetic. Never use it for results."""
import numpy as np
import pandas as pd

from backend.pipeline import ncmapss as nc
from backend.pipeline.bundle import ModelBundle

NOMINAL = np.array([600, 1500, 1800, 1300, 20, 14, 18, 25, 300, 280, 22, 2000, 8500, 2.0])
TARGETS = ["rul"] + [f"hi_{c}" for c in nc.COMPONENTS]


class Lin:
    """predict(X) = b + w * X[feature] (a fixed formula, not a fitted model)."""

    def __init__(self, feature, w, b):
        self.f, self.w, self.b = feature, w, b

    def predict(self, x):
        return self.b + self.w * x[self.f].to_numpy(dtype=float)


class StubCNN:
    """Stands in for OnnxCNN: per-cycle outputs from a fixed formula of the cycle number (NaN for cycles without
    0.1 Hz samples, like the real one); occluding T48 shifts RUL so the occlusion path is exercised."""

    def __init__(self, n_out):
        self.n_out = n_out

    def cycle_predictions(self, samples, cyc, ref, occlude=None):
        c = cyc["cycle"].to_numpy(dtype=float)
        out = np.column_stack([np.clip(42 - c + (0.5 if occlude == "T48" else 0.0), 0, None)] +
                              [np.clip(1 - 0.01 * c, 0, 1)] * (self.n_out - 1))
        have = np.zeros(len(cyc), dtype=bool)
        rows = nc.cycle_rows(samples, cyc)
        have[rows[rows >= 0]] = True
        out[~have] = np.nan
        return out


def stub_bundle(with_cnn=True, n_units=2, n_cycles=40, seed=0, smoothing=None):
    rng = np.random.default_rng(seed)
    ref = nc.HealthyReference()
    ref.coef = np.zeros((len(nc.poly_terms(4, 3)), len(nc.XS_VARS)))
    ref.coef[0] = NOMINAL                                              # ideal = nominal (phi_0 mean is 1)
    kf = nc.GasPathKF()
    s, p, k = len(nc.XS_VARS), len(nc.T_VARS), 4
    kf.b, kf.H = np.zeros((k, s)), np.zeros((p, k, s))
    kf.H[:, 0, :] = rng.normal(size=(p, s))
    kf.R, kf.Q, kf.x0, kf.P0 = np.full(s, 1e-5), np.full(p, 1e-6), np.zeros(p), np.full(p, 1e-4)
    feat = nc.CausalFeaturizer()
    n = len(feat.base_cols) + len(feat.flight_cols)
    feat.mu, feat.sd = np.zeros(n), np.ones(n)
    feat.mu[len(feat.base_cols):] = [20000, 0.5, 60, 500, 3000, 2]
    feat.sd[len(feat.base_cols):] = [10000, 0.2, 20, 30, 1000, 1]
    feat._names()
    trees, targets = {}, {}
    for ti, t in enumerate(TARGETS):
        if t == "rul":
            mods = [Lin("cycle", -1.0, 42.0 + d) for d in (0, 2, -2, 1)]
        else:
            comp = t[3:]
            eff = dict(nc.COMP_THETA)[comp][0]
            mods = [Lin(f"kf_{eff}", 8.0, 0.95 + 0.01 * d) for d in (0, 1, -1, 0)]
        trees[t] = list(zip(("xgboost", "lightgbm", "catboost", "randomforest"), mods))
        bases = [{"type": k_, "file": "stub"} for k_, _ in trees[t]] + [{"type": "cnn", "output_index": ti}]
        targets[t] = {"base_models": bases, "meta": {"coef": [0.3, 0.2, 0.2, 0.1, 0.2], "intercept": 0.0},
                      "trees_only_meta": {"coef": [0.4, 0.2, 0.2, 0.2], "intercept": 0.0}}
    manifest = {"version": 3, "dataset": "STUB (synthetic, UI/test only)", "run": "stub", "files": [],
                "components": nc.COMPONENTS, "targets": targets, "target_order": TARGETS, "rul_scale": 100.0,
                "d_ref": 0.03, "band_edges": [0.75, 0.55, 0.30], "features": feat.feature_names,
                "metrics": {"note": "stub"}, "cnn": {"file": "stub"}, "smoothing": smoothing or {}}
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, n_cycles + 1):
            frac = c / n_cycles
            r = {"unit": 1000 + u, "unit_local": u, "dataset": "STUB", "cycle": c, "fc": 1 + c % 3, "hs": int(frac < 0.6),
                 "rul": float(n_cycles - c), "duration_s": 3000 + 100 * rng.normal(), "alt_mean": 25000 + 3000 * rng.normal(),
                 "Mach_mean": 0.6, "TRA_mean": 70, "T2_mean": 495, "alt_max": 32000, "Mach_max": 0.8}
            drift = np.zeros(14)
            drift[[2, 3, 13]] = [0.02, 0.015, 0.03] * np.array(frac ** 2)       # T48, T50, Wf rise late in life
            m = NOMINAL * (1 + drift + 1e-3 * rng.normal(size=14))
            for j, sname in enumerate(nc.XS_VARS):
                r[f"m_{sname}"], r[f"sd_{sname}"] = m[j], 0.01 * m[j]
            for pn in nc.PROXIES:
                r[f"p_{pn}"] = 0.85 - 0.05 * frac ** 2 if pn.startswith("eta") else 1.0 + 0.01 * rng.normal()
            for j in range(len(nc.poly_terms(4, 3))):
                r[f"phi_{j}"] = 1.0 if j == 0 else 0.0
            r["v_T40"], r["v_P30"], r["v_P45"] = 2600 * (1 + drift[2]), 310.0, 60.0
            for comp in nc.COMPONENTS:
                r[f"hi_{comp}"] = float(np.clip(1 - (0.9 if comp == "hpt" else 0.2) * frac ** 2, 0, 1))
            rows.append(r)
    demo = pd.DataFrame(rows)
    samples = pd.DataFrame([{"unit": r["unit"], "cycle": r["cycle"], "alt": r["alt_mean"], "Mach": 0.6, "TRA": 70.0,
                             "T2": 495.0, **{s_: r[f"m_{s_}"] for s_ in nc.XS_VARS}} for r in rows for _ in range(3)])
    quant = {"q05": Lin("cycle", -1.0, 32.0), "q95": Lin("cycle", -1.0, 52.0)}
    cnn = StubCNN(len(TARGETS)) if with_cnn else None
    return ModelBundle(manifest, ref, kf, feat, trees, quant, demo, cnn, samples)


def raw_1hz_frame(n_cycles=12, sec=30, seed=1):
    """Synthetic 1 Hz upload in N-CMAPSS column layout."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(1, n_cycles + 1):
        for _ in range(sec):
            rows.append([rng.uniform(10000, 35000), rng.uniform(0.3, 0.8), rng.uniform(50, 90), rng.uniform(470, 520)]
                        + list(NOMINAL * (1 + 1e-3 * rng.normal(size=14))) + [7, c, 2])
    return pd.DataFrame(rows, columns=nc.W_VARS + nc.XS_VARS + ["unit", "cycle", "Fc"])
