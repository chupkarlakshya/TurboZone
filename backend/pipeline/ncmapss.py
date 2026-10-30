"""NASA N-CMAPSS (C-MAPSS 2) digital-twin pipeline: Layers 1-3 plus feature engineering.

This ONE module is used by the Colab training notebook (inlined) and by the backend, so models are trained and
served on identical computations. Dependencies: numpy, pandas (+ h5py only for reading the raw .h5 files).

Layer 1  load_h5 / aggregate_cycles : 1 Hz flight data -> one row per (unit, flight cycle)
Layer 2  physics_1hz + HealthyReference : first-principles ratios/efficiency proxies, and a healthy-engine map
         (sensor = f(flight condition)) giving the "ideal" sensor value and the residual (actual vs ideal)
Layer 3  GasPathKF : Kalman filter on a linearised gas-path model, estimating the 10 component health
         parameters (efficiency/flow modifiers) with uncertainty, plus per-sensor innovation checks
Layer 4 inputs  CausalFeaturizer, build_windows : causal per-engine features for the trees;
         downsample_1hz + SampleEncoder : 0.1 Hz in-flight sequences (+ residuals, KF states) for the CNN
Post-processing  smooth_target : causal per-engine smoothing of predictions (RUL drops 1 per cycle)

Conventions: health index 1.0 = healthy, 0.0 = failure-level degradation. Everything is causal per engine
(uses only the current and earlier cycles), so the same code runs cycle by cycle in real time.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------------- schema
W_VARS = ["alt", "Mach", "TRA", "T2"]
XS_VARS = ["T24", "T30", "T48", "T50", "P15", "P2", "P21", "P24", "Ps30", "P40", "P50", "Nf", "Nc", "Wf"]
XV_DISPLAY = ["T40", "P30", "P45"]   # virtual (unmeasured) sensors: kept for display only, never model inputs
T_VARS = ["fan_eff_mod", "fan_flow_mod", "LPC_eff_mod", "LPC_flow_mod", "HPC_eff_mod", "HPC_flow_mod",
          "HPT_eff_mod", "HPT_flow_mod", "LPT_eff_mod", "LPT_flow_mod"]
COMPONENTS = ["fan", "lpc", "hpc", "hpt", "lpt"]
COMP_THETA = {"fan": ("fan_eff_mod", "fan_flow_mod"), "lpc": ("LPC_eff_mod", "LPC_flow_mod"),
              "hpc": ("HPC_eff_mod", "HPC_flow_mod"), "hpt": ("HPT_eff_mod", "HPT_flow_mod"),
              "lpt": ("LPT_eff_mod", "LPT_flow_mod")}
COMP_NAMES = {"fan": "Fan", "lpc": "LPC", "hpc": "HPC", "hpt": "HPT", "lpt": "LPT"}

SENSOR_INFO = {   # description, kind ("T" degR, "P" psia, "N" rpm, "W" lbm/s)
    "T24": ("LPC outlet total temperature", "T"), "T30": ("HPC outlet total temperature", "T"),
    "T48": ("HPT outlet total temperature (EGT)", "T"), "T50": ("LPT outlet total temperature", "T"),
    "P15": ("Bypass duct total pressure", "P"), "P2": ("Fan inlet total pressure", "P"),
    "P21": ("Fan outlet total pressure", "P"), "P24": ("LPC outlet total pressure", "P"),
    "Ps30": ("HPC outlet static pressure", "P"), "P40": ("Burner outlet total pressure", "P"),
    "P50": ("LPT outlet total pressure", "P"), "Nf": ("Fan (LP spool) speed", "N"),
    "Nc": ("Core (HP spool) speed", "N"), "Wf": ("Fuel flow", "W"),
    "T40": ("Burner outlet temperature (virtual)", "T"), "P30": ("HPC outlet total pressure (virtual)", "P"),
    "P45": ("HPT outlet total pressure (virtual)", "P"),
}

GAMMA = 1.4
K_COLD = (GAMMA - 1.0) / GAMMA
GAMMA_HOT = 1.33
K_HOT = (GAMMA_HOT - 1.0) / GAMMA_HOT

# Physics proxies computed at 1 Hz from MEASURED signals only, and the sensors each one depends on
# (used by the occlusion explainer).
PROXY_DEPS = {
    "fan_pr": ["P21", "P2"], "lpc_pr": ["P24", "P21"], "hpc_pr": ["Ps30", "P24"],
    "eta_core": ["P24", "P2", "T24"], "eta_hpc": ["Ps30", "P24", "T30", "T24"],
    "burner_dp": ["P40", "Ps30"], "lpt_tr": ["T50", "T48"], "turb_pr": ["P40", "P50"],
    "nf_corr": ["Nf"], "nc_corr": ["Nc", "T24"], "wf_corr": ["Wf", "P2"], "spool_ratio": ["Nc", "Nf"],
}
PROXIES = list(PROXY_DEPS)

# Fixed (not data-fitted) centring/scaling of the flight condition, so per-cycle polynomial means can be
# accumulated before any model is fitted.
W_CENTER = {"alt": (20000.0, 15000.0), "Mach": (0.5, 0.25), "TRA": (60.0, 20.0), "T2": (500.0, 30.0)}

DEG_R_TO_K = 5.0 / 9.0
PSIA_TO_KPA = 6.894757
FT_TO_M = 0.3048
LBM_TO_KG = 0.45359237


# ----------------------------------------------------------------------------------------------- Layer 1
def h5_names(f, key: str) -> list[str]:
    return [str(s).strip() for s in np.array(f[key]).reshape(-1).astype("U32")]


def load_h5(path: str, split: str = "dev", virtual=XV_DISPLAY) -> pd.DataFrame:
    """Read one split ("dev" or "test") of an N-CMAPSS .h5 file into a 1 Hz DataFrame (float32).

    Columns are mapped by the variable names stored in the file (W_var, X_s_var, ...), not by position.
    """
    import h5py

    with h5py.File(path, "r") as f:
        frames = []
        for grp in ("W", "X_s", "T", "A"):
            names = h5_names(f, f"{grp}_var")
            dtype = np.int32 if grp == "A" else np.float32
            frames.append(pd.DataFrame(np.asarray(f[f"{grp}_{split}"], dtype=dtype), columns=names))
        frames.append(pd.DataFrame({"RUL": np.asarray(f[f"Y_{split}"], dtype=np.float32).reshape(-1)}))
        if virtual and f"X_v_{split}" in f:
            vnames = h5_names(f, "X_v_var")
            keep = [vnames.index(v) for v in virtual if v in vnames]
            if keep:
                arr = np.asarray(f[f"X_v_{split}"][:, sorted(keep)], dtype=np.float32)
                frames.append(pd.DataFrame(arr, columns=["v_" + vnames[i] for i in sorted(keep)]))
    df = pd.concat(frames, axis=1)
    missing = [c for c in W_VARS + XS_VARS + ["unit", "cycle"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} [{split}] is missing variables {missing}")
    # physically impossible rows (e.g. zero-filled bytes of a repaired truncated file): sensors are always > 0
    bad = (df[XS_VARS].to_numpy() <= 0).any(axis=1) | (df["unit"].to_numpy() < 1) | (df["cycle"].to_numpy() < 1)
    if bad.any():
        df = df.loc[~bad].reset_index(drop=True)
    df.attrs["dropped_rows"] = int(bad.sum())
    return df


def repair_truncated_h5(path: str, workdir: str, max_missing: int = 4096):
    """Some mirrors ship an .h5 a few bytes shorter than its header records (Kaggle's DS08d-010: 32 bytes), which
    HDF5 refuses to open. Copy it to `workdir` and zero-pad the copy up to the recorded size (load_h5 then drops any
    zeroed rows). Returns (path to open, repaired copy to delete afterwards or None)."""
    import os
    import re
    import shutil

    import h5py

    try:
        h5py.File(path, "r").close()
        return path, None
    except OSError as exc:
        m = re.search(r"eof = (\d+), sblock->base_addr = \d+, stored_eof = (\d+)", str(exc))
        if not m or not 0 < int(m.group(2)) - int(m.group(1)) <= max_missing:
            raise
        missing = int(m.group(2)) - int(m.group(1))
    fixed = os.path.join(workdir, "repaired_" + os.path.basename(path))
    shutil.copyfile(path, fixed)
    with open(fixed, "ab") as fh:
        fh.write(b"\0" * missing)
    return fixed, fixed


def physics_1hz(df: pd.DataFrame) -> pd.DataFrame:
    """First-principles ratios from measured signals (unit-free; temperatures in degR, pressures in psia).

    Efficiencies are isentropic PROXIES: eta_core spans fan-hub + LPC (T2,P2 -> T24,P24); eta_hpc uses the
    measured static Ps30 because total P30 is not measured.
    """
    def ratio(a, b):
        b = np.asarray(b, dtype=np.float64)
        return np.asarray(a, dtype=np.float64) / np.where(np.abs(b) < 1e-9, np.nan, b)

    def eta(pr, tr):
        pr = np.clip(pr, 1e-6, None)
        return np.clip(ratio(pr ** K_COLD - 1.0, tr - 1.0), 0.0, 1.5)

    out = pd.DataFrame(index=df.index)
    out["fan_pr"] = ratio(df["P21"], df["P2"])
    out["lpc_pr"] = ratio(df["P24"], df["P21"])
    out["hpc_pr"] = ratio(df["Ps30"], df["P24"])
    out["eta_core"] = eta(ratio(df["P24"], df["P2"]), ratio(df["T24"], df["T2"]))
    out["eta_hpc"] = eta(out["hpc_pr"].to_numpy(), ratio(df["T30"], df["T24"]))
    out["burner_dp"] = 1.0 - ratio(df["P40"], df["Ps30"])
    out["lpt_tr"] = ratio(df["T50"], df["T48"])
    out["turb_pr"] = ratio(df["P40"], df["P50"])
    theta2 = ratio(df["T2"], 518.67)
    delta2 = ratio(df["P2"], 14.696)
    out["nf_corr"] = ratio(df["Nf"], np.sqrt(theta2))
    out["nc_corr"] = ratio(df["Nc"], np.sqrt(ratio(df["T24"], 518.67)))
    out["wf_corr"] = ratio(df["Wf"], delta2 * np.sqrt(theta2))
    out["spool_ratio"] = ratio(df["Nc"], df["Nf"])
    return out.astype(np.float32)


def scaled_w(w: np.ndarray) -> np.ndarray:
    c = np.array([W_CENTER[v][0] for v in W_VARS], dtype=np.float64)
    s = np.array([W_CENTER[v][1] for v in W_VARS], dtype=np.float64)
    return (np.asarray(w, dtype=np.float64) - c) / s


def poly_terms(n_vars: int, degree: int) -> list[tuple]:
    terms = [()]
    for d in range(1, degree + 1):
        terms += list(itertools.combinations_with_replacement(range(n_vars), d))
    return terms


def poly_features(z: np.ndarray, degree: int = 3) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    cols = [np.prod(z[:, list(t)], axis=1) if t else np.ones(len(z)) for t in poly_terms(z.shape[1], degree)]
    return np.column_stack(cols)


def aggregate_cycles(df: pd.DataFrame, degree: int = 3) -> pd.DataFrame:
    """Collapse 1 Hz rows into one row per (unit, cycle).

    Stores per-cycle means of sensors (m_*), their std (sd_*), physics proxies (p_*), virtual sensors (v_*),
    flight descriptors, labels (RUL, health parameters, hs) and the mean polynomial features of the flight
    condition (phi_*). Because the healthy reference is linear in phi, mean(reference) = coef . mean(phi),
    which gives exact per-cycle "ideal" sensor means without a second pass over the 1 Hz data.
    """
    out = []
    vcols = [c for c in df.columns if c.startswith("v_")]
    tcols = [c for c in T_VARS if c in df.columns]
    for unit, part in df.groupby("unit", sort=True):
        _, inv = np.unique(part["cycle"].to_numpy(), return_inverse=True)
        cnt = np.bincount(inv).astype(np.float64)

        def gmean(a):
            """NaN-aware per-cycle mean."""
            a = np.asarray(a, dtype=np.float64)
            a = a.reshape(len(a), -1)
            ok = np.isfinite(a)
            sums = np.column_stack([np.bincount(inv, weights=np.where(ok[:, j], a[:, j], 0.0), minlength=len(cnt))
                                    for j in range(a.shape[1])])
            n = np.column_stack([np.bincount(inv, weights=ok[:, j].astype(float), minlength=len(cnt))
                                 for j in range(a.shape[1])])
            return sums / np.where(n == 0, np.nan, n)

        def gmax(a):
            res = np.full(len(cnt), -np.inf)
            np.maximum.at(res, inv, np.asarray(a, dtype=np.float64))
            return res

        def gmin(a):
            res = np.full(len(cnt), np.inf)
            np.minimum.at(res, inv, np.asarray(a, dtype=np.float64))
            return res

        rows = {"unit": np.full(len(cnt), int(unit)), "cycle": gmin(part["cycle"]).astype(int),
                "duration_s": cnt}
        if "Fc" in part:
            rows["fc"] = gmin(part["Fc"]).astype(int)
        if "hs" in part:
            rows["hs"] = gmin(part["hs"]).astype(int)
        if "RUL" in part:
            rows["rul"] = gmin(part["RUL"])
        w = part[W_VARS].to_numpy(dtype=np.float64)
        wm = gmean(w)
        for j, v in enumerate(W_VARS):
            rows[f"{v}_mean"] = wm[:, j]
        rows["alt_max"] = gmax(part["alt"])
        rows["Mach_max"] = gmax(part["Mach"])
        x = part[XS_VARS].to_numpy(dtype=np.float64)
        xm, x2 = gmean(x), gmean(x ** 2)
        for j, s in enumerate(XS_VARS):
            rows[f"m_{s}"] = xm[:, j]
            rows[f"sd_{s}"] = np.sqrt(np.clip(x2[:, j] - xm[:, j] ** 2, 0, None))
        pm = gmean(physics_1hz(part).to_numpy())
        for j, p in enumerate(PROXIES):
            rows[f"p_{p}"] = pm[:, j]
        if vcols:
            vm = gmean(part[vcols].to_numpy())
            for j, c in enumerate(vcols):
                rows[c] = vm[:, j]
        if tcols:
            tm = gmean(part[tcols].to_numpy())
            for j, c in enumerate(tcols):
                rows[c] = tm[:, j]
        phim = gmean(poly_features(scaled_w(w), degree))
        for j in range(phim.shape[1]):
            rows[f"phi_{j}"] = phim[:, j]
        out.append(pd.DataFrame(rows))
    return pd.concat(out, ignore_index=True).sort_values(["unit", "cycle"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- Layer 2
class HealthyReference:
    """Healthy-engine map: each measured sensor as a degree-3 polynomial of the flight condition (alt, Mach,
    TRA, T2), fitted by ridge regression on healthy (hs == 1) flight data of the training engines only.

    It plays the role of the diagram's "ideal performance" model: for any flight condition it predicts what a
    healthy engine would read, so residual = actual - ideal isolates degradation from operating conditions.
    """

    def __init__(self, degree: int = 3, alpha: float = 1e-3, sensors=XS_VARS):
        self.degree, self.alpha, self.sensors = degree, alpha, list(sensors)
        self.coef = None            # (n_terms, n_sensors)
        self.fit_rmse = None

    def fit(self, w: np.ndarray, x: np.ndarray) -> "HealthyReference":
        phi = poly_features(scaled_w(w), self.degree)
        a = phi.T @ phi + self.alpha * np.eye(phi.shape[1])
        self.coef = np.linalg.solve(a, phi.T @ np.asarray(x, dtype=np.float64))
        pred = phi @ self.coef
        self.fit_rmse = (np.sqrt(np.mean((pred - x) ** 2, axis=0)) / np.abs(np.mean(x, axis=0))).tolist()
        return self

    def expected_from_phi(self, phi_mean: np.ndarray) -> np.ndarray:
        return np.asarray(phi_mean, dtype=np.float64) @ self.coef

    def predict(self, w: np.ndarray) -> np.ndarray:
        return poly_features(scaled_w(w), self.degree) @ self.coef

    def to_dict(self):
        return {"degree": self.degree, "alpha": self.alpha, "sensors": self.sensors,
                "coef": np.asarray(self.coef).tolist(), "fit_rel_rmse": self.fit_rmse, "w_center": W_CENTER}

    @classmethod
    def from_dict(cls, d):
        r = cls(d["degree"], d["alpha"], d["sensors"])
        r.coef = np.asarray(d["coef"], dtype=np.float64)
        r.fit_rmse = d.get("fit_rel_rmse")
        return r


def add_residuals(cyc: pd.DataFrame, ref: HealthyReference) -> pd.DataFrame:
    """Add ref_<sensor> (ideal, healthy-engine value) and res_<sensor> (relative residual, actual/ideal - 1)."""
    cyc = cyc.copy()
    phi_cols = [f"phi_{j}" for j in range(ref.coef.shape[0])]
    exp = ref.expected_from_phi(cyc[phi_cols].to_numpy())
    for j, s in enumerate(ref.sensors):
        cyc[f"ref_{s}"] = exp[:, j]
        cyc[f"res_{s}"] = cyc[f"m_{s}"] / np.where(np.abs(exp[:, j]) < 1e-9, np.nan, exp[:, j]) - 1.0
    return cyc


# ----------------------------------------------------------------------------------------------- labels
def component_damage(df: pd.DataFrame) -> pd.DataFrame:
    """Per-component damage = larger of |efficiency modifier| and |flow modifier|."""
    return pd.DataFrame({c: np.maximum(df[e].abs(), df[f].abs()) for c, (e, f) in COMP_THETA.items()},
                        index=df.index)


def fit_damage_reference(cyc_train: pd.DataFrame, q: float = 0.95, floor: float = 1e-3) -> dict:
    """Failure-level damage PER COMPONENT: q-quantile over training engines of that component's damage at end of
    life. One shared value would let the component with the largest modifiers (e.g. fan flow) squeeze the others,
    so a turbine at end of life would still read ~0.8 health."""
    last = cyc_train.sort_values("cycle").groupby("unit").tail(1)
    dmg = component_damage(last)
    return {c: max(float(np.quantile(dmg[c], q)), floor) for c in COMPONENTS}


def d_ref_of(d_ref, comp: str) -> float:
    """d_ref may be one float (older bundles) or {component: float}."""
    return float(d_ref[comp]) if isinstance(d_ref, dict) else float(d_ref)


def component_health(df: pd.DataFrame, d_ref) -> pd.DataFrame:
    """HI_c = 1 - damage_c / d_ref_c, clipped to [0, 1]: fraction of failure-level degradation still unused."""
    dmg = component_damage(df)
    return pd.DataFrame({f"hi_{c}": (1.0 - dmg[c] / d_ref_of(d_ref, c)).clip(0.0, 1.0) for c in COMPONENTS},
                        index=df.index)


# ----------------------------------------------------------------------------------------------- Layer 3
SCHED_VARS = ["alt_mean", "Mach_mean", "TRA_mean"]


def schedule(cyc: pd.DataFrame) -> np.ndarray:
    """Operating-point variables for the influence coefficients: [1, alt, Mach, TRA] (fixed scaling)."""
    z = [np.ones(len(cyc))]
    for v, raw in zip(SCHED_VARS, ["alt", "Mach", "TRA"]):
        c, s = W_CENTER[raw]
        z.append((cyc[v].to_numpy(dtype=np.float64) - c) / s)
    return np.column_stack(z)


class GasPathKF:
    """Kalman filter on a linearised gas-path model (classic gas-path analysis).

    Measurement model per cycle:  res = b(w) + H(w) . theta + noise, where res are the 14 relative sensor
    residuals, theta the 10 component health parameters and w the operating point. H(w) (influence
    coefficients) is linear in [1, alt, Mach, TRA], i.e. the model is re-linearised at every operating point;
    because it is linear in theta at each point, the EKF update reduces to a time-varying Kalman filter.
    H and b are identified on training engines (where theta is known); at run time only sensors are used.
    State model: random walk with process noise Q estimated from cycle-to-cycle theta changes.
    """

    def __init__(self, sensors=XS_VARS, theta=T_VARS, ridge: float = 1e-4, gate: float = 9.0, q_scale: float = 1.0):
        self.sensors, self.theta = list(sensors), list(theta)
        self.ridge, self.gate, self.q_scale = ridge, gate, q_scale
        self.b = self.H = self.R = self.Q = self.x0 = self.P0 = None

    def _design(self, z, th):
        return np.column_stack([z] + [th[:, [p]] * z for p in range(th.shape[1])])

    def fit(self, cyc: pd.DataFrame) -> "GasPathKF":
        y = cyc[[f"res_{s}" for s in self.sensors]].to_numpy(dtype=np.float64)
        th = cyc[self.theta].to_numpy(dtype=np.float64)
        z = schedule(cyc)
        x = self._design(z, th)
        coef = np.linalg.solve(x.T @ x + self.ridge * np.eye(x.shape[1]), x.T @ y)    # (K + P*K, S)
        k = z.shape[1]
        self.b = coef[:k]                                                             # (K, S)
        self.H = coef[k:].reshape(len(self.theta), k, len(self.sensors))              # (P, K, S)
        self.R = np.var(y - x @ coef, axis=0) + 1e-12
        cyc_sorted = cyc.sort_values(["unit", "cycle"])
        d = cyc_sorted.groupby("unit")[self.theta].diff().dropna().to_numpy()
        self.Q = self.q_scale * (np.var(d, axis=0) + np.mean(d, axis=0) ** 2) + 1e-12
        first = cyc_sorted.groupby("unit")[self.theta].first().to_numpy()
        self.x0 = first.mean(axis=0)
        self.P0 = first.var(axis=0) + 1e-8
        return self

    def h_at(self, z_row):
        return np.einsum("pks,k->sp", self.H, z_row), z_row @ self.b                  # (S, P), (S,)

    def run(self, cyc: pd.DataFrame, res_override: np.ndarray | None = None) -> pd.DataFrame:
        """Filter every engine causally (sorted by unit, cycle). Returns states, 1-sigma, NIS and gate flags."""
        cyc = cyc.reset_index(drop=True)
        res = cyc[[f"res_{s}" for s in self.sensors]].to_numpy(dtype=np.float64) if res_override is None \
            else res_override
        z = schedule(cyc)
        n, p, s = len(cyc), len(self.theta), len(self.sensors)
        xs, sig = np.zeros((n, p)), np.zeros((n, p))
        nis, gated = np.full((n, s), np.nan), np.zeros((n, s), dtype=bool)
        units = cyc["unit"].to_numpy()
        for u in pd.unique(units):
            idx = np.flatnonzero(units == u)
            x, pm = self.x0.copy(), np.diag(self.P0)
            consec = np.zeros(s, dtype=int)
            for i in idx:
                pm = pm + np.diag(self.Q)
                h, b = self.h_at(z[i])
                for j in range(s):
                    r = res[i, j]
                    if not np.isfinite(r):
                        continue
                    hj = h[j]
                    innov = r - b[j] - hj @ x
                    ph = pm @ hj
                    sv = hj @ ph + self.R[j]
                    nis[i, j] = innov ** 2 / sv
                    if nis[i, j] > self.gate and consec[j] < 3:
                        gated[i, j] = True
                        consec[j] += 1
                        continue
                    consec[j] = 0
                    kg = ph / sv
                    x = x + kg * innov
                    ikh = np.eye(p) - np.outer(kg, hj)
                    pm = ikh @ pm @ ikh.T + self.R[j] * np.outer(kg, kg)           # Joseph form
                xs[i], sig[i] = x, np.sqrt(np.clip(np.diag(pm), 0, None))
        out = {}
        for k_, t in enumerate(self.theta):
            out[f"kf_{t}"] = xs[:, k_]
            out[f"kf_{t}_sigma"] = sig[:, k_]
        for j, sname in enumerate(self.sensors):
            out[f"nis_{sname}"] = nis[:, j]
            out[f"gated_{sname}"] = gated[:, j]
        return pd.DataFrame(out)

    def to_dict(self):
        return {"sensors": self.sensors, "theta": self.theta, "ridge": self.ridge, "gate": self.gate,
                "q_scale": self.q_scale, "sched": SCHED_VARS, "b": self.b.tolist(), "H": self.H.tolist(),
                "R": self.R.tolist(), "Q": self.Q.tolist(), "x0": self.x0.tolist(), "P0": self.P0.tolist()}

    @classmethod
    def from_dict(cls, d):
        k = cls(d["sensors"], d["theta"], d["ridge"], d["gate"], d["q_scale"])
        for a in ("b", "H", "R", "Q", "x0", "P0"):
            setattr(k, a, np.asarray(d[a], dtype=np.float64))
        return k


def kf_component_health(kf_out: pd.DataFrame, d_ref) -> pd.DataFrame:
    """Component HI from the filtered health parameters, with a 1-sigma from the dominating parameter."""
    out = {}
    for c, (e, f) in COMP_THETA.items():
        d = d_ref_of(d_ref, c)
        ae, af = kf_out[f"kf_{e}"].abs(), kf_out[f"kf_{f}"].abs()
        use_e = ae >= af
        out[f"kf_hi_{c}"] = (1.0 - np.maximum(ae, af) / d).clip(0, 1)
        out[f"kf_hi_{c}_sigma"] = np.where(use_e, kf_out[f"kf_{e}_sigma"], kf_out[f"kf_{f}_sigma"]) / d
    return pd.DataFrame(out, index=kf_out.index)


# ----------------------------------------------------------------------------------------------- Layer 4 inputs
FLIGHT_COLS = ["alt_mean", "Mach_mean", "TRA_mean", "T2_mean", "duration_s", "fc"]


def default_base_columns() -> list[str]:
    return [f"res_{s}" for s in XS_VARS] + [f"p_{p}" for p in PROXIES] + [f"kf_{t}" for t in T_VARS]


def source_of(feature: str) -> str:
    """Physical source of a model feature (for grouping SHAP values): sensor, proxy, KF state or flight."""
    base = feature.split("__")[0]
    for pre in ("res_", "p_", "kf_"):
        if base.startswith(pre):
            return base
    return base


class CausalFeaturizer:
    """Per-engine causal features: z-score, baseline-relative (vs median of the first `baseline_cycles`
    cycles of the same engine), rolling mean and rolling slope; plus flight descriptors and cycle count."""

    def __init__(self, base_cols=None, baseline_cycles: int = 10, roll: int = 5, use_cycle: bool = True):
        self.base_cols = list(base_cols) if base_cols else default_base_columns()
        self.flight_cols = list(FLIGHT_COLS)
        self.baseline_cycles, self.roll, self.use_cycle = baseline_cycles, roll, use_cycle
        self.mu = self.sd = None
        self.feature_names: list[str] = []
        self.seq_names: list[str] = []

    def fit(self, cyc: pd.DataFrame) -> "CausalFeaturizer":
        cols = self.base_cols + self.flight_cols
        self.mu = cyc[cols].mean().to_numpy(dtype=np.float64)
        sd = cyc[cols].std(ddof=0).to_numpy(dtype=np.float64)
        self.sd = np.where(sd < 1e-12, 1.0, sd)
        self._names()
        return self

    def _names(self):
        self.feature_names = []
        for c in self.base_cols:
            self.feature_names += [c, f"{c}__rel", f"{c}__rm", f"{c}__slope"]
        self.feature_names += self.flight_cols
        if self.use_cycle:
            self.feature_names.append("cycle")
        self.seq_names = self.base_cols + [f"{c}__rel" for c in self.base_cols] + self.flight_cols + \
            (["cycle_scaled"] if self.use_cycle else [])

    def transform(self, cyc: pd.DataFrame) -> pd.DataFrame:
        cyc = cyc.sort_values(["unit", "cycle"]).reset_index(drop=True)
        cols = self.base_cols + self.flight_cols
        zall = (cyc[cols].to_numpy(dtype=np.float64) - self.mu) / self.sd
        zall = np.nan_to_num(zall)
        nb = len(self.base_cols)
        z, zf = zall[:, :nb], zall[:, nb:]
        units = cyc["unit"].to_numpy()
        rel = np.empty_like(z)
        b = self.baseline_cycles
        for u in pd.unique(units):
            idx = np.flatnonzero(units == u)
            g = z[idx]
            base = np.empty_like(g)
            for i in range(min(len(g), b)):
                base[i] = np.median(g[: i + 1], axis=0)
            if len(g) > b:
                base[b:] = base[b - 1]
            rel[idx] = g - base
        reldf = pd.DataFrame(rel, columns=self.base_cols)
        rm = reldf.groupby(units).transform(lambda s_: s_.rolling(self.roll, min_periods=1).mean())
        slope = ((rm - rm.groupby(units).shift(self.roll)) / self.roll).fillna(0.0)
        out = {"unit": units, "cycle_id": cyc["cycle"].to_numpy()}
        for j, c in enumerate(self.base_cols):
            out[c] = z[:, j]
            out[f"{c}__rel"] = rel[:, j]
            out[f"{c}__rm"] = rm[c].to_numpy()
            out[f"{c}__slope"] = slope[c].to_numpy()
        for j, c in enumerate(self.flight_cols):
            out[c] = zf[:, j]
        if self.use_cycle:
            out["cycle"] = cyc["cycle"].to_numpy(dtype=np.float64)
            out["cycle_scaled"] = out["cycle"] / 100.0
        return pd.DataFrame(out)

    def to_dict(self):
        return {"base_cols": self.base_cols, "flight_cols": self.flight_cols, "baseline_cycles": self.baseline_cycles,
                "roll": self.roll, "use_cycle": self.use_cycle, "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "feature_names": self.feature_names, "seq_names": self.seq_names}

    @classmethod
    def from_dict(cls, d):
        f = cls(d["base_cols"], d["baseline_cycles"], d["roll"], d["use_cycle"])
        f.flight_cols = d["flight_cols"]
        f.mu, f.sd = np.asarray(d["mu"]), np.asarray(d["sd"])
        f._names()
        return f


def build_windows(frame: pd.DataFrame, cols: list[str], window: int):
    """Causal windows over cycles: window i = the `window` cycles ending at row i of the same engine (start of
    life padded by repeating the first cycle). Returns (big, starts); window i = big[starts[i]:starts[i]+window]."""
    return build_windows_array(frame[cols].to_numpy(np.float32), frame["unit"].to_numpy(), window)


def build_windows_array(seq: np.ndarray, units: np.ndarray, window: int):
    """build_windows on an array whose rows are grouped by engine and in time order within each engine."""
    starts = np.empty(len(seq), dtype=np.int64)
    chunks, offset = [], 0
    for u in pd.unique(units):
        idx = np.flatnonzero(units == u)
        g = seq[idx]
        padded = np.concatenate([np.repeat(g[:1], window - 1, axis=0), g], axis=0)
        starts[idx] = offset + np.arange(len(g))
        chunks.append(padded)
        offset += len(padded)
    return np.concatenate(chunks, axis=0), starts


def gather_windows(big: np.ndarray, starts: np.ndarray, window: int, rows) -> np.ndarray:
    rows = np.asarray(rows)
    return big[starts[rows][:, None] + np.arange(window)[None, :]]


def run_layers_1_to_3(cyc: pd.DataFrame, ref: HealthyReference, kf: GasPathKF, d_ref) -> pd.DataFrame:
    """Per-cycle table -> residuals (Layer 2) -> Kalman filter states and component HI (Layer 3)."""
    cyc = add_residuals(cyc.sort_values(["unit", "cycle"]).reset_index(drop=True), ref)
    k = kf.run(cyc)
    return pd.concat([cyc, k, kf_component_health(k, d_ref)], axis=1)


# ----------------------------------------------------------------------------------------------- post-processing
def clip_target(tgt: str, v) -> np.ndarray:
    return np.clip(v, 0, None) if tgt == "rul" else np.clip(v, 0, 1)


def causal_smooth(units, cycles, values, alpha: float, kind: str = "hi") -> np.ndarray:
    """Per-engine causal exponential smoothing of per-cycle predictions (rows sorted by unit, cycle; alpha=1: off).
    For RUL the previous estimate is first propagated by the known dynamics: RUL drops by 1 per flight cycle."""
    v = np.asarray(values, dtype=np.float64).copy()
    if alpha >= 1.0 or len(v) == 0:
        return v
    units, cycles = np.asarray(units), np.asarray(cycles, dtype=np.float64)
    for u in pd.unique(units):
        idx = np.flatnonzero(units == u)
        est = v[idx[0]]
        for k in range(1, len(idx)):
            prior = est - (cycles[idx[k]] - cycles[idx[k - 1]]) if kind == "rul" else est
            est = alpha * v[idx[k]] + (1.0 - alpha) * prior
            v[idx[k]] = est
    return v


def smooth_target(tgt: str, units, cycles, values, alpha: float) -> np.ndarray:
    return clip_target(tgt, causal_smooth(units, cycles, clip_target(tgt, values), alpha,
                                          "rul" if tgt == "rul" else "hi"))


# ----------------------------------------------------------------------------------------------- 0.1 Hz CNN inputs
SEQ_EVERY = 10          # keep every 10th 1 Hz row -> 0.1 Hz, as in Arias Chao et al. (hybrid physics + CNN)


def downsample_1hz(df: pd.DataFrame, every: int = SEQ_EVERY) -> pd.DataFrame:
    """Every `every`-th row of each (unit, cycle), keeping time order: unit, cycle, W, X_s (float32)."""
    pos = df.groupby(["unit", "cycle"], sort=False).cumcount().to_numpy()
    out = df.loc[pos % every == 0, ["unit", "cycle"] + W_VARS + XS_VARS].reset_index(drop=True)
    out[W_VARS + XS_VARS] = out[W_VARS + XS_VARS].astype(np.float32)
    out["unit"], out["cycle"] = out["unit"].astype(np.int64), out["cycle"].astype(np.int64)
    return out


def cycle_rows(samples: pd.DataFrame, cyc: pd.DataFrame) -> np.ndarray:
    """Row of `cyc` holding each sample's (unit, cycle), -1 if absent."""
    key = lambda d: d["unit"].to_numpy(np.int64) * 1_000_000 + d["cycle"].to_numpy(np.int64)   # noqa: E731
    return pd.Index(key(cyc)).get_indexer(key(samples))


class SampleEncoder:
    """CNN input channels for each 0.1 Hz sample: scaled flight condition, raw sensors, residual vs the healthy-engine
    map (Layer 2) and the engine's Kalman-filter health parameters for that cycle (Layer 3), z-scored on training
    samples. `occlude` replaces one sensor by its healthy-engine value (residual 0) for the occlusion explainer."""

    def __init__(self, sensors=XS_VARS, theta=T_VARS):
        self.sensors, self.theta = list(sensors), list(theta)
        self.mu = self.sd = None

    @property
    def channels(self) -> list[str]:
        return [f"w_{v}" for v in W_VARS] + [f"x_{s}" for s in self.sensors] + [f"res_{s}" for s in self.sensors] + \
            [f"kf_{t}" for t in self.theta]

    def raw(self, samples: pd.DataFrame, cyc: pd.DataFrame, ref: HealthyReference, occlude: str | None = None,
            chunk: int = 500_000):
        """-> (channels float32 (n, C), units (n,), cyc rows (n,)) for the samples whose cycle is in `cyc`."""
        rows = cycle_rows(samples, cyc)
        keep = np.flatnonzero(rows >= 0)
        rows = rows[keep]
        w_all = samples[W_VARS].to_numpy(np.float32)
        x_all = samples[self.sensors].to_numpy(np.float32)
        kf = cyc[[f"kf_{t}" for t in self.theta]].to_numpy(np.float64)
        n_w, n_s = len(W_VARS), len(self.sensors)
        out = np.empty((len(keep), len(self.channels)), dtype=np.float32)
        for i in range(0, len(keep), chunk):
            sel = keep[i:i + chunk]
            w, x = w_all[sel].astype(np.float64), x_all[sel].astype(np.float64)
            ideal = ref.predict(w)
            res = x / np.where(np.abs(ideal) < 1e-9, np.nan, ideal) - 1.0
            if occlude is not None:
                j = self.sensors.index(occlude)
                x[:, j], res[:, j] = ideal[:, j], 0.0
            out[i:i + chunk] = np.column_stack([scaled_w(w), x, np.nan_to_num(res), kf[rows[i:i + chunk]]])
        return out, samples["unit"].to_numpy()[keep], rows

    def fit(self, raw: np.ndarray, mask: np.ndarray, chunk: int = 500_000) -> "SampleEncoder":
        s = np.zeros(raw.shape[1])
        s2 = np.zeros(raw.shape[1])
        n = 0
        for i in range(0, len(raw), chunk):
            r = raw[i:i + chunk][mask[i:i + chunk]].astype(np.float64)
            s, s2, n = s + r.sum(0), s2 + (r ** 2).sum(0), n + len(r)
        self.mu = s / n
        sd = np.sqrt(np.clip(s2 / n - self.mu ** 2, 0, None))
        self.sd = np.where(sd < 1e-12, 1.0, sd)
        return self

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        raw -= self.mu.astype(np.float32)
        raw /= self.sd.astype(np.float32)
        return raw

    def transform(self, samples, cyc, ref, occlude=None):
        raw, units, rows = self.raw(samples, cyc, ref, occlude)
        return self.normalize(raw), units, rows

    def to_dict(self):
        return {"sensors": self.sensors, "theta": self.theta, "channels": self.channels,
                "mu": self.mu.tolist(), "sd": self.sd.tolist()}

    @classmethod
    def from_dict(cls, d):
        e = cls(d["sensors"], d["theta"])
        e.mu, e.sd = np.asarray(d["mu"], dtype=np.float64), np.asarray(d["sd"], dtype=np.float64)
        return e


def cycle_mean(rows: np.ndarray, values: np.ndarray, n_rows: int) -> np.ndarray:
    """Average per-sample predictions (n, k) into per-cycle rows (n_rows, k); NaN for cycles without samples."""
    values = np.asarray(values, dtype=np.float64).reshape(len(rows), -1)
    cnt = np.bincount(rows, minlength=n_rows).astype(np.float64)
    sums = np.column_stack([np.bincount(rows, weights=values[:, j], minlength=n_rows) for j in range(values.shape[1])])
    return sums / np.where(cnt == 0, np.nan, cnt)[:, None]


def cnn_cycle_predictions(predict_windows, enc: SampleEncoder, samples: pd.DataFrame, cyc: pd.DataFrame,
                          ref: HealthyReference, window: int, occlude: str | None = None, batch: int = 8192):
    """Per-cycle CNN outputs (len(cyc), k): mean over the 0.1 Hz windows ending inside each flight cycle.
    `predict_windows` maps float32 (b, window, C) -> (b, k) in target units."""
    ch, units, rows = enc.transform(samples, cyc, ref, occlude)
    if len(ch) == 0:
        return None
    big, starts = build_windows_array(ch, units, window)
    del ch
    out = [predict_windows(gather_windows(big, starts, window, np.arange(i, min(i + batch, len(rows)))))
           for i in range(0, len(rows), batch)]
    return cycle_mean(rows, np.concatenate(out), len(cyc))
