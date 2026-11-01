"""NASA N-CMAPSS physics-informed pipeline, implemented as specified in pipeline.md.

Layers (all causal per engine, one prediction per completed flight = flight-end cadence):
  1  ingestion          inspect_h5, load_h5 (columns mapped BY NAME), validate_split, aggregate_cycles
  2  physics            physics_1hz (pressure ratios, efficiency proxies) + HealthyReference g(W) (healthy-condition
                        baseline) -> residual = measured / healthy - 1
  3  state estimation   GasPathEKF: reduced-order observation model  res = b(W) + B(W) x + v,  random-walk state x,
                        Kalman update with Joseph covariance, innovation gating; states that the sensors cannot
                        separate are merged (observability analysis)
  4  features           CausalFeaturizer: current values + trailing-window mean/std/min/max/slope, exposure

This ONE module is used by the Colab training notebook (inlined) and by the backend, so training and serving compute
identical features. Dependencies: numpy, pandas (+ h5py for reading .h5 files).

Conventions: hidden degradation parameters `T` are used ONLY offline (labels, physics calibration, validation), never
as online inputs; `Y` (RUL) is only a target. Health index 1.0 = healthy, 0.0 = failure-level damage of that component.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd

AGG_VERSION = 2          # bump when load_h5 / aggregate_cycles change: invalidates cached aggregates

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
UNITS = {"alt": "ft", "Mach": "-", "TRA": "%", "T2": "degR", "T": "degR", "P": "psia", "N": "rpm", "W": "lbm/s"}

GAMMA = 1.4
K_COLD = (GAMMA - 1.0) / GAMMA
GAMMA_HOT = 1.33
K_HOT = (GAMMA_HOT - 1.0) / GAMMA_HOT

# Physics proxies computed at 1 Hz from MEASURED signals only, and the sensors each one depends on (occlusion explainer).
PROXY_DEPS = {
    "fan_pr": ["P21", "P2"], "lpc_pr": ["P24", "P21"], "hpc_pr": ["Ps30", "P24"],
    "eta_core": ["P24", "P2", "T24"], "eta_hpc": ["Ps30", "P24", "T30", "T24"],
    "burner_dp": ["P40", "Ps30"], "lpt_tr": ["T50", "T48"], "turb_pr": ["P40", "P50"],
    "nf_corr": ["Nf"], "nc_corr": ["Nc", "T24"], "wf_corr": ["Wf", "P2"], "spool_ratio": ["Nc", "Nf"],
}
PROXIES = list(PROXY_DEPS)

# Fixed (not data-fitted) centring/scaling of the flight condition, so per-cycle polynomial means can be accumulated
# before any model is fitted.
W_CENTER = {"alt": (20000.0, 15000.0), "Mach": (0.5, 0.25), "TRA": (60.0, 20.0), "T2": (500.0, 30.0)}

DEG_R_TO_K = 5.0 / 9.0
PSIA_TO_KPA = 6.894757
FT_TO_M = 0.3048
LBM_TO_KG = 0.45359237


# ----------------------------------------------------------------------------------------------- Layer 1: ingestion
def h5_names(f, key: str) -> list[str]:
    return [str(s).strip() for s in np.array(f[key]).reshape(-1).astype("U32")]


def inspect_h5(path: str) -> dict:
    """Keys, shapes, dtypes and attributes of an HDF5 file (pipeline.md section 2.5, step 1)."""
    import h5py

    info = {"file": os.path.basename(path), "bytes": int(os.path.getsize(path)), "datasets": {}, "groups": [],
            "attrs": {}}
    with h5py.File(path, "r") as f:
        info["attrs"] = {str(k): str(v)[:200] for k, v in f.attrs.items()}

        def visit(name, obj):
            if hasattr(obj, "shape"):
                info["datasets"][name] = {"shape": [int(d) for d in obj.shape], "dtype": str(obj.dtype)}
            else:
                info["groups"].append(name)
        f.visititems(visit)
    return info


def load_h5(path: str, split: str = "dev", virtual=XV_DISPLAY) -> pd.DataFrame:
    """Read one split ("dev" or "test") of an N-CMAPSS .h5 file into a 1 Hz DataFrame (float32).

    Columns are mapped by the variable names stored in the file (W_var, X_s_var, ...), never by position; the groups
    must be row-aligned. Corrupt rows (non-finite W/X_s, sensor <= 0, unit/cycle < 1) are removed and counted in
    df.attrs["dropped_rows"]; nothing else is altered.
    """
    import h5py

    with h5py.File(path, "r") as f:
        frames, lengths = [], {}
        for grp in ("W", "X_s", "T", "A"):
            names = h5_names(f, f"{grp}_var")
            arr = np.asarray(f[f"{grp}_{split}"], dtype=np.int32 if grp == "A" else np.float32)
            if arr.ndim != 2 or arr.shape[1] != len(names):
                raise ValueError(f"{path} [{split}]: {grp} has shape {arr.shape} but {len(names)} variable names")
            lengths[grp] = arr.shape[0]
            frames.append(pd.DataFrame(arr, columns=names))
        y = np.asarray(f[f"Y_{split}"], dtype=np.float32).reshape(-1)
        lengths["Y"] = len(y)
        frames.append(pd.DataFrame({"RUL": y}))
        if virtual and f"X_v_{split}" in f:
            vnames = h5_names(f, "X_v_var")
            keep = sorted(vnames.index(v) for v in virtual if v in vnames)
            if keep:
                arr = np.asarray(f[f"X_v_{split}"][:, keep], dtype=np.float32)
                lengths["X_v"] = arr.shape[0]
                frames.append(pd.DataFrame(arr, columns=["v_" + vnames[i] for i in keep]))
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{path} [{split}]: groups are not row-aligned: {lengths}")
    df = pd.concat(frames, axis=1)
    missing = [c for c in W_VARS + XS_VARS + ["unit", "cycle"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} [{split}] is missing variables {missing}")
    ws = df[W_VARS + XS_VARS].to_numpy()
    bad = ~np.isfinite(ws).all(axis=1) | (df[XS_VARS].to_numpy() <= 0).any(axis=1) | \
        (df["unit"].to_numpy() < 1) | (df["cycle"].to_numpy() < 1)
    if bad.any():
        df = df.loc[~bad].reset_index(drop=True)
    df.attrs["dropped_rows"] = int(bad.sum())
    return df


def validate_split(df: pd.DataFrame, split: str = "dev") -> dict:
    """Schema/ordering/label checks of a loaded split (pipeline.md section 2.5, steps 3-7).

    Raises ValueError for violations the pipeline depends on (engine rows not contiguous, cycles going backwards);
    everything else is reported (e.g. RUL not monotone, missing values) so it can be inspected in the bundle.
    """
    u, c = df["unit"].to_numpy(), df["cycle"].to_numpy()
    same = u[1:] == u[:-1]
    contiguous = len(np.unique(u)) == 1 + int(np.count_nonzero(~same))
    monotone = bool(np.all(c[1:][same] >= c[:-1][same]))
    if not contiguous:
        raise ValueError(f"[{split}] rows of an engine are not contiguous")
    if not monotone:
        raise ValueError(f"[{split}] cycle numbers go backwards within an engine")
    rep = {"split": split, "rows_1hz": int(len(df)), "engines": int(len(np.unique(u))),
           "dropped_rows": int(df.attrs.get("dropped_rows", 0)), "units": sorted(int(v) for v in np.unique(u))}
    if "RUL" in df:
        y = df["RUL"].to_numpy()
        rep["rul_non_increasing_within_engine"] = not bool(np.any(same & (y[1:] > y[:-1] + 1e-6)))
        rep["rul_min"], rep["rul_max"] = float(np.nanmin(y)), float(np.nanmax(y))
    if "hs" in df:
        rep["hs_values"] = sorted(int(v) for v in np.unique(df["hs"]))
    na = df.isna().sum()
    rep["missing_values"] = {k: int(v) for k, v in na.items() if v}
    rep["sensor_range"] = {s: [float(df[s].min()), float(df[s].max())] for s in XS_VARS}
    rep["condition_range"] = {w: [float(df[w].min()), float(df[w].max())] for w in W_VARS}
    return rep


def repair_truncated_h5(path: str, workdir: str, max_missing: int = 4096):
    """Some mirrors ship an .h5 a few bytes shorter than its header records (Kaggle's DS08d-010: 32 bytes), which HDF5
    refuses to open. Copy it to `workdir` and zero-pad the copy up to the recorded size (load_h5 then drops any zeroed
    rows). Returns (path to open, repaired copy to delete afterwards or None)."""
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

    Efficiencies are isentropic PROXIES (pipeline.md 4.3): eta_core spans fan-hub + LPC (T2,P2 -> T24,P24); eta_hpc uses
    the measured static Ps30 because total P30 is not measured. No station is invented.
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


SUMMARY_PREFIXES = ("m_", "sd_", "mn_", "mx_", "ls_")   # flight mean, std, min, max, last value of each sensor


def aggregate_cycles(df: pd.DataFrame, degree: int = 3) -> pd.DataFrame:
    """Collapse 1 Hz rows into one row per (unit, flight cycle) - the flight-end cadence of pipeline.md 3.2.

    Per sensor: mean (m_), std (sd_), min (mn_), max (mx_) and last value (ls_). Also: physics-proxy means (p_),
    virtual-sensor means (v_, display only), flight descriptors, labels (rul, health parameters, hs) and the mean
    polynomial features of the flight condition (phi_). Because the healthy reference is linear in phi,
    mean(reference) = coef . mean(phi), which gives exact per-flight "healthy" sensor means without a second pass.
    Rows of a flight must be contiguous and in time order (checked by validate_split); flights are never mixed.
    """
    out = []
    vcols = [c for c in df.columns if c.startswith("v_")]
    tcols = [c for c in T_VARS if c in df.columns]
    for unit, part in df.groupby("unit", sort=True):
        cyc = part["cycle"].to_numpy()
        if np.any(np.diff(cyc) < 0):
            part = part.sort_values("cycle", kind="stable")
            cyc = part["cycle"].to_numpy()
        change = np.flatnonzero(np.diff(cyc) != 0) + 1
        starts = np.concatenate([[0], change]).astype(np.int64)
        ends = np.concatenate([change, [len(part)]]).astype(np.int64) - 1
        cnt = (ends - starts + 1).astype(np.float64)

        def rmean(a):
            a = np.asarray(a, dtype=np.float64).reshape(len(part), -1)
            ok = np.isfinite(a)
            s = np.add.reduceat(np.where(ok, a, 0.0), starts, axis=0)
            n = np.add.reduceat(ok.astype(np.float64), starts, axis=0)
            return s / np.where(n == 0, np.nan, n)

        def rmin(a):
            return np.minimum.reduceat(np.asarray(a, dtype=np.float64), starts, axis=0)

        def rmax(a):
            return np.maximum.reduceat(np.asarray(a, dtype=np.float64), starts, axis=0)

        rows = {"unit": np.full(len(starts), int(unit)), "cycle": cyc[starts].astype(int), "duration_s": cnt}
        if "Fc" in part:
            rows["fc"] = part["Fc"].to_numpy()[starts].astype(int)
        if "hs" in part:
            rows["hs"] = rmin(part["hs"].to_numpy()).astype(int)
        if "RUL" in part:
            rows["rul"] = rmin(part["RUL"].to_numpy())
        w = part[W_VARS].to_numpy(dtype=np.float64)
        wm = rmean(w)
        for j, v in enumerate(W_VARS):
            rows[f"{v}_mean"] = wm[:, j]
        rows["alt_max"] = rmax(part["alt"].to_numpy())
        rows["Mach_max"] = rmax(part["Mach"].to_numpy())
        x = part[XS_VARS].to_numpy(dtype=np.float64)
        xm, x2 = rmean(x), rmean(x ** 2)
        xmin, xmax, xlast = rmin(x), rmax(x), x[ends]
        for j, s in enumerate(XS_VARS):
            rows[f"m_{s}"] = xm[:, j]
            rows[f"sd_{s}"] = np.sqrt(np.clip(x2[:, j] - xm[:, j] ** 2, 0, None))
            rows[f"mn_{s}"] = xmin[:, j]
            rows[f"mx_{s}"] = xmax[:, j]
            rows[f"ls_{s}"] = xlast[:, j]
        pm = rmean(physics_1hz(part).to_numpy())
        for j, p in enumerate(PROXIES):
            rows[f"p_{p}"] = pm[:, j]
        if vcols:
            vm = rmean(part[vcols].to_numpy())
            for j, c in enumerate(vcols):
                rows[c] = vm[:, j]
        if tcols:
            tm = rmean(part[tcols].to_numpy())
            for j, c in enumerate(tcols):
                rows[c] = tm[:, j]
        phim = rmean(poly_features(scaled_w(w), degree))
        for j in range(phim.shape[1]):
            rows[f"phi_{j}"] = phim[:, j]
        out.append(pd.DataFrame(rows))
    return pd.concat(out, ignore_index=True).sort_values(["unit", "cycle"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- Layer 2: physics
class HealthyReference:
    """Healthy-condition baseline g(W) (pipeline.md 3.4 / 4.5): each measured sensor as a ridge-regularised degree-3
    polynomial of the flight condition (alt, Mach, TRA, T2), fitted on healthy (hs == 1) rows of TRAINING engines only.
    residual = measured / healthy - 1 removes the effect of the flight condition."""

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
    """Add ref_<sensor> (healthy-engine value) and res_<sensor> (relative residual, measured/healthy - 1)."""
    cyc = cyc.copy()
    phi_cols = [f"phi_{j}" for j in range(ref.coef.shape[0])]
    exp = ref.expected_from_phi(cyc[phi_cols].to_numpy())
    for j, s in enumerate(ref.sensors):
        cyc[f"ref_{s}"] = exp[:, j]
        cyc[f"res_{s}"] = cyc[f"m_{s}"] / np.where(np.abs(exp[:, j]) < 1e-9, np.nan, exp[:, j]) - 1.0
    return cyc


def residual_by_condition(cyc: pd.DataFrame, n_bins: int = 3) -> pd.DataFrame:
    """Healthy-phase residual mean/std per sensor by altitude band (inspect the baseline by operating range, 4.5)."""
    h = cyc[cyc["hs"] == 1] if "hs" in cyc else cyc
    bins = pd.qcut(h["alt_mean"], n_bins, duplicates="drop")
    rows = []
    for b, g in h.groupby(bins, observed=True):
        for s in XS_VARS:
            r = g[f"res_{s}"]
            rows.append({"altitude_band": str(b), "sensor": s, "mean_pct": 100 * float(r.mean()),
                         "std_pct": 100 * float(r.std()), "cycles": int(len(g))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- health labels / mapping
def component_damage(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Per-component damage = larger of |efficiency modifier| and |flow modifier|."""
    return pd.DataFrame({c: np.maximum(df[prefix + e].abs(), df[prefix + f].abs()) for c, (e, f) in COMP_THETA.items()},
                        index=df.index)


def fit_damage_reference(cyc_train: pd.DataFrame, q: float = 0.95, floor: float = 1e-3) -> dict:
    """Failure-level damage per component: q-quantile over TRAINING engines of that component's damage at end of
    life. Defines the health-index mapping HI = 1 - damage / d_ref (validated against the truth in the notebook)."""
    last = cyc_train.sort_values("cycle").groupby("unit").tail(1)
    dmg = component_damage(last)
    return {c: max(float(np.quantile(dmg[c], q)), floor) for c in COMPONENTS}


def d_ref_of(d_ref, comp: str) -> float:
    return float(d_ref[comp]) if isinstance(d_ref, dict) else float(d_ref)


def component_health(df: pd.DataFrame, d_ref) -> pd.DataFrame:
    """TRUE health index from the hidden parameters T (offline: validation of the mapping only)."""
    dmg = component_damage(df)
    return pd.DataFrame({f"hi_{c}": (1.0 - dmg[c] / d_ref_of(d_ref, c)).clip(0.0, 1.0) for c in COMPONENTS},
                        index=df.index)


def kf_component_health(kf_out: pd.DataFrame, d_ref) -> pd.DataFrame:
    """Health index from the EKF parameter estimates kf_<param> (online), with a 1-sigma from the dominating parameter."""
    out = {}
    for c, (e, f) in COMP_THETA.items():
        d = d_ref_of(d_ref, c)
        ae, af = kf_out[f"kf_{e}"].abs(), kf_out[f"kf_{f}"].abs()
        use_e = ae >= af
        out[f"kf_hi_{c}"] = (1.0 - np.maximum(ae, af) / d).clip(0, 1)
        out[f"kf_hi_{c}_sigma"] = np.where(use_e, kf_out[f"kf_{e}_sigma"], kf_out[f"kf_{f}_sigma"]) / d
    return pd.DataFrame(out, index=kf_out.index)


# ----------------------------------------------------------------------------------------------- Layer 3: EKF
SCHED_VARS = ["alt_mean", "Mach_mean", "TRA_mean"]


def schedule(cyc: pd.DataFrame) -> np.ndarray:
    """Operating-point variables of the influence coefficients: [1, alt, Mach, TRA] (fixed scaling)."""
    z = [np.ones(len(cyc))]
    for v, raw in zip(SCHED_VARS, ["alt", "Mach", "TRA"]):
        c, s = W_CENTER[raw]
        z.append((cyc[v].to_numpy(dtype=np.float64) - c) / s)
    return np.column_stack(z)


def robust_var(a: np.ndarray) -> np.ndarray:
    """Per-column robust variance (1.4826 * MAD)^2."""
    a = np.asarray(a, dtype=np.float64)
    med = np.nanmedian(a, axis=0)
    return (1.4826 * np.nanmedian(np.abs(a - med), axis=0)) ** 2


def unique_shares(h: np.ndarray) -> np.ndarray:
    """For each column of h (S, G): the share of its norm NOT explained by a least-squares fit on the other columns
    (1 = fully separable, ~0 = a combination of the others; a zero column scores 0)."""
    norms = np.linalg.norm(h, axis=0)
    u = np.ones(h.shape[1])
    if h.shape[1] == 1:
        return u
    for j in range(h.shape[1]):
        others = np.delete(h, j, axis=1)
        c, *_ = np.linalg.lstsq(others, h[:, j], rcond=None)
        u[j] = np.linalg.norm(h[:, j] - others @ c) / max(norms[j], 1e-300) if norms[j] > 0 else 0.0
    return u


def merge_states(hw: np.ndarray, theta: np.ndarray, merge_cos: float, min_unique: float = 0.0, active=None):
    """Observability-driven state reduction (pipeline.md 4.2 / 4.8 step 9 / 5.4).

    hw: (n_points, S, P) whitened influence matrices at sampled operating points; theta: (n, P) training values of the
    P candidate parameters; active: indices of the parameters to consider (default all; never-excited parameters are
    excluded by the caller). Two passes:
      1. pairwise: merge the two groups whose sensor signatures are most similar (median |cosine| over operating points)
         while that similarity is >= merge_cos;
      2. joint: while some group's signature is mostly a combination of the others (median unique share < min_unique),
         merge that group with the group it is most similar to.
    A merged state moves its members together along their dominant joint direction v (first right singular vector of
    theta[:, members], uncentred): theta_members ~= v * x. Returns (groups, loadings)."""
    groups = [[int(j)] for j in (range(hw.shape[2]) if active is None else active)]

    def loading(members):
        if len(members) == 1:
            return np.ones(1)
        _, _, vt = np.linalg.svd(theta[:, members], full_matrices=False)
        v = vt[0]
        return -v if (theta[:, members] @ v).mean() < 0 else v

    def columns():
        cols = np.stack([np.einsum("nsj,j->ns", hw[:, :, g], loading(g)) for g in groups], axis=2)   # (n, S, G)
        unit = cols / np.maximum(np.linalg.norm(cols, axis=1, keepdims=True), 1e-300)
        cos = np.median(np.abs(np.einsum("nsa,nsb->nab", unit, unit)), axis=0)
        np.fill_diagonal(cos, 0.0)
        return cols, cos

    while len(groups) > 1:                                          # 1. pairwise
        _, cos = columns()
        a, b = np.unravel_index(np.argmax(cos), cos.shape)
        if cos[a, b] < merge_cos:
            break
        groups[a] = sorted(groups[a] + groups[b])
        del groups[b]
    while len(groups) > 1 and min_unique > 0:                       # 2. joint collinearity
        cols, cos = columns()
        uniq = np.median([unique_shares(c) for c in cols], axis=0)
        g = int(np.argmin(uniq))
        if uniq[g] >= min_unique:
            break
        partner = int(np.argmax(cos[g]))
        a, b = min(g, partner), max(g, partner)
        groups[a] = sorted(groups[a] + groups[b])
        del groups[b]
    return groups, [loading(g) for g in groups]


class GasPathEKF:
    """Extended Kalman filter on a reduced-order, dataset-calibrated gas-path observation model (pipeline.md 4-5).

    Observation (per flight):  res = b(w) + B(w) x + v,  res = 14 relative sensor residuals vs the healthy baseline,
    w = operating point [1, alt, Mach, TRA], b and B linear in w (re-linearised at every operating point; the model is
    linear in x, so the EKF update is exact). State: random walk x_k = x_(k-1) + w_k, advanced once per FLIGHT.

    Calibration (training engines only; T used offline as allowed by 4.7):
      * b, B per candidate parameter: ridge regression of residuals on the 10 N-CMAPSS health parameters;
      * R: robust variance of healthy-phase residuals (hs == 1) after removing b(w);
      * states: observability analysis merges parameters whose sensor signatures are indistinguishable (merge_cos);
      * Q: variance of cycle-to-cycle state changes, scaled by q_scale tuned on VALIDATION engines (one-step-ahead
        predictive log-likelihood of the sensors - no labels); x0 / P0 from first flights.
    Outputs per flight: state estimate + sigma, parameter estimates kf_<param> (= A x) + sigma, prior innovations,
    per-sensor NIS and gating flags, joint NIS, and the smallest covariance eigenvalue (PSD check).
    """

    def __init__(self, sensors=XS_VARS, params=T_VARS, ridge: float = 1e-4, gate: float = 9.0, merge_cos: float = 0.95,
                 merge_unique: float = 0.15, min_excitation: float = 1e-6):
        self.sensors, self.params = list(sensors), list(params)
        self.ridge, self.gate, self.merge_cos = ridge, gate, merge_cos
        self.merge_unique, self.min_excitation = merge_unique, min_excitation
        self.b = self.H = self.H_params = self.R = self.Q0 = self.x0 = self.P0 = self.A = None
        self.offset, self.excluded = np.zeros(len(self.params)), []
        self.groups, self.state_names, self.q_scale = None, None, 1.0
        self.last_loglik = None

    # -------------------------------------------------------------- calibration
    @staticmethod
    def _design(z, th):
        return np.column_stack([z] + [th[:, [p]] * z for p in range(th.shape[1])])

    def _whitened(self, z_rows):
        sw = 1.0 / np.sqrt(self.R)
        return np.stack([np.einsum("pks,k->sp", self.H_params, zr) * sw[:, None] for zr in z_rows])   # (n, S, P)

    def fit(self, cyc: pd.DataFrame, cyc_val: pd.DataFrame | None = None, q_grid=(0.1, 0.3, 1.0, 3.0, 10.0),
            n_points: int = 200, seed: int = 0) -> "GasPathEKF":
        cyc = cyc.sort_values(["unit", "cycle"]).reset_index(drop=True)
        y = cyc[[f"res_{s}" for s in self.sensors]].to_numpy(dtype=np.float64)
        th = cyc[self.params].to_numpy(dtype=np.float64)
        z = schedule(cyc)
        x = self._design(z, th)
        coef = np.linalg.solve(x.T @ x + self.ridge * np.eye(x.shape[1]), x.T @ y)     # (K + P*K, S)
        k = z.shape[1]
        self.b = coef[:k]                                                              # (K, S)
        self.H_params = coef[k:].reshape(len(self.params), k, len(self.sensors))       # (P, K, S)
        base = y - z @ self.b
        healthy = cyc["hs"].to_numpy() == 1 if "hs" in cyc else np.zeros(len(cyc), dtype=bool)
        r = robust_var(base[healthy]) if healthy.sum() >= 30 else np.var(y - x @ coef, axis=0)
        self.R = np.maximum(np.nan_to_num(r, nan=0.0), 1e-12)
        rng = np.random.default_rng(seed)
        pts = rng.choice(len(z), size=min(n_points, len(z)), replace=False)
        excited = th.std(axis=0) > self.min_excitation     # parameters that never move in training are not states
        if not excited.any():
            raise ValueError("no health parameter varies in the training engines - nothing for the EKF to estimate")
        self.excluded = [t for t, e in zip(self.params, excited) if not e]
        self.offset = np.where(excited, 0.0, th.mean(axis=0))   # held at their (constant) training value
        self.groups, loads = merge_states(self._whitened(z[pts]), th, self.merge_cos, self.merge_unique,
                                          active=np.flatnonzero(excited))
        p, g = len(self.params), len(self.groups)
        self.A = np.zeros((p, g))
        for j, (members, v) in enumerate(zip(self.groups, loads)):
            self.A[members, j] = v
        self.state_names = ["+".join(self.params[m] for m in members) for members in self.groups]
        self.H = np.einsum("pg,pks->gks", self.A, self.H_params)                        # (G, K, S)
        states = th @ self.A                                                            # (n, G) training state values
        units = cyc["unit"].to_numpy()
        d = np.concatenate([np.diff(states[units == u], axis=0) for u in pd.unique(units)])
        self.Q0 = np.var(d, axis=0) + np.mean(d, axis=0) ** 2 + 1e-12
        first = np.stack([states[units == u][0] for u in pd.unique(units)])
        self.x0, self.P0 = first.mean(axis=0), first.var(axis=0) + 1e-8
        self.q_scale = 1.0
        if cyc_val is not None and len(cyc_val):
            ll = {}
            for q in q_grid:
                self.q_scale = float(q)
                self._filter(cyc_val)
                ll[q] = self.last_loglik
            self.q_scale = float(max(ll, key=ll.get))
            self.q_loglik = {str(k_): float(v) for k_, v in ll.items()}
        return self

    @property
    def Q(self):
        return self.q_scale * self.Q0

    # -------------------------------------------------------------- filtering
    def h_at(self, z_row):
        """Observation Jacobian B(w) (S, G) and bias b(w) (S,) at operating point z_row."""
        return np.einsum("gks,k->sg", self.H, z_row), z_row @ self.b

    def observe(self, z_row, x):
        """Predicted residuals h(w, x) = b(w) + B(w) x."""
        h, b = self.h_at(z_row)
        return b + h @ x

    def _filter(self, cyc: pd.DataFrame, res_override: np.ndarray | None = None) -> dict:
        cyc = cyc.reset_index(drop=True)
        res = cyc[[f"res_{s}" for s in self.sensors]].to_numpy(dtype=np.float64) if res_override is None \
            else np.asarray(res_override, dtype=np.float64)
        z = schedule(cyc)
        n, s, g = len(cyc), len(self.sensors), len(self.state_names)
        xs, sig, psig = np.zeros((n, g)), np.zeros((n, g)), np.zeros((n, len(self.params)))
        innov, nis = np.full((n, s), np.nan), np.full((n, s), np.nan)
        gated, nis_tot, min_eig = np.zeros((n, s), dtype=bool), np.full(n, np.nan), np.zeros(n)
        q, eye = self.Q, np.eye(g)
        ll, units = 0.0, cyc["unit"].to_numpy()
        for u in pd.unique(units):
            idx = np.flatnonzero(units == u)
            x, pm = self.x0.copy(), np.diag(self.P0)
            consec = np.zeros(s, dtype=int)
            for i in idx:
                pm = pm + np.diag(q)
                h, b = self.h_at(z[i])
                ok = np.isfinite(res[i])
                nu = res[i] - b - h @ x
                innov[i, ok] = nu[ok]
                if ok.any():
                    sfull = h[ok] @ pm @ h[ok].T + np.diag(self.R[ok])
                    try:
                        nis_tot[i] = float(nu[ok] @ np.linalg.solve(sfull, nu[ok]))
                    except np.linalg.LinAlgError:
                        pass
                for j in range(s):
                    if not ok[j]:
                        continue
                    hj = h[j]
                    v = res[i, j] - b[j] - hj @ x
                    ph = pm @ hj
                    sv = hj @ ph + self.R[j]
                    nis[i, j] = v * v / sv
                    if nis[i, j] > self.gate and consec[j] < 3:
                        gated[i, j] = True
                        consec[j] += 1
                        continue
                    consec[j] = 0
                    ll += -0.5 * (np.log(2 * np.pi * sv) + v * v / sv)
                    kg = ph / sv
                    x = x + kg * v
                    ikh = eye - np.outer(kg, hj)
                    pm = ikh @ pm @ ikh.T + self.R[j] * np.outer(kg, kg)            # Joseph form
                pm = 0.5 * (pm + pm.T)                                              # symmetrise
                xs[i], sig[i] = x, np.sqrt(np.clip(np.diag(pm), 0, None))
                psig[i] = np.sqrt(np.clip(np.einsum("pg,gh,ph->p", self.A, pm, self.A), 0, None))
                min_eig[i] = float(np.linalg.eigvalsh(pm)[0])
        self.last_loglik = ll / max(n, 1)
        return {"xs": xs, "sig": sig, "psig": psig, "innov": innov, "nis": nis, "gated": gated, "nis_tot": nis_tot,
                "min_eig": min_eig}

    def run(self, cyc: pd.DataFrame, res_override: np.ndarray | None = None) -> pd.DataFrame:
        """Filter every engine causally (rows sorted by unit, cycle)."""
        r = self._filter(cyc, res_override)
        theta = r["xs"] @ self.A.T + self.offset
        out = {}
        for j, name in enumerate(self.state_names):
            out[f"ekf_{j}"] = r["xs"][:, j]
            out[f"ekf_{j}_sigma"] = r["sig"][:, j]
        for k_, t in enumerate(self.params):
            out[f"kf_{t}"] = theta[:, k_]
            out[f"kf_{t}_sigma"] = r["psig"][:, k_]
        for j, sname in enumerate(self.sensors):
            out[f"innov_{sname}"] = r["innov"][:, j]
            out[f"nis_{sname}"] = r["nis"][:, j]
            out[f"gated_{sname}"] = r["gated"][:, j]
        out["nis_total"] = r["nis_tot"]
        out["ekf_min_eig"] = r["min_eig"]
        return pd.DataFrame(out)

    def diagnostics(self, cyc: pd.DataFrame, n_points: int = 200, seed: int = 0, weak: float | None = None) -> dict:
        """Observability report (pipeline.md 5.4). Unique share = part of a whitened sensor signature NOT explained by
        the other columns (1 = separable, ~0 = collinear), for the raw parameters and for the STATES actually estimated.
        A component is identifiable if it was excited in training, none of its states is merged with another
        component's parameters, and its state(s) are separable (share >= `weak`, default merge_unique). Merging a
        component's own efficiency and flow into one state is the intended remedy and keeps it identifiable."""
        weak = self.merge_unique if weak is None else weak
        z = schedule(cyc)
        rng = np.random.default_rng(seed)
        pts = z[rng.choice(len(z), size=min(n_points, len(z)), replace=False)]
        hw = self._whitened(pts)                                                      # (n, S, P)
        active = [k for k, t in enumerate(self.params) if t not in self.excluded]
        uniq_param = np.full(len(self.params), np.nan)
        uniq_param[active] = np.median([unique_shares(h[:, active]) for h in hw], axis=0)
        hs_ = np.einsum("nsp,pg->nsg", hw, self.A)                                    # whitened STATE Jacobians
        uniq_state = np.median([unique_shares(h) for h in hs_], axis=0)
        sv = np.median([np.linalg.svd(h, compute_uv=False) for h in hs_], axis=0)
        comp = {}
        for c, (e, f) in COMP_THETA.items():
            ie, if_ = self.params.index(e), self.params.index(f)
            states = [g for g, grp in enumerate(self.groups) if ie in grp or if_ in grp]
            if not states:
                comp[c] = {"merged_with": [], "states": [], "excited": False, "own_parameters_merged": False,
                           "min_unique_share": None, "identifiable": False,
                           "reason": "no degradation of this component in the training engines"}
                continue
            members = {m for g in states for m in self.groups[g]}
            others = sorted({cc for cc, (e2, f2) in COMP_THETA.items() if cc != c and
                             (self.params.index(e2) in members or self.params.index(f2) in members)})
            m = float(min(uniq_state[g] for g in states))
            ident = bool(not others and m >= weak)
            reason = "ok" if ident else (f"indistinguishable from {', '.join(others)}" if others
                                         else "sensor signature too close to other states")
            comp[c] = {"merged_with": others, "states": [self.state_names[g] for g in states], "excited": True,
                       "own_parameters_merged": len(states) == 1 and len(members & {ie, if_}) == 2,
                       "min_unique_share": m, "identifiable": ident, "reason": reason}
        return {"unique_share_per_parameter": {t: (None if np.isnan(v) else float(v)) for t, v in zip(self.params, uniq_param)},
                "unique_share_per_state": {n: float(v) for n, v in zip(self.state_names, uniq_state)},
                "excluded_not_excited": list(self.excluded), "states": self.state_names,
                "state_singular_values_median": [float(v) for v in sv],
                "state_condition_number_median": float(sv[0] / max(sv[-1], 1e-300)),
                "components": comp, "merge_cos": self.merge_cos, "merge_unique": self.merge_unique,
                "q_scale": self.q_scale}

    def to_dict(self):
        return {"sensors": self.sensors, "params": self.params, "ridge": self.ridge, "gate": self.gate,
                "merge_cos": self.merge_cos, "merge_unique": self.merge_unique, "min_excitation": self.min_excitation,
                "offset": np.asarray(self.offset).tolist(), "excluded": list(self.excluded), "sched": SCHED_VARS, "groups": self.groups, "state_names": self.state_names,
                "A": self.A.tolist(), "b": self.b.tolist(), "H": self.H.tolist(), "H_params": self.H_params.tolist(),
                "R": self.R.tolist(), "Q0": self.Q0.tolist(), "q_scale": self.q_scale, "x0": self.x0.tolist(),
                "P0": self.P0.tolist()}

    @classmethod
    def from_dict(cls, d):
        k = cls(d["sensors"], d["params"], d["ridge"], d["gate"], d["merge_cos"], d.get("merge_unique", 0.15),
                d.get("min_excitation", 1e-6))
        k.offset = np.asarray(d.get("offset", np.zeros(len(d["params"]))), dtype=np.float64)
        k.excluded = list(d.get("excluded", []))
        for a in ("A", "b", "H", "H_params", "R", "Q0", "x0", "P0"):
            setattr(k, a, np.asarray(d[a], dtype=np.float64))
        k.groups, k.state_names, k.q_scale = d["groups"], d["state_names"], float(d["q_scale"])
        return k


def run_layers(cyc: pd.DataFrame, ref: HealthyReference, ekf: GasPathEKF, d_ref) -> pd.DataFrame:
    """Per-flight table -> residuals (Layer 2) -> EKF states, innovations and health mapping (Layer 3)."""
    cyc = add_residuals(cyc.sort_values(["unit", "cycle"]).reset_index(drop=True), ref)
    k = ekf.run(cyc)
    return pd.concat([cyc, k, kf_component_health(k, d_ref)], axis=1)


# ----------------------------------------------------------------------------------------------- Layer 4: features
FLIGHT_COLS = ["alt_mean", "Mach_mean", "TRA_mean", "T2_mean", "alt_max", "Mach_max", "duration_s", "fc"]
EXPOSURE_COLS = ["cycle", "exposure_h", "hist_len"]
WINDOW_STATS = ("rm", "rs", "rmin", "rmax", "slope")      # trailing-window mean, std, min, max, slope (6.2)
FORBIDDEN_FEATURES = {"rul", "unit", "unit_local", "hs", "dataset", "split", "cycle_id"}


def sensor_summary_cols() -> list[str]:
    return [f"{p}{s}" for p in SUMMARY_PREFIXES for s in XS_VARS]


def default_feature_columns(n_states: int):
    """Feature blocks of pipeline.md 6.1: (temporal bases, static columns)."""
    temporal = [f"res_{s}" for s in XS_VARS] + [f"p_{p}" for p in PROXIES] + [f"ekf_{j}" for j in range(n_states)]
    static = sensor_summary_cols() + [f"ekf_{j}_sigma" for j in range(n_states)] + [f"innov_{s}" for s in XS_VARS]
    return temporal, static


def feature_group(name: str) -> str:
    """Block of a feature: raw | res | phys | ekf | innov | cond (flight condition / exposure)."""
    base = name.split("__")[0]
    if base in FLIGHT_COLS or base in EXPOSURE_COLS:
        return "cond"
    for pre, grp in (("res_", "res"), ("p_", "phys"), ("ekf_", "ekf"), ("innov_", "innov")):
        if base.startswith(pre):
            return grp
    if base.startswith(SUMMARY_PREFIXES):
        return "raw"
    return "other"


def is_window_feature(name: str) -> bool:
    return "__" in name


def source_of(feature: str) -> str:
    """Physical source of a model feature (for grouping SHAP values)."""
    return feature.split("__")[0]


def audit_features(names) -> list[str]:
    """Leakage audit (6.3): no target, hidden parameters, identifiers or health flags among the model inputs."""
    bad = [n for n in names if n.split("__")[0] in FORBIDDEN_FEATURES or n.split("__")[0] in T_VARS
           or n.startswith(("hi_", "truth_", "kf_hi_", "rul"))]
    if bad:
        raise ValueError(f"leaking features: {bad}")
    return list(names)


class CausalFeaturizer:
    """Causal per-engine features (pipeline.md 6): current values of all blocks, trailing-window mean / std / min / max
    / slope of the temporal bases over `roll` flights, flight condition and operating exposure (cycle count, cumulative
    flight hours). Minimum-history rule: windows use the flights available so far (std = 0 and slope = 0 with fewer
    than 2 flights; `hist_len` tells the model how many). Missing values are filled with training medians."""

    def __init__(self, temporal_cols, static_cols, roll: int = 5):
        self.temporal_cols, self.static_cols, self.roll = list(temporal_cols), list(static_cols), int(roll)
        self.fill: dict = {}
        self.feature_names = [c for b in self.temporal_cols for c in [b] + [f"{b}__{st}" for st in WINDOW_STATS]] + \
            self.static_cols + FLIGHT_COLS + EXPOSURE_COLS
        audit_features(self.feature_names)

    def _build(self, cyc: pd.DataFrame) -> pd.DataFrame:
        cyc = cyc.sort_values(["unit", "cycle"]).reset_index(drop=True)
        units = cyc["unit"].to_numpy()
        a = cyc[self.temporal_cols].to_numpy(dtype=np.float64)
        w = self.roll
        stats = {st: np.zeros_like(a) for st in WINDOW_STATS}
        hist = np.zeros(len(cyc))
        expo = np.zeros(len(cyc))
        for u in pd.unique(units):
            idx = np.flatnonzero(units == u)
            g = pd.DataFrame(a[idx])
            roll = g.rolling(w, min_periods=1)
            stats["rm"][idx] = roll.mean().to_numpy()
            stats["rs"][idx] = roll.std(ddof=0).to_numpy()
            stats["rmin"][idx] = roll.min().to_numpy()
            stats["rmax"][idx] = roll.max().to_numpy()
            x = a[idx]
            sl = np.zeros_like(x)
            for i in range(len(idx)):
                win = x[max(0, i - w + 1): i + 1]
                if len(win) >= 2:
                    t = np.arange(len(win)) - (len(win) - 1) / 2.0
                    sl[i] = (t[:, None] * (win - np.nanmean(win, axis=0))).sum(axis=0) / (t ** 2).sum()
            stats["slope"][idx] = sl
            hist[idx] = np.minimum(np.arange(1, len(idx) + 1), w)
            expo[idx] = np.cumsum(cyc["duration_s"].to_numpy(dtype=np.float64)[idx]) / 3600.0
        out = {"unit": units, "cycle_id": cyc["cycle"].to_numpy()}
        for j, b in enumerate(self.temporal_cols):
            out[b] = a[:, j]
            for st in WINDOW_STATS:
                out[f"{b}__{st}"] = stats[st][:, j]
        for c in self.static_cols + FLIGHT_COLS:
            out[c] = cyc[c].to_numpy(dtype=np.float64) if c in cyc else np.full(len(cyc), np.nan)
        out["cycle"] = cyc["cycle"].to_numpy(dtype=np.float64)
        out["exposure_h"], out["hist_len"] = expo, hist
        return pd.DataFrame(out)

    def fit(self, cyc_train: pd.DataFrame) -> "CausalFeaturizer":
        f = self._build(cyc_train)[self.feature_names]
        self.fill = {c: float(v) if np.isfinite(v) else 0.0 for c, v in f.median().items()}
        return self

    def transform(self, cyc: pd.DataFrame) -> pd.DataFrame:
        out = self._build(cyc)
        if self.fill:
            out[self.feature_names] = out[self.feature_names].fillna(self.fill)
        return out

    def to_dict(self):
        return {"temporal_cols": self.temporal_cols, "static_cols": self.static_cols, "roll": self.roll,
                "fill": self.fill, "feature_names": self.feature_names}

    @classmethod
    def from_dict(cls, d):
        f = cls(d["temporal_cols"], d["static_cols"], d["roll"])
        f.fill = d.get("fill", {})
        return f


# ----------------------------------------------------------------------------------------------- robustness test helper
def inject_sensor_fault(cyc: pd.DataFrame, ref: HealthyReference, sensor: str, rel_bias: float,
                        start_frac: float = 0.5, every: int | None = None) -> pd.DataFrame:
    """Copy of a per-flight table with a fault in sensor `sensor`: from `start_frac` of each engine's life its flight
    mean (and min/max/last) is scaled by (1 + rel_bias) - every flight (persistent bias) or every `every`-th flight
    (spikes). Residuals are recomputed; physics proxies are left untouched (a conservative test). Adds `faulted`.
    Used on REAL test engines only to test robustness - nothing is fitted on it."""
    out = cyc.sort_values(["unit", "cycle"]).reset_index(drop=True)
    pos = out.groupby("unit").cumcount().to_numpy()
    size = out.groupby("unit")["cycle"].transform("size").to_numpy()
    faulted = pos >= start_frac * size
    if every:
        faulted &= (pos % every == 0)
    for pre in ("m_", "mn_", "mx_", "ls_"):
        col = f"{pre}{sensor}"
        if col in out:
            out.loc[faulted, col] = out.loc[faulted, col] * (1.0 + rel_bias)
    out = add_residuals(out, ref)
    out["faulted"] = faulted
    return out
