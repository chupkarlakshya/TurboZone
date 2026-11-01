# %% [markdown]
# # GENZONE | AI Manthan - N-CMAPSS physics-informed EKF + stacking ensemble (training, per `pipeline.md`)
# Trains the pipeline specified in `pipeline.md` from the raw NASA N-CMAPSS files and exports one **model bundle**:
#
# ```text
# N-CMAPSS HDF5 -> schema inspection + validation -> flight-end aggregation -> engine split (train / validation / test)
#   -> healthy-condition baseline g(W) (cross-fitted residuals) -> reduced-order gas-path model + EKF (states, covariance,
#   innovations) -> causal features -> Random Forest, XGBoost, LightGBM, CatBoost -> out-of-fold predictions
#   -> scaler + Ridge -> RUL + empirical interval -> evaluation, ablation E0-E7, SHAP + Ridge contributions -> bundle
#   (+ sections 14 / 14b, EXTENSIONS beyond pipeline.md: component-health forecasts and component RUL)
# ```
#
# **Run on Google Colab (full guide: `training/README.md`)**
# 1. A CPU runtime is enough (no GPU is used). High-RAM helps for `RUN = "full"`.
# 2. Edit the configuration cell: `RUN = "quick"` (DS02 only, check the pipeline) or `RUN = "full"` (9 files).
# 3. Run all cells and accept the Google Drive prompt (cache of aggregated data, so re-runs skip the download).
# 4. Data comes from Kaggle (`bishals098/nasa-cmapss-2-engine-degradation`, public); add Colab secrets
#    `KAGGLE_USERNAME` / `KAGGLE_KEY` only if the anonymous download fails.
# 5. The last cell downloads `genzone_ncmapss_<RUN>_bundle.zip` (a copy goes to Drive). Unzip it into `backend/models/`
#    and run `python -m backend.check_parity`.
#
# **Rules kept (pipeline.md section 0):** engines are split before anything is fitted; the official test engines are used
# once, for the final evaluation; hidden degradation parameters `T` are used only offline (physics calibration, labels for
# validating the health mapping), never as inputs; `Y` is only the target; all features are causal; nothing is fitted
# on synthetic data; no guessed EKF values - everything is estimated from training data and saved in the bundle.

# %% [markdown]
# ## 0. Setup

# %%
import gc
import glob
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import warnings
import zlib

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = os.path.exists("/kaggle/input")
T_START = time.time()
TIMINGS = {}


def stage(name):
    print(f"\n=== {name}   [{(time.time() - T_START) / 60:.1f} min elapsed]")


if IN_COLAB or IN_KAGGLE:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "xgboost", "lightgbm", "catboost", "shap", "h5py",
                    "kagglehub", "pyarrow"], check=False)
    try:    # make Colab secrets visible to kagglehub (only needed if the anonymous download fails)
        from google.colab import userdata
        for k in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
            try:
                os.environ[k] = userdata.get(k)
            except Exception:
                pass
    except Exception:
        pass

# %% [markdown]
# ## 1. Configuration (edit this cell)

# %%
RUN = os.environ.get("RUN", "quick")                          # "quick" (DS02) or "full" (9 files; DS08d excluded)
DATASET_HANDLE = "bishals098/nasa-cmapss-2-engine-degradation"
FILES = {"quick": ["N-CMAPSS_DS02-006.h5"],
         "full": ["N-CMAPSS_DS01-005.h5", "N-CMAPSS_DS02-006.h5", "N-CMAPSS_DS03-012.h5", "N-CMAPSS_DS04.h5",
                  "N-CMAPSS_DS05.h5", "N-CMAPSS_DS06.h5", "N-CMAPSS_DS07.h5", "N-CMAPSS_DS08a-009.h5",
                  "N-CMAPSS_DS08c-008.h5"]}[RUN]
# DS08d-010 is excluded: Kaggle's copy is truncated inside HDF5's internal index (cannot be read even after padding).
DATA_DIR = os.environ.get("NCMAPSS_DIR")                      # optional folder that already holds the .h5 files
DELETE_AFTER_USE = True                                       # delete each downloaded .h5 after aggregation (disk)
USE_DRIVE = True                                              # Colab: mount Drive for the aggregate cache
DRIVE_DIR = "/content/drive/MyDrive/genzone"                  # cache and a copy of the bundle go here
CACHE_AGGREGATES = True                                       # reuse per-file aggregates from earlier runs
HEALTHY_SAMPLES_PER_FILE = 200_000                            # 1 Hz healthy rows per file for the healthy baseline
SEED = 42
VAL_FRACTION = 0.2                                            # share of development engines held out for validation
N_FOLDS = 5                                                   # engine-grouped folds for out-of-fold predictions
ROLL = 5                                                      # trailing window (flights) for temporal features
MERGE_COS = 0.95                                              # EKF: merge states whose sensor signatures are this similar
MERGE_UNIQUE = 0.15                                           # EKF: merge a state whose signature is mostly the others'
EKF_GATE = 9.0                                                # EKF: per-sensor NIS gate (reject up to 3 flights in a row)
Q_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)                           # EKF: process-noise scales tried on validation engines
TUNE = True                                                   # small validation-engine grids for the four models
EKF_IN_RUL = "auto"                                           # RUL feature set: "auto" (validation engines choose), True (spec's
EKF_SELECT_MARGIN = 0.02                                      #   full list incl. EKF features) or False (without EKF features)
EKF_SELECT_MIN_VAL_ENGINES = 5                                # "auto" only chooses with >= this many validation engines;
                                                              #   it keeps the full list unless "no EKF" wins by > MARGIN
INTERVAL_LEVEL = 0.90                                         # empirical RUL interval from validation residuals
RELIABLE_MAX_RMSE = 0.10                                      # component health shown as reliable if validation RMSE on
RELIABLE_MIN_CYCLES = 30                                      #   degraded flights (HI < 0.9) <= 10 points (>= 30 flights)
RUN_ABLATION = True                                           # E0-E7 (pipeline.md section 10.2)
RUN_FORECAST = True                                           # EXTENSION: component-health forecasts (section 14)
FORECAST_HORIZONS = (10, 25, 50)                              # flights ahead
FORECAST_DEGRADED_BELOW = 0.9                                 # score forecasts on flights whose current true health < this
RUN_COMPONENT_RUL = True                                      # EXTENSION: component RUL (section 14b)
COMP_FAIL_HI = 0.30                                           # component "failure level" = the Emergency band edge (BAND_EDGES[2])
COMP_RUL_CAP = 100.0                                          # component RUL is capped at this many flights
COMP_RUL_SHOW_BELOW = 0.9                                     # the dashboard shows it only while the EKF health is below this
SHAP_ROWS = 300                                               # training rows explained per model (Random Forest: <= 100)
BAND_EDGES = [0.75, 0.55, 0.30]                               # Normal / Warning / Alarm / Emergency health bands
BUILD_INFO = {"commit": None, "code_sha256": None}            # filled in by training/build_notebook.py

# %% [markdown]
# ## 2. Google Drive (aggregate cache)

# %%
DRIVE_OK = False
if USE_DRIVE and IN_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        os.makedirs(DRIVE_DIR, exist_ok=True)
        DRIVE_OK = True
    except Exception as exc:
        print("Google Drive not mounted -> no cache this session:", repr(exc))
CACHE_DIR = os.environ.get("NCMAPSS_CACHE") or (DRIVE_DIR if DRIVE_OK else None)
if CACHE_DIR:
    os.makedirs(CACHE_DIR, exist_ok=True)
print("cache folder:", CACHE_DIR or "none (nothing is reused between sessions)")

# %% [markdown]
# ## 3. Shared pipeline code (identical to backend/pipeline/ncmapss.py and stacking.py - do not edit here)

# %% [pipeline:ncmapss]
from backend.pipeline import ncmapss as nc  # noqa: E402

# %%
if "nc" not in globals():          # in the notebook the module cell defines everything at top level
    import types
    nc = types.SimpleNamespace(**{k: v for k, v in globals().items() if not k.startswith("_")})

# %% [pipeline:stacking]
from backend.pipeline import stacking as st  # noqa: E402

# %%
if "st" not in globals():
    import types
    st = types.SimpleNamespace(**{k: v for k, v in globals().items() if not k.startswith("_")})

# %% [markdown]
# ## 4. Pre-flight: files on Kaggle, sizes, disk

# %%
EXPECTED_BYTES = {                                            # sizes listed by Kaggle (checked 2026-10)
    "N-CMAPSS_DS01-005.h5": 2873351432, "N-CMAPSS_DS02-006.h5": 2450472504, "N-CMAPSS_DS03-012.h5": 3693395776,
    "N-CMAPSS_DS04.h5": 3752493952, "N-CMAPSS_DS05.h5": 2599166216, "N-CMAPSS_DS06.h5": 2549159720,
    "N-CMAPSS_DS07.h5": 2714714776, "N-CMAPSS_DS08a-009.h5": 3236762200, "N-CMAPSS_DS08c-008.h5": 2413078176,
    "N-CMAPSS_DS08d-010.h5": 2885034848}                      # DS08d-010 is 32 bytes short of its own HDF5 header


def remote_sizes():
    try:
        import urllib.request
        with urllib.request.urlopen(f"https://www.kaggle.com/api/v1/datasets/list/{DATASET_HANDLE}", timeout=30) as r:
            return {f["name"]: int(f["totalBytes"]) for f in json.load(r)["datasetFiles"]}
    except Exception as exc:
        print("could not list the Kaggle dataset (continuing with the built-in size table):", repr(exc))
        return {}


REMOTE = remote_sizes()
print(f"RUN={RUN}: {len(FILES)} file(s)")
for f_ in FILES:
    exp = REMOTE.get(f_) or EXPECTED_BYTES.get(f_)
    print(f"  {f_:26s} {exp / 1e9 if exp else float('nan'):5.2f} GB   on Kaggle: {f_ in REMOTE}")
free_gb = shutil.disk_usage(os.getcwd()).free / 1e9
print(f"free disk: {free_gb:.0f} GB (one file at a time needs ~4 GB)", "" if free_gb > 6 else "  <-- LOW")

# %% [markdown]
# ## 5. Layer 1: download, schema inspection + validation, flight-end aggregation (file by file, cached)

# %%
SEARCH = [(DATA_DIR, True), ("/kaggle/input", True), (DRIVE_DIR if DRIVE_OK else None, True), ("/content", False),
          ("/content/drive/MyDrive", False), (os.getcwd(), False)]


def find_local(fname):
    for root, recursive in SEARCH:
        if not root or not os.path.isdir(root):
            continue
        pattern = os.path.join(root, "**", fname) if recursive else os.path.join(root, fname)
        hits = glob.glob(pattern, recursive=recursive)
        if hits:
            return hits[0]
    return None


def get_file(fname):
    """(path, downloaded). Local copies are never deleted; downloads are deleted after use if DELETE_AFTER_USE."""
    local = find_local(fname)
    if local:
        return local, False
    import kagglehub
    try:
        return kagglehub.dataset_download(DATASET_HANDLE, path=fname), True
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {fname} from Kaggle ({exc!r}).\nAdd Colab secrets KAGGLE_USERNAME / KAGGLE_KEY "
            "(kaggle.com > Settings > API > Create New Token), run kagglehub.login(), or set NCMAPSS_DIR.") from exc


def file_fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(4_000_000))
    return {"bytes": os.path.getsize(path), "sha256_first_4MB": h.hexdigest()}


def cache_base(fname):
    if not (CACHE_DIR and CACHE_AGGREGATES):
        return None
    return os.path.join(CACHE_DIR, f"agg_{fname.replace('.h5', '')}_v{nc.AGG_VERSION}_h{HEALTHY_SAMPLES_PER_FILE}_s{SEED}")


def cache_load(base):
    if not base or not os.path.exists(base + ".meta.json"):
        return None
    try:
        with open(base + ".meta.json") as fh:
            meta = json.load(fh)
        z = np.load(base + ".healthy.npz")
        return meta, pd.read_parquet(base + ".cycles.parquet"), z["w"], z["x"], z["u"]
    except Exception as exc:
        print("  cache unreadable, recomputing:", repr(exc))
        return None


def cache_save(base, meta, cycles, hw, hx, hu):
    if not base:
        return
    try:
        cycles.to_parquet(base + ".cycles.parquet", index=False)
        np.savez(base + ".healthy.npz", w=hw, x=hx, u=hu)
        with open(base + ".meta.json", "w") as fh:                    # written last: marks the cache entry complete
            json.dump(meta, fh)
    except Exception as exc:
        print("  could not write the cache (continuing):", repr(exc))


def process_file(fname):
    """Download (if needed), inspect, validate and aggregate one file. Unit ids stay file-local here."""
    rng_f = np.random.default_rng([SEED, zlib.crc32(fname.encode())])    # same healthy sample with or without cache
    path, downloaded = get_file(fname)
    fp = file_fingerprint(path)
    expected = REMOTE.get(fname) or EXPECTED_BYTES.get(fname)
    if expected and fp["bytes"] != expected:
        print(f"  WARNING: {fname} has {fp['bytes']} bytes, Kaggle lists {expected}")
    h5_path, repaired = nc.repair_truncated_h5(path, os.getcwd())         # Kaggle's DS08d-010 is 32 bytes short
    if repaired:
        print(f"  NOTE: {fname} is shorter than its HDF5 header records (mirror defect); opened a zero-padded copy")
    schema = {"inspect": nc.inspect_h5(h5_path), "splits": {}}
    ds = fname.replace("N-CMAPSS_", "").replace(".h5", "")
    cyc_parts, hw, hx, hu, logs = [], [], [], [], []
    for split in ("dev", "test"):
        t1 = time.time()
        df = nc.load_h5(h5_path, split)
        rep = nc.validate_split(df, split)
        schema["splits"][split] = rep
        if rep["dropped_rows"] or rep["missing_values"] or not rep.get("rul_non_increasing_within_engine", True):
            print(f"  NOTE [{split}]: dropped {rep['dropped_rows']} corrupt rows; missing values {rep['missing_values']}; "
                  f"RUL non-increasing: {rep.get('rul_non_increasing_within_engine')}")
        agg = nc.aggregate_cycles(df)
        agg["split"], agg["dataset"], agg["unit_local"] = split, ds, agg["unit"]
        cyc_parts.append(agg)
        if split == "dev":
            h = df[df["hs"] == 1] if "hs" in df else df
            take = rng_f.choice(len(h), size=min(HEALTHY_SAMPLES_PER_FILE, len(h)), replace=False)
            hw.append(h[nc.W_VARS].to_numpy()[take])
            hx.append(h[nc.XS_VARS].to_numpy()[take])
            hu.append(h["unit"].to_numpy()[take])
            del h
        logs.append({"file": fname, "split": split, "rows_1hz": rep["rows_1hz"], "engines": rep["engines"],
                     "flights": int(len(agg)), "seconds": round(time.time() - t1, 1)})
        print("  ", logs[-1])
        del df
        gc.collect()
    if repaired:
        os.remove(repaired)
    if downloaded and DELETE_AFTER_USE:
        try:
            os.remove(path)
        except OSError:                             # read-only mount (e.g. /kaggle/input): nothing to free
            pass
    return pd.concat(cyc_parts, ignore_index=True), np.vstack(hw), np.vstack(hx), np.concatenate(hu), logs, fp, schema


stage("Layer 1: download + aggregate")
cycles, healthy_w, healthy_x, healthy_u, load_log, FINGERPRINTS, SCHEMA = [], [], [], [], [], {}, {}
for fi, fname in enumerate(FILES):
    t0 = time.time()
    base = cache_base(fname)
    hit = cache_load(base)
    if hit:
        meta, agg, hw, hx, hu = hit
        fp, logs, schema, src = meta["fingerprint"], meta["load_log"], meta["schema"], "cache"
    else:
        agg, hw, hx, hu, logs, fp, schema = process_file(fname)
        cache_save(base, {"fingerprint": fp, "load_log": logs, "schema": schema}, agg, hw, hx, hu)
        src = "downloaded"
    agg = agg.copy()
    agg["unit"] = (fi + 1) * 1000 + agg["unit_local"]               # composite engine id: unique across files
    cycles.append(agg)
    healthy_w.append(hw)
    healthy_x.append(hx)
    healthy_u.append((fi + 1) * 1000 + np.asarray(hu).astype(np.int64))
    FINGERPRINTS[fname], SCHEMA[fname] = fp, schema
    load_log.extend(logs)
    print(f"{fname}: {src}, {time.time() - t0:.0f}s")
cyc = pd.concat(cycles, ignore_index=True).sort_values(["unit", "cycle"]).reset_index(drop=True)
del cycles
assert not (set(cyc[cyc.split == "dev"].unit) & set(cyc[cyc.split == "test"].unit)), "dev/test engine overlap"
TIMINGS["download_and_aggregate_s"] = time.time() - T_START
print(f"{cyc.unit.nunique()} engines, {len(cyc)} flights")

# %% [markdown]
# ## 6. Engine split: train / validation / test - BEFORE anything is fitted (pipeline.md rule 4, 2.4)
# Test = NASA's official test engines (untouched until the final evaluation). Validation = a fixed-seed share of the
# development engines, stratified by subset, used for tuning, early stopping, the Ridge alpha and the EKF process noise.

# %%
stage("Engine split")
groups = cyc["unit"].to_numpy()
subset_of = cyc.drop_duplicates("unit").set_index("unit")["dataset"].to_dict()
dev_units = np.unique(groups[(cyc.split == "dev").to_numpy()])
train_units, val_units = st.split_engines(dev_units, {u: subset_of[u] for u in dev_units}, VAL_FRACTION, SEED)
test_units = np.unique(groups[(cyc.split == "test").to_numpy()])
cyc["partition"] = np.where(np.isin(groups, train_units), "train", np.where(np.isin(groups, val_units), "val", "test"))
train_m, val_m, test_m = [(cyc.partition == p).to_numpy() for p in ("train", "val", "test")]
tr_idx, va_idx, te_idx = np.flatnonzero(train_m), np.flatnonzero(val_m), np.flatnonzero(test_m)
assert set(np.unique(groups[te_idx])) == set(test_units) and not set(train_units) & set(val_units)
folds = st.engine_folds(groups[tr_idx], N_FOLDS)                  # positions within tr_idx
print(pd.crosstab(cyc.drop_duplicates("unit")["dataset"], cyc.drop_duplicates("unit")["partition"]))
print(f"flights: train {len(tr_idx)}, validation {len(va_idx)}, test {len(te_idx)}; {len(folds)} engine-grouped folds")

# %% [markdown]
# ## 7. Layer 2: healthy-condition baseline g(W) and residuals (cross-fitted for training engines)
# g(W) is fitted on healthy (hs == 1) rows of TRAINING engines only. Training-engine residuals use a baseline fitted
# without that engine's fold (pipeline.md 3.4), so they are as out-of-sample as validation/test residuals.

# %%
stage("Layer 2: healthy baseline")
t0 = time.time()
hw_all, hx_all, hu_all = np.vstack(healthy_w), np.vstack(healthy_x), np.concatenate(healthy_u)
del healthy_w, healthy_x, healthy_u
in_train = np.isin(hu_all, train_units)
ref = nc.HealthyReference().fit(hw_all[in_train], hx_all[in_train])          # the deployed baseline
print("healthy baseline relative RMSE per sensor (1 Hz, healthy training rows):")
print(pd.Series(ref.fit_rmse, index=nc.XS_VARS).map(lambda v: f"{100 * v:.3f}%").to_string())
cyc = nc.add_residuals(cyc, ref)
res_cols = [f"ref_{s}" for s in nc.XS_VARS] + [f"res_{s}" for s in nc.XS_VARS]
for a, b in folds:                                                            # cross-fitting for training rows
    rows = tr_idx[b]
    keep = in_train & ~np.isin(hu_all, np.unique(groups[rows]))
    sub = nc.add_residuals(cyc.iloc[rows], nc.HealthyReference().fit(hw_all[keep], hx_all[keep]))
    cyc.loc[cyc.index[rows], res_cols] = sub[res_cols].to_numpy()
del hw_all, hx_all, hu_all
print("\nhealthy-phase residuals of VALIDATION engines by altitude band (should be ~0 with small spread):")
res_band = nc.residual_by_condition(cyc[val_m])
print(res_band.pivot_table(index="sensor", columns="altitude_band", values="mean_pct").round(3).to_string())

# Health-index mapping (offline, from T): failure-level damage per component, on training engines only
D_REF = nc.fit_damage_reference(cyc[train_m])
cyc = pd.concat([cyc, nc.component_health(cyc, D_REF)], axis=1)               # TRUE health: validation only
print("\nfailure-level damage per component D_REF =", {c: round(v, 5) for c, v in D_REF.items()})
TIMINGS["layer2_s"] = time.time() - t0

# %% [markdown]
# ## 8. Layer 3: reduced-order gas-path model + EKF (states, covariance, innovations) and its diagnostics

# %%
stage("Layer 3: EKF")
t0 = time.time()
ekf = nc.GasPathEKF(gate=EKF_GATE, merge_cos=MERGE_COS, merge_unique=MERGE_UNIQUE).fit(cyc[train_m], cyc_val=cyc[val_m], q_grid=Q_GRID)
k_out = ekf.run(cyc)
cyc = pd.concat([cyc, k_out, nc.kf_component_health(k_out, D_REF)], axis=1)
OBS = ekf.diagnostics(cyc[train_m])
print(f"EKF states ({len(ekf.state_names)}): {ekf.state_names}")
print("not estimated (never moved in the training engines, held at their training value):", ekf.excluded or "none")
print(f"process-noise scale chosen on validation engines: {ekf.q_scale} (log-likelihood per flight "
      f"{getattr(ekf, 'q_loglik', {})})")
print(f"observability: median condition number of the whitened state Jacobian {OBS['state_condition_number_median']:.1f}")
print(pd.DataFrame(OBS["components"]).T.to_string())

v = cyc[val_m]
EKF_DIAG = {
    "mean_nis_per_sensor_val": {s: float(v[f"nis_{s}"].mean()) for s in nc.XS_VARS},
    "mean_joint_nis_val": float(v["nis_total"].mean()), "joint_nis_dof": len(nc.XS_VARS),
    "gated_share_val": float(v[[f"gated_{s}" for s in nc.XS_VARS]].to_numpy().mean()),
    "innovation_mean_val": {s: float(v[f"innov_{s}"].mean()) for s in nc.XS_VARS},
    "min_covariance_eigenvalue": float(cyc["ekf_min_eig"].min()),
    "efficiency_mods_improving_share": float((cyc[[f"kf_{t}" for t in nc.T_VARS if t.endswith("eff_mod")]] > 0.005)
                                             .to_numpy().mean()),
}
print(f"validation: mean joint NIS {EKF_DIAG['mean_joint_nis_val']:.1f} (expected ~{len(nc.XS_VARS)} if consistent), "
      f"gated share {EKF_DIAG['gated_share_val']:.2%}, min covariance eigenvalue {EKF_DIAG['min_covariance_eigenvalue']:.2e}")

# Validate the health-index mapping against the truth (T): validation engines decide reliability, test is reported
HEALTH_MAP, RELIABILITY = {}, {}
for c in nc.COMPONENTS:
    row = {}
    for part, m in (("val", val_m), ("test", test_m)):
        t, e = cyc.loc[m, f"hi_{c}"].to_numpy(), cyc.loc[m, f"kf_hi_{c}"].to_numpy()
        dg = t < 0.9
        row[part] = {"rmse": st.rmse(e, t), "mae": st.mae(e, t), "n_degraded": int(dg.sum()),
                     "rmse_degraded": st.rmse(e[dg], t[dg]) if dg.any() else None,
                     "corr": float(np.corrcoef(t, e)[0, 1]) if np.std(t) > 0 and np.std(e) > 0 else None}
    HEALTH_MAP[c] = row
    vd = row["val"]
    ident = OBS["components"][c]["identifiable"]
    if vd["n_degraded"] < RELIABLE_MIN_CYCLES:
        status = "insufficient_degradation_in_validation"
    else:
        status = "ok" if ident and vd["rmse_degraded"] <= RELIABLE_MAX_RMSE else "low"
    RELIABILITY[c] = {"status": status, "identifiable": ident, "merged_with": OBS["components"][c]["merged_with"],
                      "val_rmse_degraded_points": None if vd["rmse_degraded"] is None else 100 * vd["rmse_degraded"]}
print("\nhealth-index mapping vs truth (points out of 100) and reliability (decided on VALIDATION engines):")
print(pd.DataFrame({c: {"val RMSE degraded": None if r["val"]["rmse_degraded"] is None else 100 * r["val"]["rmse_degraded"],
                        "test RMSE degraded": None if r["test"]["rmse_degraded"] is None else 100 * r["test"]["rmse_degraded"],
                        "reliability": RELIABILITY[c]["status"]} for c, r in HEALTH_MAP.items()}).T.to_string())
TIMINGS["layer3_s"] = time.time() - t0

# %% [markdown]
# ## 9. Causal features (pipeline.md section 6) + leakage audit

# %%
stage("Features")
temporal_cols, static_cols = nc.default_feature_columns(len(ekf.state_names))
feat = nc.CausalFeaturizer(temporal_cols, static_cols, roll=ROLL).fit(cyc[train_m])    # medians from training only
frame = feat.transform(cyc)                                    # same row order as cyc (sorted by unit, cycle)
assert (frame["unit"].to_numpy() == groups).all()
FEATURES = nc.audit_features(feat.feature_names)
X = frame[FEATURES]
y = cyc["rul"].to_numpy(dtype=np.float64)
Xtr, ytr, Xva, yva, Xte, yte = X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx], y[va_idx], X.iloc[te_idx], y[te_idx]
blocks = pd.Series([nc.feature_group(f) for f in FEATURES]).value_counts().to_dict()
print(f"{len(FEATURES)} features by block: {blocks}; window features: {sum(map(nc.is_window_feature, FEATURES))}")

# %% [markdown]
# ## 10. Four base regressors (pipeline.md section 7): tuning on validation engines, out-of-fold predictions
# The pipeline is fitted for TWO feature sets and both are reported: **full** (the spec's feature list, EKF states /
# uncertainty / innovations included) and **no_ekf** (the same without the EKF features). `EKF_IN_RUL` decides which one
# ships: "auto" lets the VALIDATION engines choose (and keeps the spec's full list unless no_ekf wins by more than
# `EKF_SELECT_MARGIN`, and only if there are at least `EKF_SELECT_MIN_VAL_ENGINES` validation engines); True / False force it.
# Test engines never take part in this choice.

# %%
stage("Base models + Ridge stack")
t0 = time.time()
NAMES = list(st.BASE_NAMES)
FEATURE_SETS = {"full": list(FEATURES),
                "no_ekf": [f for f in FEATURES if nc.feature_group(f) not in ("ekf", "innov")]}


def fit_rul_pipeline(cols, label):
    """Tune the four models on validation engines, out-of-fold predictions, refit, Ridge stack for one feature set."""
    params, tuning = {}, []
    for name in NAMES:
        grid = st.TUNE_GRIDS[name] if TUNE else [{}]               # the boosted models always early-stop on validation
        best, table = st.tune_model(name, Xtr[cols], ytr, Xva[cols], yva, grid)
        params[name] = best
        tuning += [{**r, "feature_set": label} for r in table]
        print(f"  [{label}] {name:13s} chosen {best}  (validation RMSE {min(r['val_rmse'] for r in table):.2f}) "
              f"[{(time.time() - t0) / 60:.1f} min]")
    oof_ = st.oof_predictions(NAMES, params, Xtr[cols], ytr, folds)
    models_ = st.fit_base_models(NAMES, params, Xtr[cols], ytr)
    b_va, b_te = st.predict_bases(models_, NAMES, Xva[cols]), st.predict_bases(models_, NAMES, Xte[cols])
    stack_ = st.fit_stack(oof_, ytr, NAMES, b_va, yva)
    p_va, p_te = np.clip(stack_.predict(b_va), 0, None), np.clip(stack_.predict(b_te), 0, None)
    print(f"  [{label}] {len(cols)} features; out-of-fold RMSE per model "
          f"{ {n: round(st.rmse(oof_[:, i], ytr), 2) for i, n in enumerate(NAMES)} }; "
          f"stack validation RMSE {st.rmse(p_va, yva):.2f}")
    return {"cols": cols, "params": params, "tuning": tuning, "oof": oof_, "models": models_, "base_va": b_va,
            "base_te": b_te, "stack": stack_, "pred_va": p_va, "pred_te": p_te}


RUL_RUNS = {label: fit_rul_pipeline(cols, label) for label, cols in FEATURE_SETS.items()}
n_val_engines = int(len(val_units))
val_rmse = {k_: st.rmse(r["pred_va"], yva) for k_, r in RUL_RUNS.items()}
test_rmse = {k_: st.rmse(r["pred_te"], yte) for k_, r in RUL_RUNS.items()}       # reported, never used to choose
CHOICE, CHOICE_REASON = st.choose_feature_set(EKF_IN_RUL, val_rmse, n_val_engines, EKF_SELECT_MARGIN,
                                               EKF_SELECT_MIN_VAL_ENGINES)
print(f"\nEKF features in the RUL model? validation RMSE: full {val_rmse['full']:.2f} | no_ekf {val_rmse['no_ekf']:.2f} "
      f"({n_val_engines} validation engines)\n  -> shipping '{CHOICE}': {CHOICE_REASON}")
print(f"  (for the record, TEST RMSE: full {test_rmse['full']:.2f} | no_ekf {test_rmse['no_ekf']:.2f} - not used to choose)")

_run = RUL_RUNS[CHOICE]                                            # everything below uses the shipped feature set
RUL_FEATURES, PARAMS, TUNING = _run["cols"], _run["params"], sum((r["tuning"] for r in RUL_RUNS.values()), [])
oof, models, base_va, base_te = _run["oof"], _run["models"], _run["base_va"], _run["base_te"]
stack, pred_va, pred_te = _run["stack"], _run["pred_va"], _run["pred_te"]
TIMINGS["base_models_s"] = time.time() - t0

# %% [markdown]
# ## 11. Ridge stacking (pipeline.md section 8): scaler on OOF predictions, alpha on validation engines
# (fitted in section 10 for each feature set; the shipped one is shown here)

# %%
stage("Ridge stack")
print(f"shipped feature set: {CHOICE} ({len(RUL_FEATURES)} features)")
print(f"alpha = {stack.alpha} (validation RMSE per alpha {stack.diagnostics['alpha_val_rmse']})")
print("coefficients (scaled base predictions):", dict(zip(NAMES, np.round(stack.coef, 3))), "intercept", round(stack.intercept, 2))
print("base-prediction correlation (OOF):")
print(pd.DataFrame(stack.diagnostics["base_prediction_correlation"], index=NAMES, columns=NAMES).round(3))
print(f"validation RMSE: stack {st.rmse(pred_va, yva):.2f}; best single "
      f"{min(st.rmse(base_va[:, i], yva) for i in range(len(NAMES))):.2f}")

# %% [markdown]
# ## 12. Evaluation (pipeline.md section 10) - test engines used once, here

# %%
stage("Evaluation")
metrics = {"test_engines": int(len(test_units)), "test_flights": int(len(te_idx)),
           "validation_engines": int(len(val_units)), "train_engines": int(len(train_units))}
metrics["rul_stack"] = st.report(pred_te, yte)
metrics["ekf_in_rul"] = {
    "setting": EKF_IN_RUL, "shipped": CHOICE, "reason": CHOICE_REASON, "n_validation_engines": n_val_engines,
    "validation_rmse": val_rmse, "test_rmse": test_rmse,
    "test_r2": {k_: st.r2(r["pred_te"], yte) for k_, r in RUL_RUNS.items()},
    "n_features": {k_: len(r["cols"]) for k_, r in RUL_RUNS.items()},
    "finding": ("EKF features improve RUL on the test engines" if test_rmse["full"] < test_rmse["no_ekf"] else
                "EKF features do NOT improve RUL on the test engines (the EKF's role is component health, uncertainty "
                "and sensor checks); pipeline.md rule 9: do not claim otherwise")}
print("EKF features in RUL:", metrics["ekf_in_rul"]["finding"])
metrics["rul_stack_validation"] = st.report(pred_va, yva)
for i, n in enumerate(NAMES):
    metrics[f"rul_{n}"] = st.report(np.clip(base_te[:, i], 0, None), yte)
rows = [{"model": "Ridge stack", **{k: metrics["rul_stack"][k] for k in ("rmse", "mae", "r2", "nasa_score_mean")}}] + \
       [{"model": n, **{k: metrics[f"rul_{n}"][k] for k in ("rmse", "mae", "r2", "nasa_score_mean")}} for n in NAMES]
print("TEST engines - RUL in flight cycles:")
print(pd.DataFrame(rows).set_index("model").round(3))

# Empirical interval per predicted-RUL range (pipeline.md 9): residuals of the validation engines plus engine-held-out
# residuals of the whole stack on the training engines (more engines than validation alone; slightly pessimistic)
cv_tr = np.clip(st.cv_stack_predictions(oof, ytr, folds, NAMES, stack.alpha), 0, None)
INTERVAL = st.banded_interval(np.concatenate([pred_va, cv_tr]), np.concatenate([yva - pred_va, ytr - cv_tr]),
                              INTERVAL_LEVEL)
lo, hi = st.apply_interval(pred_te, INTERVAL)
inside = (yte >= lo) & (yte <= hi)
metrics["interval"] = {**INTERVAL, "method": "empirical: residual quantiles per predicted-RUL range (validation engines "
                                             "+ engine-held-out stack residuals of training engines)",
                       "coverage_test": float(inside.mean()), "mean_width_test": float(np.mean(hi - lo)),
                       "coverage_test_by_phase": {g: float(inside[st.rul_phase(yte) == g].mean())
                                                  for g in np.unique(st.rul_phase(yte))}}
print(f"empirical {INTERVAL_LEVEL:.0%} interval per predicted-RUL range: test coverage {inside.mean():.1%}, "
      f"mean width {np.mean(hi - lo):.1f} cycles")
print(pd.DataFrame(INTERVAL["bands"]).round(2).to_string(index=False))
print("coverage by true phase:", {k: f"{v:.0%}" for k, v in metrics["interval"]["coverage_test_by_phase"].items()})

te_ds = cyc["dataset"].to_numpy()[te_idx]
metrics["by_phase"] = st.error_by_group(pred_te, yte, st.rul_phase(yte))
metrics["by_subset"] = st.error_by_group(pred_te, yte, te_ds)
metrics["by_flight_class"] = st.error_by_group(pred_te, yte, cyc["fc"].to_numpy()[te_idx])
metrics["validation_by_engine"] = st.error_by_group(pred_va, yva, groups[va_idx])
metrics["validation_by_phase"] = st.error_by_group(pred_va, yva, st.rul_phase(yva))
per_engine = pd.DataFrame(st.error_by_group(pred_te, yte, groups[te_idx])).rename(columns={"group": "unit"})
per_engine["dataset"] = [subset_of[u] for u in per_engine["unit"]]
print("\nby degradation phase:")
print(pd.DataFrame(metrics["by_phase"]).round(2).to_string(index=False))
print("\nby subset:")
print(pd.DataFrame(metrics["by_subset"]).round(2).to_string(index=False))
metrics["health_mapping"] = HEALTH_MAP
TIMINGS["evaluation_s"] = time.time() - T_START

# %% [markdown]
# ## 13. Ablation E0-E7 (pipeline.md section 10.2) - same engines, folds, hyperparameters; only the features differ

# %%
stage("Ablation")
t0 = time.time()
ablation = pd.DataFrame()
if RUN_ABLATION:
    ABL_PARAMS = RUL_RUNS["full"]["params"]                    # E0-E3, E6, E7 use the full set's tuned hyperparameters
    fgroup = np.array([nc.feature_group(f) for f in FEATURES])
    fwin = np.array([nc.is_window_feature(f) for f in FEATURES])

    def cols_of(blocks_, temporal):
        keep = np.isin(fgroup, list(blocks_) + ["cond"]) & (temporal | ~fwin)
        return [f for f, k in zip(FEATURES, keep) if k]

    EXPERIMENTS = {   # id: (purpose, feature blocks, temporal features?, model)
        "E0": ("raw-sensor Random Forest (baseline)", ("raw",), False, "rf"),
        "E1": ("raw-sensor four-model stack (ensemble benefit)", ("raw",), False, "stack"),
        "E2": ("healthy residual features + stack (condition compensation)", ("raw", "res"), False, "stack"),
        "E3": ("physics residual / EKF state features + RF", ("raw", "res", "phys", "ekf", "innov"), False, "rf"),
        "E4": ("physics + EKF + four-model stack (full pipeline)", ("raw", "res", "phys", "ekf", "innov"), True, "stack"),
        "E5": ("full pipeline without EKF features", ("raw", "res", "phys"), True, "stack"),
        "E6": ("full pipeline without physics residuals", ("raw", "ekf"), True, "stack"),
        "E7": ("full pipeline without temporal features", ("raw", "res", "phys", "ekf", "innov"), False, "stack"),
    }
    abl_rows = []
    for eid, (purpose, blocks_, temporal, kind) in EXPERIMENTS.items():
        cols = cols_of(blocks_, temporal)
        if eid in ("E4", "E5"):                                 # the two feature sets fitted in section 10: reuse them
            run_ = RUL_RUNS["full" if eid == "E4" else "no_ekf"]
            assert sorted(run_["cols"]) == sorted(cols), f"{eid} feature list differs from section 10"
            p_va, p_te = run_["pred_va"], run_["pred_te"]
        elif kind == "rf":
            m = st.make_model("randomforest", ABL_PARAMS["randomforest"]).fit(Xtr[cols], ytr)
            p_va, p_te = np.clip(m.predict(Xva[cols]), 0, None), np.clip(m.predict(Xte[cols]), 0, None)
        else:
            o = st.oof_predictions(NAMES, ABL_PARAMS, Xtr[cols], ytr, folds)
            ms = st.fit_base_models(NAMES, ABL_PARAMS, Xtr[cols], ytr)
            bv, bt = st.predict_bases(ms, NAMES, Xva[cols]), st.predict_bases(ms, NAMES, Xte[cols])
            sk = st.fit_stack(o, ytr, NAMES, bv, yva)
            p_va, p_te = np.clip(sk.predict(bv), 0, None), np.clip(sk.predict(bt), 0, None)
        r = st.report(p_te, yte)
        abl_rows.append({"id": eid, "experiment": purpose, "features": len(cols), "test_rmse": r["rmse"],
                         "test_mae": r["mae"], "test_r2": r["r2"], "test_nasa_mean": r["nasa_score_mean"],
                         "test_rmse_end_of_life": st.rmse(p_te[yte <= 20], yte[yte <= 20]),
                         "validation_rmse": st.rmse(p_va, yva)})
        print(f"{eid} {purpose:58s} {len(cols):4d} features  test RMSE {r['rmse']:.2f}   "
              f"[{(time.time() - t0) / 60:.1f} min]")
    ablation = pd.DataFrame(abl_rows)
    metrics["ablation"] = ablation.to_dict(orient="records")
    print("\nAblation on TEST engines (RUL, cycles):")
    print(ablation.round(3).to_string(index=False))
    print("Claim only what this table shows: the EKF helps if E5 is worse than E4; physics residuals help if E6 is "
          "worse than E4; temporal features help if E7 is worse than E4.")
    print(f"(the shipped RUL model uses the '{CHOICE}' feature set: {CHOICE_REASON})")
TIMINGS["ablation_s"] = time.time() - t0

# %% [markdown]
# ## 14. Component-health forecasts (EXTENSION beyond `pipeline.md`): health h flights ahead
# `pipeline.md` defines RUL as the only ML target; this section adds direct multi-horizon forecasts of every component's
# health. For each component and horizon h one LightGBM model predicts the CHANGE in health over the next h flights from
# today's causal features (+ today's EKF health estimates); forecast = EKF health now + predicted change, clipped to
# [0, 1]. Future labels (the true health h flights later, from the hidden parameters T) are used as targets only.
# Compared on the same engines against two baselines - "no change" (persistence) and a straight-line trend of the EKF
# health - and a forecast is marked `useful` only if it beats BOTH on the validation engines' degraded flights.
# Flights without h further flights (the last h of each engine) have no label and are excluded from training and scoring.

# %%
stage("Forecasting")
t0 = time.time()
FORECAST = {}
if RUN_FORECAST:
    kf_cols = [f"kf_hi_{c}" for c in nc.COMPONENTS]
    FC_FEATURES = FEATURES + kf_cols                              # same column order the backend rebuilds
    XF = X.copy()
    for col in kf_cols:
        XF[col] = cyc[col].to_numpy()
    cyc_cycle = cyc["cycle"].to_numpy()
    fc_models, fc_pred_te, fc_intervals, fc_useful, fc_metrics, fc_rows = {}, {}, {}, {}, {}, []
    for c in nc.COMPONENTS:
        cur, true_now = cyc[f"kf_hi_{c}"].to_numpy(), cyc[f"hi_{c}"].to_numpy()
        fc_intervals[c], fc_useful[c], fc_metrics[c] = {}, {}, {}
        for h in FORECAST_HORIZONS:
            fut = st.future_values(groups, cyc_cycle, true_now, h)
            trend = st.trend_projection(groups, cyc_cycle, cur, h, ROLL)
            has = ~np.isnan(fut)
            tr_r, va_r, te_r = (i[has[i]] for i in (tr_idx, va_idx, te_idx))
            if len(tr_r) < 50:
                print(f"  {c} h={h}: only {len(tr_r)} training flights have a label {h} flights ahead - skipped")
                continue
            model = st.make_model("lightgbm", PARAMS["lightgbm"]).fit(XF.iloc[tr_r], (fut - cur)[tr_r])

            def predict(rows):
                return np.clip(cur[rows] + model.predict(XF.iloc[rows]), 0.0, 1.0)

            met = {part: st.forecast_metrics(predict(r), fut[r], cur[r], trend[r], true_now[r], FORECAST_DEGRADED_BELOW)
                   for part, r in (("validation", va_r), ("test", te_r))}
            fc_metrics[c][str(h)] = met
            if len(va_r) >= 20:                                   # empirical interval from validation residuals
                lo_, hi_ = st.empirical_interval(fut[va_r] - predict(va_r), INTERVAL_LEVEL)
                fc_intervals[c][str(h)] = {"low": lo_, "high": hi_}
            vd = met["validation"]["degraded"]
            fc_useful[c][str(h)] = bool(vd["n"] >= RELIABLE_MIN_CYCLES and (vd["skill_vs_persistence"] or -1) > 0
                                        and (vd["skill_vs_trend"] or -1) > 0)
            fc_models[(c, h)] = model
            fc_pred_te[(c, h)] = predict(te_idx)                  # every test flight (also those without a future label)
            td = met["test"]["degraded"]
            fc_rows.append({"component": c, "h": h, "useful": fc_useful[c][str(h)], "test_degraded_n": td["n"],
                            "forecast_pts": None if td["forecast_rmse"] is None else 100 * td["forecast_rmse"],
                            "persistence_pts": None if td["persistence_rmse"] is None else 100 * td["persistence_rmse"],
                            "trend_pts": None if td["trend_rmse"] is None else 100 * td["trend_rmse"],
                            "skill_vs_persistence": td["skill_vs_persistence"], "skill_vs_trend": td["skill_vs_trend"],
                            "ekf_reliability": RELIABILITY[c]["status"]})
    FORECAST = {"models": fc_models, "pred_te": fc_pred_te, "intervals": fc_intervals, "useful": fc_useful,
                "metrics": fc_metrics, "features": FC_FEATURES}
    metrics["forecast"] = fc_metrics
    print("Forecast RMSE in health points (out of 100) on TEST engines, flights whose current true health < "
          f"{FORECAST_DEGRADED_BELOW}:")
    print(pd.DataFrame(fc_rows).round(3).to_string(index=False))
    print("`useful` was decided on VALIDATION engines (beats both baselines); skill > 0 means better than the baseline. "
          "Forecasts of components with ekf_reliability 'low' inherit the weakness of today's estimate.")
TIMINGS["forecast_s"] = time.time() - t0

# %% [markdown]
# ## 14b. Component RUL (EXTENSION beyond `pipeline.md`): flights until a component reaches its failure level
# N-CMAPSS has one RUL per engine and no per-component remaining life, so a component RUL is DERIVED: from the hidden
# parameters `T` (used as labels only, offline) we compute when each component's TRUE health first falls to
# `COMP_FAIL_HI` (the dashboard's Emergency band edge) and train one LightGBM model per component on "flights until that
# happens", from the same causal features as the engine RUL plus today's EKF health of all components.
# - **Censoring:** engines whose component never reaches the level have no label; they are excluded from training and
#   scoring (their number is printed). A prediction is only meaningful for a component that is degrading, so the
#   backend shows it only while the component's EKF health is below `COMP_RUL_SHOW_BELOW`.
# - **Honest comparison:** each model is scored on the validation and test engines against two baselines - the engine RUL
#   used as the component RUL, and a straight-line extrapolation of the EKF health to the failure level - and is marked
#   `useful` only if it beats BOTH on the validation engines (with enough flights and at least 2 engines).
# - Components that never degrade in the training engines get no model. HPT and LPT may not be separable (see the
#   observability table), so check `ekf_reliability` before presenting a component RUL.

# %%
stage("Component RUL")
t0 = time.time()
COMPONENT_RUL = {}
if RUN_COMPONENT_RUL:
    kf_cols_c = [f"kf_hi_{c}" for c in nc.COMPONENTS]
    CR_FEATURES = FEATURES + kf_cols_c                             # same column order the backend rebuilds
    XCR = X.copy()
    for col in kf_cols_c:
        XCR[col] = cyc[col].to_numpy()
    cyc_cycle_c = cyc["cycle"].to_numpy()
    eng_pred_all = np.full(len(cyc), np.nan)                        # baseline 1: the engine RUL as the component RUL
    eng_pred_all[va_idx], eng_pred_all[te_idx] = pred_va, pred_te
    cr_models, cr_pred_te, cr_intervals, cr_useful, cr_metrics, cr_rows = {}, {}, {}, {}, {}, []
    for c in nc.COMPONENTS:
        lab = np.minimum(st.component_rul_labels(groups, cyc_cycle_c, cyc[f"hi_{c}"].to_numpy(), COMP_FAIL_HI),
                         COMP_RUL_CAP)
        trend = st.trend_time_to_level(groups, cyc_cycle_c, cyc[f"kf_hi_{c}"].to_numpy(), COMP_FAIL_HI, ROLL,
                                       COMP_RUL_CAP)                 # baseline 2: EKF-health trend to the failure level
        has = ~np.isnan(lab)
        tr_r, va_r, te_r = (i[has[i]] for i in (tr_idx, va_idx, te_idx))
        n_eng = {part: int(len(np.unique(groups[r]))) for part, r in (("train", tr_r), ("validation", va_r), ("test", te_r))}
        cr_metrics[c] = {"level": COMP_FAIL_HI, "engines_reaching_level": n_eng}
        cr_useful[c] = False
        if n_eng["train"] < 3 or len(tr_r) < 50:
            print(f"  {c}: only {n_eng['train']} training engine(s) reach health {COMP_FAIL_HI} - no component RUL model")
            continue
        model = st.make_model("lightgbm", PARAMS["lightgbm"]).fit(XCR.iloc[tr_r], lab[tr_r])

        def predict(rows):
            return np.clip(model.predict(XCR.iloc[rows]), 0.0, COMP_RUL_CAP) if len(rows) else np.zeros(0)

        met = {part: st.component_rul_metrics(predict(r), lab[r], eng_pred_all[r], trend[r], groups[r])
               for part, r in (("validation", va_r), ("test", te_r))}
        cr_metrics[c].update(met)
        if len(va_r) >= 20:                                           # empirical interval from validation residuals
            lo_, hi_ = st.empirical_interval(lab[va_r] - predict(va_r), INTERVAL_LEVEL)
            cr_intervals[c] = {"low": lo_, "high": hi_}
        v = met["validation"]
        cr_useful[c] = bool(v["n"] >= RELIABLE_MIN_CYCLES and v["n_engines"] >= 2 and
                            v["model_rmse"] < v["engine_rul_rmse"] and v["model_rmse"] < v["trend_rmse"])
        cr_models[c] = model
        cr_pred_te[c] = predict(te_idx)                               # every test flight (also censored engines)
        t_ = met["test"]
        cr_rows.append({"component": c, "useful": cr_useful[c], "engines val/test": f"{n_eng['validation']}/{n_eng['test']}",
                        "test_flights": t_["n"], "model_rmse": t_["model_rmse"], "engine_rul_rmse": t_["engine_rul_rmse"],
                        "trend_rmse": t_["trend_rmse"], "skill_vs_engine": t_["skill_vs_engine"],
                        "skill_vs_trend": t_["skill_vs_trend"], "ekf_reliability": RELIABILITY[c]["status"]})
    COMPONENT_RUL = {"models": cr_models, "pred_te": cr_pred_te, "intervals": cr_intervals, "useful": cr_useful,
                     "metrics": cr_metrics, "features": CR_FEATURES}
    metrics["component_rul"] = cr_metrics
    print(f"Component RUL: flights until the component's TRUE health reaches {COMP_FAIL_HI} (RMSE in flights, TEST engines "
          "that reach it; censored engines excluded):")
    print(pd.DataFrame(cr_rows).round(3).to_string(index=False) if cr_rows else "(no component qualified)")
    print("`useful` was decided on VALIDATION engines (beats both baselines). skill > 0 = better than that baseline. "
          "A component RUL inherits the weakness of today's EKF health estimate (see ekf_reliability).")
TIMINGS["component_rul_s"] = time.time() - t0

# %% [markdown]
# ## 15. Explainability (pipeline.md section 11): SHAP, Ridge contributions, physics/EKF diagnostics

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

stage("Explainability")
t0 = time.time()
w_eff = stack.effective_weights()
total = np.zeros(len(RUL_FEATURES))
for i, name in enumerate(NAMES):
    n_rows = min(SHAP_ROWS if name != "randomforest" else min(SHAP_ROWS, 100), len(Xtr))
    sample = Xtr[RUL_FEATURES].sample(n_rows, random_state=SEED)
    sv = np.asarray(shap.TreeExplainer(models[name]).shap_values(sample)).reshape(n_rows, -1)[:, :len(RUL_FEATURES)]
    total += w_eff[i] * np.abs(sv).mean(axis=0)               # weighted by d RUL / d model prediction
shap_df = pd.DataFrame({"feature": RUL_FEATURES, "group": [nc.feature_group(f) for f in RUL_FEATURES],
                        "source": [nc.source_of(f) for f in RUL_FEATURES], "mean_abs_shap": np.abs(total)})
shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
shap_df.to_csv("shap_importance.csv", index=False)
by_block = shap_df.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False)
contrib_te = pd.DataFrame(stack.contributions(base_te), columns=NAMES).abs().mean().to_dict()
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
top = shap_df.head(15)[::-1]
axes[0].barh(top["feature"], top["mean_abs_shap"], color="#0284c7")
axes[0].set_title("RUL: top features (|SHAP| x Ridge weight)")
axes[1].bar(by_block.index, by_block.values, color="#0f766e")
axes[1].set_title("RUL: importance by feature block")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=130)
print("importance by block:", by_block.round(3).to_dict())
print("mean |Ridge contribution| per model on test (cycles):", {k: round(v, 2) for k, v in contrib_te.items()})
print("top features:", shap_df.head(10)[["feature", "mean_abs_shap"]].round(3).to_dict(orient="records"))
metrics["mean_abs_ridge_contribution_test"] = contrib_te
TIMINGS["shap_s"] = time.time() - t0

# %% [markdown]
# ## 16. Cost: timings and inference latency

# %%
one_unit = test_units[0]
cols_raw = [c for c in cyc.columns if not c.startswith(("ref_", "res_", "ekf_", "kf_", "innov_", "nis_", "gated_", "hi_"))]
t0 = time.time()
for _ in range(3):
    one = nc.run_layers(cyc.loc[cyc.unit == one_unit, cols_raw], ref, ekf, D_REF)
    stack.predict(st.predict_bases(models, NAMES, feat.transform(one)[RUL_FEATURES]))
metrics["inference_ms_one_engine_all_flights"] = (time.time() - t0) / 3 * 1000
TIMINGS["total_s"] = time.time() - T_START
metrics["timings_s"] = {k: round(v, 1) for k, v in TIMINGS.items()}
print(json.dumps(metrics["timings_s"], indent=1),
      f"\none engine, all flights, full pipeline: {metrics['inference_ms_one_engine_all_flights']:.0f} ms")

# %% [markdown]
# ## 17. Export the versioned bundle (pipeline.md 13.1), save a copy to Drive, download

# %%
import joblib

stage("Export")


def jclean(o):
    """JSON-safe copy: numpy scalars -> python, NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): jclean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jclean(v) for v in o]
    if isinstance(o, np.ndarray):
        return jclean(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def dump(obj, name):
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(jclean(obj), fh, indent=1)


OUT = f"bundle_{RUN}"
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)
EXT = {"randomforest": "joblib", "xgboost": "json", "lightgbm": "txt", "catboost": "cbm"}
base_entries = []
for name in NAMES:
    fn = f"{name}_rul.{EXT[name]}"
    m = models[name]
    if name == "randomforest":
        joblib.dump(m, os.path.join(OUT, fn), compress=3)
    elif name == "lightgbm":
        m.booster_.save_model(os.path.join(OUT, fn))
    else:
        m.save_model(os.path.join(OUT, fn))
    base_entries.append({"name": name, "file": fn, "params": {**st.SPEC_PARAMS[name], **PARAMS[name]}})
dump(ref.to_dict(), "reference.json")
dump(ekf.to_dict(), "ekf.json")
dump(feat.to_dict(), "featurizer.json")
dump(stack.to_dict(), "stack.json")
COMP_RUL_MANIFEST = None
if COMPONENT_RUL.get("models"):
    cr_files = {}
    for c_, m_ in COMPONENT_RUL["models"].items():
        cr_files[c_] = f"component_rul_{c_}.txt"
        m_.booster_.save_model(os.path.join(OUT, cr_files[c_]))
    COMP_RUL_MANIFEST = {"model": "lightgbm", "level": COMP_FAIL_HI, "cap": COMP_RUL_CAP,
                         "show_below": COMP_RUL_SHOW_BELOW, "features": COMPONENT_RUL["features"], "files": cr_files,
                         "intervals": COMPONENT_RUL["intervals"], "useful": COMPONENT_RUL["useful"],
                         "metrics": COMPONENT_RUL["metrics"],
                         "target": "flights until the component's TRUE health first falls to `level` (derived from the "
                                   "hidden parameters; censored engines excluded)",
                         "note": "EXTENSION beyond pipeline.md; use a component RUL only where useful is true"}
FORECAST_MANIFEST = None
if FORECAST:
    fc_files = {}
    for (c_, h_), m_ in FORECAST["models"].items():
        fn_ = f"forecast_{c_}_h{h_}.txt"
        m_.booster_.save_model(os.path.join(OUT, fn_))
        fc_files.setdefault(c_, {})[str(h_)] = fn_
    FORECAST_MANIFEST = {"model": "lightgbm", "horizons": list(FORECAST_HORIZONS), "features": FORECAST["features"],
                         "target": "true health h flights ahead minus the current EKF health estimate",
                         "files": fc_files, "intervals": FORECAST["intervals"], "useful": FORECAST["useful"],
                         "degraded_below": FORECAST_DEGRADED_BELOW, "metrics": FORECAST["metrics"],
                         "note": "EXTENSION beyond pipeline.md; use a forecast only where useful is true"}
dump({"sensors": {s: {"description": nc.SENSOR_INFO[s][0], "unit": nc.UNITS[nc.SENSOR_INFO[s][1]]} for s in nc.XS_VARS},
      "conditions": {w: nc.UNITS[w] for w in nc.W_VARS}, "hidden_parameters_offline_only": nc.T_VARS,
      "inference_cadence": "flight_end", "features": FEATURES, "rul_model_features": RUL_FEATURES,
      "feature_blocks": blocks, "files": SCHEMA}, "schema.json")
shutil.copy("shap_importance.csv", OUT)
shutil.copy("shap_summary.png", OUT)
per_engine.to_csv(os.path.join(OUT, "per_engine_metrics.csv"), index=False)
pd.DataFrame(metrics["by_subset"]).to_csv(os.path.join(OUT, "per_subset_metrics.csv"), index=False)
res_band.to_csv(os.path.join(OUT, "healthy_residuals_by_altitude.csv"), index=False)
if len(ablation):
    ablation.to_csv(os.path.join(OUT, "ablation.csv"), index=False)

# Demo data for the dashboard = the held-out TEST engines (per-flight aggregates; the backend recomputes everything)
demo_cols = ["unit", "unit_local", "dataset", "cycle", "fc", "hs", "rul", "duration_s", "alt_max", "Mach_max"] + \
    [f"{w}_mean" for w in nc.W_VARS] + [c for c in cyc.columns if c.startswith(nc.SUMMARY_PREFIXES + ("p_", "v_", "phi_"))] + \
    nc.T_VARS + [f"hi_{c}" for c in nc.COMPONENTS]
cyc.loc[test_m, list(dict.fromkeys(c for c in demo_cols if c in cyc.columns))].to_csv(
    os.path.join(OUT, "demo_cycles.csv"), index=False)
par = cyc.loc[test_m, ["unit", "cycle"]].copy()
par["pred_rul"] = pred_te
for i, n in enumerate(NAMES):
    par[f"pred_base_{n}"] = base_te[:, i]
for c in nc.COMPONENTS:
    par[f"kf_hi_{c}"] = cyc.loc[test_m, f"kf_hi_{c}"].to_numpy()
for (c, h), v_ in (FORECAST.get("pred_te") or {}).items():
    par[f"pred_fc_{c}_h{h}"] = v_
for c, v_ in (COMPONENT_RUL.get("pred_te") or {}).items():
    par[f"pred_crul_{c}"] = v_
par.to_csv(os.path.join(OUT, "parity_outputs.csv"), index=False)

import catboost
import lightgbm
import sklearn
import xgboost
versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__,
            "catboost": catboost.__version__, "shap": shap.__version__, "joblib": joblib.__version__}
dump(versions, "versions.json")
config = {k: globals()[k] for k in ("EKF_IN_RUL", "EKF_SELECT_MARGIN", "EKF_SELECT_MIN_VAL_ENGINES", "RUN", "SEED", "VAL_FRACTION", "N_FOLDS", "ROLL", "MERGE_COS", "MERGE_UNIQUE", "EKF_GATE", "Q_GRID",
                                    "TUNE", "INTERVAL_LEVEL", "RELIABLE_MAX_RMSE", "RELIABLE_MIN_CYCLES",
                                    "HEALTHY_SAMPLES_PER_FILE")}
manifest = {
    "version": 4, "spec": "pipeline.md", "dataset": "NASA N-CMAPSS", "run": RUN, "files": FILES,
    "components": nc.COMPONENTS, "targets": ["rul"], "d_ref": D_REF, "band_edges": BAND_EDGES, "features": FEATURES,
    "model_features": RUL_FEATURES,
    "reference_file": "reference.json", "ekf_file": "ekf.json", "featurizer_file": "featurizer.json",
    "stack_file": "stack.json", "schema_file": "schema.json", "base_models": base_entries,
    "interval": metrics["interval"], **({"forecast": FORECAST_MANIFEST} if FORECAST_MANIFEST else {}),
    **({"component_rul": COMP_RUL_MANIFEST} if COMP_RUL_MANIFEST else {}), "demo_file": "demo_cycles.csv", "parity_file": "parity_outputs.csv",
    "splits": {"train_engines": train_units, "validation_engines": val_units, "test_engines": test_units},
    "dataset_fingerprints": FINGERPRINTS, "observability": OBS, "ekf_diagnostics": EKF_DIAG,
    "health_mapping": HEALTH_MAP, "reliability": RELIABILITY, "tuning": TUNING, "metrics": metrics, "config": config,
    "versions": versions, "load_log": load_log,
    "build": {**BUILD_INFO, "python": platform.python_version(),
              "trained_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    "notes": ["RUL = Ridge stack (scaler fitted on out-of-fold predictions) of RF, XGBoost, LightGBM, CatBoost.",
              "RUL feature set: 'model_features' (see metrics.ekf_in_rul for how it was chosen and what each option scored).",
              "Component health = EKF parameter estimates mapped as 1 - damage / d_ref; not an ML target. Show only "
              "components whose 'reliability' status is ok as component health; others as low-confidence.",
              "RUL interval is empirical (validation residual quantiles), not a calibrated probability.",
              "Hidden parameters T were used offline only (EKF calibration, health-mapping validation).",
              "Research prototype - not for maintenance decisions."],
}
dump(manifest, "manifest.json")
dump(metrics, "metrics.json")
zip_path = shutil.make_archive(f"genzone_ncmapss_{RUN}_bundle", "zip", OUT)
print("Bundle:", zip_path, f"({os.path.getsize(zip_path) / 1e6:.1f} MB); total time {(time.time() - T_START) / 60:.1f} min")
if CACHE_DIR:
    shutil.copy(zip_path, CACHE_DIR)
    print("copy saved to", CACHE_DIR)
print("\nNEXT: unzip into backend/models/, run `python -m backend.check_parity` (must print PARITY OK), and copy the "
      "tables of sections 8, 11, 12, 13, 14 and 14b into the report.")
if IN_COLAB:
    from google.colab import files
    files.download(zip_path)
