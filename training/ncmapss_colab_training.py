# %% [markdown]
# # GENZONE | AI Manthan - N-CMAPSS physics-informed, interpretable health & RUL training
# Layers (same as the architecture diagram):
# 1. **Ingestion & preprocessing** - 1 Hz flight data -> one row per flight cycle; healthy baseline per engine
# 2. **Physics model** - first-principles ratios/efficiency proxies + healthy-engine map ("ideal" sensor values)
# 3. **Kalman filter** - estimates the 10 component health parameters (fan/LPC/HPC/HPT/LPT efficiency & flow)
# 4. **Interpretable ML** - XGBoost + LightGBM + CatBoost + Random Forest on per-cycle physics features, plus a 1D-CNN
#    on 0.1 Hz in-flight windows enriched with the Layer 2/3 outputs (hybrid design of Arias Chao et al., RESS 2022);
#    per target the non-negative Ridge blend is kept only if it beats the best single model in cross-validation;
#    causal per-engine smoothing; targets: RUL and component health indices (fan, LPC, HPC, HPT, LPT); TreeSHAP
# 5/6. Exported bundle is served by the backend/dashboard
#
# **How to run on Google Colab**
# 1. Runtime > Change runtime type > **T4 GPU** (CPU works too, the CNN is just slower). High-RAM helps for RUN="full".
# 2. Run all cells. Data is downloaded per file from Kaggle (`bishals098/nasa-cmapss-2-engine-degradation`).
#    If Kaggle asks for credentials: add Colab secrets KAGGLE_USERNAME and KAGGLE_KEY (from kaggle.com > Settings >
#    API > Create New Token), or put the .h5 files in Google Drive and set DATA_DIR.
# 3. The last cell downloads `genzone_ncmapss_<RUN>_bundle.zip`. Unzip it into `backend/models/` and start the backend.
#
# **RUN = "quick"**: DS02 only (PHM 2021 challenge subset) - checks the pipeline end to end.
# **RUN = "full"**: DS01-DS08 (10 files) combined - every component has abnormal degradation in at least one subset, which the
# component-wise health indices need. Use this for the final model.

# %% [markdown]
# ## 0. Setup

# %%
import gc
import glob
import json
import os
import platform
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = os.path.exists("/kaggle/input")
T_START = time.time()
TIMINGS = {}
if IN_COLAB or IN_KAGGLE:          # Kaggle images lack onnx/onnxruntime -> the CNN would silently miss the bundle
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "xgboost", "lightgbm", "catboost", "shap", "h5py",
                    "kagglehub", "onnx", "onnxruntime", "onnxscript"], check=False)
    try:    # make Colab secrets visible to kagglehub
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
RUN = os.environ.get("RUN", "quick")                          # "quick" (DS02) or "full" (DS01-DS08d)
DATASET_HANDLE = "bishals098/nasa-cmapss-2-engine-degradation"
FILES = {"quick": ["N-CMAPSS_DS02-006.h5"],
         "full": ["N-CMAPSS_DS01-005.h5", "N-CMAPSS_DS02-006.h5", "N-CMAPSS_DS03-012.h5", "N-CMAPSS_DS04.h5",
                  "N-CMAPSS_DS05.h5", "N-CMAPSS_DS06.h5", "N-CMAPSS_DS07.h5", "N-CMAPSS_DS08a-009.h5",
                  "N-CMAPSS_DS08c-008.h5", "N-CMAPSS_DS08d-010.h5"]}[RUN]
DATA_DIR = os.environ.get("NCMAPSS_DIR")                      # e.g. "/content/drive/MyDrive/ncmapss" (optional)
DELETE_AFTER_USE = True                                       # delete each downloaded .h5 after aggregation (disk)
HEALTHY_SAMPLES_PER_FILE = 200_000                            # 1 Hz healthy rows per file for the healthy-engine map
BASELINE_CYCLES = 10                                          # healthy baseline window per engine (cycles)
ROLL = 5                                                      # rolling window for mean/slope features (cycles)
N_FOLDS = 5                                                   # GroupKFold by engine (capped at #engines)
SEED = 42
RUL_SCALE = 100.0
BAND_EDGES = [0.75, 0.55, 0.30]                               # Normal / Warning / Alarm / Emergency health bands
USE_CNN = True                                                # 1D-CNN on 0.1 Hz in-flight windows
CNN_EVERY = 10                                                # keep every 10th 1 Hz row -> 0.1 Hz
CNN_WINDOW = 50                                               # samples per window = 500 s of flight
CNN_EPOCHS = 20
CNN_BATCH = 1024
CNN_LR = 1e-3
CNN_SAMPLES_PER_EPOCH = 300_000                               # random windows drawn per epoch
STACK_MIN_GAIN = 0.03                                         # keep a blend only if >=3% better than best single (CV)
SMOOTH_GRID = [1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1]          # causal smoothing strengths tried (1 = off)
rng = np.random.default_rng(SEED)

# %% [markdown]
# ## 2. Shared pipeline code (identical to backend/pipeline/ncmapss.py - do not edit here)

# %% [pipeline]
from backend.pipeline import ncmapss as nc  # noqa: E402

# %% [markdown]
# ## 3. Download + Layer 1: aggregate 1 Hz flight data into one row per flight cycle (file by file)

# %%
if "nc" not in globals():          # in the notebook the pipeline cell defines everything at top level
    import types
    nc = types.SimpleNamespace(**{k: v for k, v in globals().items() if not k.startswith("_")})


def get_file(fname):
    for root in [r for r in (DATA_DIR, "/kaggle/input", "/content", "/content/drive/MyDrive", os.getcwd()) if r]:
        hits = glob.glob(os.path.join(root, "**", fname), recursive=True)
        if hits:
            return hits[0], False
    import kagglehub
    try:
        return kagglehub.dataset_download(DATASET_HANDLE, path=fname), True
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {fname} from Kaggle ({exc!r}).\nAdd Colab secrets KAGGLE_USERNAME / KAGGLE_KEY "
            "(kaggle.com > Settings > API > Create New Token), run kagglehub.login(), or set DATA_DIR.") from exc


cycles, healthy_w, healthy_x, load_log, seq_parts = [], [], [], [], []
for fi, fname in enumerate(FILES):
    t0 = time.time()
    path, downloaded = get_file(fname)
    t_dl = time.time() - t0
    h5_path, repaired = nc.repair_truncated_h5(path, os.getcwd())   # Kaggle's DS08d-010 is 32 bytes short
    if repaired:
        print(f"NOTE: {fname} is shorter than its HDF5 header records (mirror defect); opened a zero-padded copy")
    ds = fname.replace("N-CMAPSS_", "").replace(".h5", "")
    for split in ("dev", "test"):
        t1 = time.time()
        df = nc.load_h5(h5_path, split)
        if df.attrs.get("dropped_rows"):
            print(f"NOTE: dropped {df.attrs['dropped_rows']} physically impossible rows from {fname} [{split}]")
        agg = nc.aggregate_cycles(df)
        agg["split"], agg["dataset"], agg["unit_local"] = split, ds, agg["unit"]
        agg["unit"] = (fi + 1) * 1000 + agg["unit"]
        cycles.append(agg)
        if USE_CNN:
            sp = nc.downsample_1hz(df, CNN_EVERY)
            sp["unit"] = (fi + 1) * 1000 + sp["unit"]
            seq_parts.append(sp)
        if split == "dev":
            h = df[df["hs"] == 1] if "hs" in df else df
            take = rng.choice(len(h), size=min(HEALTHY_SAMPLES_PER_FILE, len(h)), replace=False)
            healthy_w.append(h[nc.W_VARS].to_numpy()[take])
            healthy_x.append(h[nc.XS_VARS].to_numpy()[take])
            del h                                   # copy of up to millions of 1 Hz rows; free before the next file
        load_log.append({"file": fname, "split": split, "rows_1hz": len(df), "engines": agg.unit.nunique(),
                         "cycles": len(agg), "seconds": round(time.time() - t1, 1)})
        print(load_log[-1])
        del df, agg
        gc.collect()
    if repaired:
        os.remove(repaired)
    if downloaded and DELETE_AFTER_USE:
        try:
            os.remove(path)
        except OSError:                             # read-only mount (e.g. /kaggle/input on Kaggle): nothing to free
            pass
    print(f"{fname}: download {t_dl:.0f}s, total {time.time() - t0:.0f}s")
cyc = pd.concat(cycles, ignore_index=True).sort_values(["unit", "cycle"]).reset_index(drop=True)
del cycles
samples = None
if USE_CNN:     # multi-column sort is a stable lexsort -> time order within each flight is kept
    samples = pd.concat(seq_parts, ignore_index=True).sort_values(["unit", "cycle"]).reset_index(drop=True)
del seq_parts
if USE_CNN:
    print(f"{len(samples):,} in-flight samples at {1 / CNN_EVERY:.1f} Hz for the CNN "
          f"({samples.memory_usage().sum() / 1e9:.2f} GB)")
assert not (set(cyc[cyc.split == "dev"].unit) & set(cyc[cyc.split == "test"].unit)), "dev/test engine overlap"
TIMINGS["download_and_aggregate_s"] = time.time() - T_START
print(f"{cyc.unit.nunique()} engines, {len(cyc)} flight cycles "
      f"(train engines {cyc[cyc.split == 'dev'].unit.nunique()}, test engines {cyc[cyc.split == 'test'].unit.nunique()})")

# %% [markdown]
# ## 4. Layer 2: healthy-engine map ("ideal" sensor values) and residuals

# %%
t0 = time.time()
ref = nc.HealthyReference().fit(np.vstack(healthy_w), np.vstack(healthy_x))
del healthy_w, healthy_x
print("healthy-map relative RMSE per sensor (1 Hz, healthy data):")
print(pd.Series(ref.fit_rmse, index=nc.XS_VARS).map(lambda v: f"{100 * v:.2f}%").to_string())
cyc = nc.add_residuals(cyc, ref)
train_m, test_m = (cyc.split == "dev").to_numpy(), (cyc.split == "test").to_numpy()

# Labels: component health indices from the true health parameters (efficiency/flow modifiers)
D_REF = nc.fit_damage_reference(cyc[train_m])
cyc = pd.concat([cyc, nc.component_health(cyc, D_REF)], axis=1)
label_summary = {}
for c in nc.COMPONENTS:
    hi = cyc.loc[train_m, f"hi_{c}"]
    label_summary[c] = {"min": float(hi.min()), "mean": float(hi.mean()), "std": float(hi.std())}
print("failure-level damage per component D_REF =", {c: round(v, 5) for c, v in D_REF.items()})
print(pd.DataFrame(label_summary).T.round(3))
for c, s in label_summary.items():
    if s["std"] < 0.02:
        print(f"NOTE: {c} health barely varies in the training data - its model can only learn a near-constant value. "
              "Use RUN='full' so every component has abnormal degradation somewhere.")
TIMINGS["layer2_s"] = time.time() - t0

# %% [markdown]
# ## 5. Layer 3: Kalman filter on the linearised gas-path model (component health parameters + uncertainty)

# %%
t0 = time.time()
kf = nc.GasPathKF().fit(cyc[train_m])
kf_out = kf.run(cyc)
cyc = pd.concat([cyc, kf_out, nc.kf_component_health(kf_out, D_REF)], axis=1)
kf_quality = {}
for c in nc.COMPONENTS:
    t = cyc.loc[test_m, f"hi_{c}"]
    e = cyc.loc[test_m, f"kf_hi_{c}"]
    r = t.corr(e)                                   # NaN when the component never degrades (constant truth)
    kf_quality[c] = {"mae": float((t - e).abs().mean()), "corr": float(r) if np.isfinite(r) else None}
print("Kalman-filter (physics only) component health vs truth on TEST engines:")
print(pd.DataFrame(kf_quality).T.round(3))
TIMINGS["layer3_s"] = time.time() - t0

# %% [markdown]
# ## 6. Layer 4 inputs: causal per-engine features

# %%
feat = nc.CausalFeaturizer(baseline_cycles=BASELINE_CYCLES, roll=ROLL).fit(cyc[train_m])
frame = feat.transform(cyc)                              # same row order as cyc (sorted by unit, cycle)
assert (frame["unit"].to_numpy() == cyc["unit"].to_numpy()).all()
FEATURES = feat.feature_names
TARGETS = ["rul"] + [f"hi_{c}" for c in nc.COMPONENTS]
SCALE = np.array([RUL_SCALE] + [1.0] * len(nc.COMPONENTS))
X = frame[FEATURES]
Y = cyc[TARGETS].to_numpy(dtype=np.float64)
groups = cyc["unit"].to_numpy()
tr_idx, te_idx = np.flatnonzero(train_m), np.flatnonzero(test_m)
n_folds = min(N_FOLDS, len(np.unique(groups[tr_idx])))
print(f"{len(FEATURES)} tabular features, {len(tr_idx)} train / {len(te_idx)} test cycles, {n_folds}-fold GroupKFold")
if USE_CNN:     # 0.1 Hz windows: flight condition + sensors + residual vs healthy map + this cycle's KF states
    t0 = time.time()
    enc = nc.SampleEncoder()
    s_raw, s_units, s_rows = enc.raw(samples, cyc, ref)
    enc.fit(s_raw, train_m[s_rows])                      # z-score statistics from training engines only
    sbig, sstarts = nc.build_windows_array(enc.normalize(s_raw), s_units, CNN_WINDOW)
    del s_raw
    gc.collect()
    print(f"CNN inputs: {len(enc.channels)} channels x {CNN_WINDOW}-sample windows, {len(s_rows):,} windows "
          f"({time.time() - t0:.0f}s)")

# %% [markdown]
# ## 7. Tree models per target with out-of-fold predictions (split by engine)

# %%
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold

MODELS = {
    "xgboost": lambda: xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=5, subsample=0.8,
                                        colsample_bytree=0.6, tree_method="hist", random_state=SEED, n_jobs=-1),
    "lightgbm": lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                                          subsample=0.8, subsample_freq=1, colsample_bytree=0.6, random_state=SEED,
                                          verbose=-1, n_jobs=-1),
    "catboost": lambda: CatBoostRegressor(iterations=500, learning_rate=0.05, depth=5, random_seed=SEED, verbose=0,
                                          thread_count=-1),
    "randomforest": lambda: RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_leaf=3, max_features=0.4,
                                                  random_state=SEED, n_jobs=-1),
}
TREES = list(MODELS)
BASES = TREES + (["cnn"] if USE_CNN else [])
folds = list(GroupKFold(n_folds).split(tr_idx, groups=groups[tr_idx]))
oof = np.zeros((len(tr_idx), len(TARGETS), len(BASES)))
final = {t: {} for t in TARGETS}
t0 = time.time()
Xtr = X.iloc[tr_idx]


class ConstPred:
    """Used when a target does not vary (a component that never degrades in these engines/folds)."""

    def __init__(self, v):
        self.v = float(v)

    def predict(self, x):
        return np.full(len(x), self.v)


def fit_model(name, x, y):
    if np.ptp(y) < 1e-9:              # CatBoost refuses constant targets; a constant is the honest prediction
        return ConstPred(y[0])
    return MODELS[name]().fit(x, y)


CONST = {tgt: float(Y[tr_idx, ti][0]) for ti, tgt in enumerate(TARGETS) if np.ptp(Y[tr_idx, ti]) < 1e-9}
if CONST:
    print("Constant in the training data (component never degrades in these files) -> predicted as a constant:",
          CONST, "- use RUN='full' so these components have degradation to learn from.")
for ti, tgt in enumerate(TARGETS):
    y = Y[tr_idx, ti]
    if tgt in CONST:
        oof[:, ti, :] = CONST[tgt]
        continue
    for bi, name in enumerate(TREES):
        for a, b in folds:
            m = fit_model(name, Xtr.iloc[a], y[a])
            oof[b, ti, bi] = m.predict(Xtr.iloc[b])
        final[tgt][name] = fit_model(name, Xtr, y)
    oof_rmse = {n: round(float(np.sqrt(np.mean((oof[:, ti, k] - y) ** 2))), 4) for k, n in enumerate(TREES)}
    print(f"{tgt:7s} OOF RMSE", oof_rmse)
TIMINGS["trees_s"] = time.time() - t0

# %% [markdown]
# ## 8. 1D-CNN on 0.1 Hz in-flight data (multi-output: RUL + 5 component health indices)
# Hybrid design of Arias Chao et al. (RESS 2022): 500 s windows of flight condition + sensors, enriched with the
# residuals vs the healthy-engine map and the Kalman-filter health parameters. Per-cycle prediction = mean over the
# windows ending in that flight.

# %%
cnn_final = cnn_cycles_all = None
TIMINGS["cnn_s"] = 0.0
if USE_CNN:
    import torch
    import torch.nn as nn

    DEVICE = "cpu"
    if torch.cuda.is_available():
        try:    # recent PyTorch builds have no kernels for old GPUs (e.g. Kaggle P100): probe before training
            nn.Conv1d(2, 2, 3).cuda()(torch.zeros(1, 2, 8, device="cuda")).sum().item()
            DEVICE = "cuda"
        except Exception as exc:
            print("CUDA GPU not usable with this PyTorch build, training on CPU:", repr(exc))
    Ys = torch.tensor(Y / SCALE, dtype=torch.float32)

    class CNNNet(nn.Module):
        """3 conv layers (kernel 10, 'same' zero padding, 10/10/1 channels), flatten, dense 50, linear multi-output
        head. Input (batch, window, channels)."""

        def __init__(self, n_in, n_out, window=CNN_WINDOW):
            super().__init__()
            layers, chans = [], [n_in, 10, 10, 1]
            for a, b in zip(chans[:-1], chans[1:]):
                layers += [nn.ConstantPad1d((4, 5), 0.0), nn.Conv1d(a, b, 10), nn.ReLU()]
            self.conv = nn.Sequential(*layers)
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(window, 50), nn.ReLU(), nn.Linear(50, n_out))

        def forward(self, x):
            return self.head(self.conv(x.transpose(1, 2)))

    def train_cnn(cycle_rows_allowed, seed=SEED):
        torch.manual_seed(seed)
        r = np.random.default_rng(seed)
        idx = np.flatnonzero(np.isin(s_rows, cycle_rows_allowed))
        model = CNNNet(len(enc.channels), len(TARGETS)).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=CNN_LR, amsgrad=True)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CNN_EPOCHS)
        for _ in range(CNN_EPOCHS):
            model.train()
            perm = r.choice(idx, size=min(len(idx), CNN_SAMPLES_PER_EPOCH), replace=False)
            for i in range(0, len(perm), CNN_BATCH):
                bt = perm[i:i + CNN_BATCH]
                xb = torch.from_numpy(nc.gather_windows(sbig, sstarts, CNN_WINDOW, bt)).to(DEVICE)
                loss = nn.functional.mse_loss(model(xb), Ys[torch.from_numpy(s_rows[bt])].to(DEVICE))
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        return model.eval()

    def predict_cnn_cycles(model, cycle_rows_wanted):
        """Per-cycle CNN outputs in target units, shape (len(cyc), n_targets); NaN for cycles not requested."""
        idx = np.flatnonzero(np.isin(s_rows, cycle_rows_wanted))
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 8192):
                xb = torch.from_numpy(nc.gather_windows(sbig, sstarts, CNN_WINDOW, idx[i:i + 8192])).to(DEVICE)
                out.append(model(xb).cpu().numpy() * SCALE)
        return nc.cycle_mean(s_rows[idx], np.concatenate(out), len(cyc))

    t0 = time.time()
    ci = BASES.index("cnn")
    for k, (a, b) in enumerate(folds):
        m = train_cnn(tr_idx[a], seed=SEED + k)
        oof[b, :, ci] = predict_cnn_cycles(m, tr_idx[b])[tr_idx[b]]
    cnn_final = train_cnn(tr_idx)
    cnn_cycles_all = predict_cnn_cycles(cnn_final, np.arange(len(cyc)))
    TIMINGS["cnn_s"] = time.time() - t0
    print("CNN OOF RMSE per target:",
          {t: round(float(np.sqrt(np.mean((oof[:, k, ci] - Y[tr_idx, k]) ** 2))), 4) for k, t in enumerate(TARGETS)},
          f"({TIMINGS['cnn_s']:.0f}s on {DEVICE})")

# %% [markdown]
# ## 9. Per target: non-negative Ridge blend OR the best single model (chosen by cross-validation on training engines), then causal smoothing

# %%
class Blend:
    """Linear blend of base predictions: Ridge weights, or one-hot weights = a single model."""

    def __init__(self, coef, intercept):
        self.coef_, self.intercept_ = np.asarray(coef, dtype=np.float64), float(intercept)

    def predict(self, b):
        return b @ self.coef_ + self.intercept_


def clip_target(tgt, v):
    return nc.clip_target(tgt, v)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def ridge_for(tgt):
    return Ridge(alpha=1.0 if tgt == "rul" else 1e-3, positive=True)


def choose_blend(ti, tgt, cols):
    """Blend of base columns `cols` if its cross-validated RMSE beats the best single model by STACK_MIN_GAIN,
    otherwise that single model. Returns (Blend, choice, CV predictions on training rows, report)."""
    y = Y[tr_idx, ti]
    cv = np.zeros(len(tr_idx))
    for a, b in folds:      # meta-model fitted without the fold it predicts -> honest blend RMSE
        cv[b] = ridge_for(tgt).fit(oof[a][:, ti, cols], y[a]).predict(oof[b][:, ti, cols])
    cv = clip_target(tgt, cv)
    singles = {BASES[c]: rmse(clip_target(tgt, oof[:, ti, c]), y) for c in cols}
    best = min(singles, key=singles.get)
    rep = {"blend_cv_rmse": rmse(cv, y), "single_cv_rmse": singles}
    if rep["blend_cv_rmse"] <= (1 - STACK_MIN_GAIN) * singles[best]:
        m = ridge_for(tgt).fit(oof[:, ti, cols], y)
        return Blend(m.coef_, m.intercept_), "blend", cv, rep
    coef = np.zeros(len(cols))
    coef[cols.index(BASES.index(best))] = 1.0
    return Blend(coef, 0.0), best, clip_target(tgt, oof[:, ti, BASES.index(best)]), rep


meta, meta_trees, model_choice, SMOOTH = {}, {}, {}, {}
tr_units, tr_cycles = groups[tr_idx], cyc["cycle"].to_numpy()[tr_idx]
for ti, tgt in enumerate(TARGETS):
    if tgt in CONST:
        meta[tgt] = meta_trees[tgt] = None
        SMOOTH[tgt] = 1.0
        print(f"{tgt:7s} constant = {CONST[tgt]}")
        continue
    meta_trees[tgt], tchoice, _, _ = choose_blend(ti, tgt, list(range(len(TREES))))
    meta[tgt], choice, cv, rep = choose_blend(ti, tgt, list(range(len(BASES))))
    y = Y[tr_idx, ti]
    sm = {a: rmse(nc.smooth_target(tgt, tr_units, tr_cycles, cv, a), y) for a in SMOOTH_GRID}
    SMOOTH[tgt] = min(sm, key=sm.get)
    model_choice[tgt] = {"full": choice, "trees_only": tchoice, **rep, "smoothing_alpha": SMOOTH[tgt],
                         "smoothed_cv_rmse": sm[SMOOTH[tgt]]}
    best = min(rep["single_cv_rmse"], key=rep["single_cv_rmse"].get)
    print(f"{tgt:7s} -> {choice:12s} (blend CV RMSE {rep['blend_cv_rmse']:.4f}, best single {best} "
          f"{rep['single_cv_rmse'][best]:.4f}); smoothing alpha {SMOOTH[tgt]} -> CV RMSE {sm[SMOOTH[tgt]]:.4f}; "
          f"weights {dict(zip(BASES, np.round(meta[tgt].coef_, 3)))}")


def base_predictions(rows):
    """(len(rows), n_targets, n_bases) predictions of every base model."""
    out = np.zeros((len(rows), len(TARGETS), len(BASES)))
    xr = X.iloc[rows]
    for ti, tgt in enumerate(TARGETS):
        if tgt in CONST:
            out[:, ti, :] = CONST[tgt]
            continue
        for bi, name in enumerate(TREES):
            out[:, ti, bi] = final[tgt][name].predict(xr)
    if USE_CNN:
        out[:, :, BASES.index("cnn")] = cnn_cycles_all[rows]
    return out


def stack(base, rows, trees_only=False, smooth=True):
    """Blend (or chosen single model) per target, clipped and causally smoothed per engine (rows sorted)."""
    res = np.zeros(base.shape[:2])
    units, cycles = groups[rows], cyc["cycle"].to_numpy()[rows]
    for ti, tgt in enumerate(TARGETS):
        if tgt in CONST:
            res[:, ti] = CONST[tgt]
            continue
        m = meta_trees[tgt] if trees_only else meta[tgt]
        v = m.predict(base[:, ti, :len(TREES)] if trees_only else base[:, ti, :])
        res[:, ti] = nc.smooth_target(tgt, units, cycles, v, SMOOTH[tgt] if smooth else 1.0)
    return res


# %% [markdown]
# ## 10. Evaluation on held-out TEST engines (all cycles) + baselines + ablation

# %%
def nasa_score(pred, true):
    d = np.asarray(pred) - np.asarray(true)
    return np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)


def report(pred, true, rul=False):
    d = np.asarray(pred) - np.asarray(true)
    r = {"rmse": float(np.sqrt(np.mean(d ** 2))), "mae": float(np.mean(np.abs(d)))}
    if rul:
        s = nasa_score(pred, true)
        r.update({"nasa_score_sum": float(s.sum()), "nasa_score_mean": float(s.mean())})
    return r


base_te = base_predictions(te_idx)
pred_te, pred_te_trees = stack(base_te, te_idx), stack(base_te, te_idx, trees_only=True)
pred_te_unsmoothed = stack(base_te, te_idx, smooth=False)
metrics = {"test_engines": int(len(np.unique(groups[te_idx]))), "test_cycles": int(len(te_idx))}
rows = []
for ti, tgt in enumerate(TARGETS):
    truth = Y[te_idx, ti]
    is_rul = tgt == "rul"
    metrics[f"{tgt}_stack"] = report(pred_te[:, ti], truth, is_rul)
    metrics[f"{tgt}_trees_only"] = report(pred_te_trees[:, ti], truth, is_rul)
    metrics[f"{tgt}_stack_unsmoothed"] = report(pred_te_unsmoothed[:, ti], truth, is_rul)
    deg = truth <= 20 if is_rul else truth < 0.9          # where it matters: end of life / degraded component
    if deg.sum() >= 5:
        metrics[f"{tgt}_stack_degraded"] = report(pred_te[deg, ti], truth[deg], is_rul)
        if not is_rul:
            metrics[f"{tgt}_kalman_filter_only_degraded"] = report(cyc[f"kf_{tgt}"].to_numpy()[te_idx][deg], truth[deg])
    for bi, name in enumerate(BASES):
        metrics[f"{tgt}_{name}"] = report(clip_target(tgt, base_te[:, ti, bi]), truth, is_rul)
    if not is_rul:
        metrics[f"{tgt}_kalman_filter_only"] = report(cyc[f"kf_{tgt}"].to_numpy()[te_idx], truth)
    rows.append({"target": tgt, "chosen": model_choice.get(tgt, {}).get("full", "constant"),
                 "stack": metrics[f"{tgt}_stack"]["rmse"], "unsmoothed": metrics[f"{tgt}_stack_unsmoothed"]["rmse"],
                 "degraded_only": metrics.get(f"{tgt}_stack_degraded", {}).get("rmse"),
                 "trees_only": metrics[f"{tgt}_trees_only"]["rmse"],
                 "physics_KF_only": metrics.get(f"{tgt}_kalman_filter_only", {}).get("rmse")})
# naive baseline the spec asks to beat: linear regression on raw per-cycle sensor means
raw_cols = [f"m_{s}" for s in nc.XS_VARS] + nc.FLIGHT_COLS
lin = LinearRegression().fit(cyc.loc[train_m, raw_cols], Y[tr_idx, 0])
metrics["rul_naive_linear_raw_sensors"] = report(np.clip(lin.predict(cyc.loc[test_m, raw_cols]), 0, None), Y[te_idx, 0], True)
ablation = pd.DataFrame(rows).set_index("target")
print("Test RMSE (RUL in cycles; health indices on 0-1 scale, x100 = health points; "
      "degraded_only = RUL <= 20 / health < 0.9):")
print(ablation.round(4))
print("RUL naive linear baseline RMSE:", round(metrics["rul_naive_linear_raw_sensors"]["rmse"], 2))
if USE_CNN:
    print("'stack' vs 'trees_only' shows what the CNN adds (it costs interpretability).")

# %% [markdown]
# ## 11. Uncertainty: RUL quantile models (90% interval)

# %%
q_models = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=400, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                                 random_state=SEED, verbose=-1, n_jobs=-1).fit(Xtr, Y[tr_idx, 0])
            for q in (0.05, 0.95)}
lo = np.clip(q_models[0.05].predict(X.iloc[te_idx]), 0, None)
hi = np.clip(q_models[0.95].predict(X.iloc[te_idx]), 0, None)
cover = float(np.mean((Y[te_idx, 0] >= lo) & (Y[te_idx, 0] <= hi)))
metrics["rul_interval90_coverage"] = cover
metrics["rul_interval90_mean_width"] = float(np.mean(hi - lo))
print(f"RUL 90% interval coverage on test engines: {cover:.1%} (target 90%), mean width {np.mean(hi - lo):.1f} cycles")

# %% [markdown]
# ## 12. Explainability: TreeSHAP per target (weighted by the Ridge blend weights), grouped by physical source

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

t0 = time.time()
sample = Xtr.sample(min(1500, len(Xtr)), random_state=SEED)
shap_rows = []
for tgt in TARGETS:
    if tgt in CONST:
        continue
    total = np.zeros(sample.shape)
    for bi, name in enumerate(TREES):
        w = float(meta[tgt].coef_[bi])
        if w == 0:
            continue
        sv = shap.TreeExplainer(final[tgt][name]).shap_values(sample)
        total += w * np.asarray(sv).reshape(sample.shape)
    imp = np.abs(total).mean(0)
    for f_, v in zip(FEATURES, imp):
        shap_rows.append({"target": tgt, "feature": f_, "group": nc.source_of(f_), "mean_abs_shap": float(v)})
shap_df = pd.DataFrame(shap_rows)
shap_df.to_csv("shap_importance.csv", index=False)
by_group = shap_df.groupby(["target", "group"])["mean_abs_shap"].sum().reset_index()
by_group.to_csv("shap_importance_by_group.csv", index=False)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, tgt in zip(axes.ravel(), TARGETS):
    if tgt in CONST:
        ax.set_title(f"{tgt}: constant in training data")
        ax.axis("off")
        continue
    top = by_group[by_group.target == tgt].nlargest(8, "mean_abs_shap")[::-1]
    ax.barh(top["group"], top["mean_abs_shap"], color="#0284c7")
    ax.set_title(f"{tgt}: top drivers (tree part, |SHAP|)")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=130)
TIMINGS["shap_s"] = time.time() - t0
print(by_group.sort_values("mean_abs_shap", ascending=False).groupby("target").head(3).to_string(index=False))

# %% [markdown]
# ## 13. Cost: timings and inference latency

# %%
one = np.flatnonzero(groups == groups[te_idx[0]])                  # all cycles of one test engine
t0 = time.time()
for _ in range(5):
    stack(base_predictions(one), one)
metrics["inference_ms_one_engine_all_cycles"] = (time.time() - t0) / 5 * 1000
TIMINGS["total_s"] = time.time() - T_START
metrics["timings_s"] = {k: round(v, 1) for k, v in TIMINGS.items()}
print(json.dumps(metrics["timings_s"], indent=1), f"\none-engine inference {metrics['inference_ms_one_engine_all_cycles']:.0f} ms")

# %% [markdown]
# ## 14. Export the bundle for the backend (native model formats + JSON) and download

# %%
import copy

import joblib

OUT = f"bundle_{RUN}"
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)
EXT = {"xgboost": "json", "lightgbm": "txt", "catboost": "cbm", "randomforest": "joblib"}
targets_manifest = {}
for ti, tgt in enumerate(TARGETS):
    if tgt in CONST:
        cm = {"coef": [], "intercept": CONST[tgt]}
        targets_manifest[tgt] = {"constant": CONST[tgt], "base_models": [], "meta": cm, "trees_only_meta": cm}
        continue
    bases = []
    for name in TREES:
        fn = f"{name}_{tgt}.{EXT[name]}"
        mdl = final[tgt][name]
        if name == "xgboost":
            mdl.save_model(os.path.join(OUT, fn))
        elif name == "lightgbm":
            mdl.booster_.save_model(os.path.join(OUT, fn))
        elif name == "catboost":
            mdl.save_model(os.path.join(OUT, fn))
        else:
            joblib.dump(mdl, os.path.join(OUT, fn), compress=3)
        bases.append({"type": name, "file": fn})
    if USE_CNN:
        bases.append({"type": "cnn", "output_index": ti})
    targets_manifest[tgt] = {
        "base_models": bases,
        "meta": {"coef": [float(c) for c in meta[tgt].coef_], "intercept": float(meta[tgt].intercept_)},
        "trees_only_meta": {"coef": [float(c) for c in meta_trees[tgt].coef_],
                            "intercept": float(meta_trees[tgt].intercept_)},
    }
for q, m in q_models.items():
    m.booster_.save_model(os.path.join(OUT, f"lightgbm_rul_q{int(q * 100):02d}.txt"))
json.dump(ref.to_dict(), open(os.path.join(OUT, "reference.json"), "w"))
json.dump(kf.to_dict(), open(os.path.join(OUT, "kf.json"), "w"))
json.dump(feat.to_dict(), open(os.path.join(OUT, "featurizer.json"), "w"))
for f_ in ("shap_importance.csv", "shap_importance_by_group.csv", "shap_summary.png"):
    shutil.copy(f_, OUT)

cnn_manifest = None
if USE_CNN:
    torch.save(cnn_final.state_dict(), os.path.join(OUT, "cnn.pt"))
    cnn_manifest = {"file": None, "weights_file": "cnn.pt", "window": CNN_WINDOW, "every": CNN_EVERY,
                    "encoder": enc.to_dict(), "outputs": TARGETS, "output_scale": SCALE.tolist(),
                    "padding": "repeat_first_sample"}
    try:
        class Scaled(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.register_buffer("scale", torch.tensor(SCALE, dtype=torch.float32))

            def forward(self, x):
                return self.m(x) * self.scale

        cpu_model = Scaled(copy.deepcopy(cnn_final).cpu().eval()).eval()
        kw = dict(input_names=["window"], output_names=["targets"],
                  dynamic_axes={"window": {0: "batch"}, "targets": {0: "batch"}}, opset_version=17)
        onnx_path = os.path.join(OUT, "cnn.onnx")
        example = torch.zeros(1, CNN_WINDOW, len(enc.channels))
        try:
            torch.onnx.export(cpu_model, example, onnx_path, dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(cpu_model, example, onnx_path, **kw)
        import onnxruntime as ort
        wins = nc.gather_windows(sbig, sstarts, CNN_WINDOW, np.flatnonzero(np.isin(s_rows, te_idx))[:256])
        with torch.no_grad():
            ref_out = cpu_model(torch.from_numpy(wins)).numpy()
        got = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).run(None, {"window": wins})[0]
        diff = float(np.max(np.abs(ref_out - got)))
        metrics["cnn_onnx_max_abs_diff"] = diff
        if diff < 1e-3:
            cnn_manifest["file"] = "cnn.onnx"
        print(f"CNN ONNX export: max |torch - onnx| = {diff:.2e}")
    except Exception as exc:
        print("WARNING: CNN ONNX export failed; the backend will fall back to the tree-only blend:", repr(exc))
    # raw 0.1 Hz samples of the test engines: the backend recomputes residuals, KF states and CNN inputs itself
    ds = samples[np.isin(samples["unit"].to_numpy(), np.unique(groups[te_idx]))]
    np.savez_compressed(os.path.join(OUT, "demo_samples.npz"), unit=ds["unit"].to_numpy(np.int32),
                        cycle=ds["cycle"].to_numpy(np.int32), values=ds[nc.W_VARS + nc.XS_VARS].to_numpy(np.float32),
                        columns=np.array(nc.W_VARS + nc.XS_VARS))

# Demo data for the dashboard = the held-out TEST engines (per-cycle aggregates; the backend recomputes Layers 2-4)
demo_cols = ["unit", "unit_local", "dataset", "cycle", "fc", "hs", "rul", "duration_s", "alt_max", "Mach_max"] + \
    [f"{v}_mean" for v in nc.W_VARS] + [c for c in cyc.columns if c.startswith(("m_", "sd_", "p_", "v_", "phi_"))] + \
    nc.T_VARS + [f"hi_{c}" for c in nc.COMPONENTS]
cyc.loc[test_m, [c for c in demo_cols if c in cyc.columns]].to_csv(os.path.join(OUT, "demo_cycles.csv"), index=False)
par = cyc.loc[test_m, ["unit", "cycle"]].copy()
for ti, tgt in enumerate(TARGETS):
    par[f"pred_{tgt}"] = pred_te[:, ti]
    par[f"pred_{tgt}_trees_only"] = pred_te_trees[:, ti]
for c in nc.COMPONENTS:
    par[f"kf_hi_{c}"] = cyc.loc[test_m, f"kf_hi_{c}"].to_numpy()
par["rul_q05"], par["rul_q95"] = lo, hi
par.to_csv(os.path.join(OUT, "parity_outputs.csv"), index=False)

manifest = {
    "version": 3, "dataset": "NASA N-CMAPSS", "run": RUN, "files": FILES, "components": nc.COMPONENTS,
    "targets": targets_manifest, "target_order": TARGETS, "rul_scale": RUL_SCALE, "d_ref": D_REF,
    "band_edges": BAND_EDGES, "features": FEATURES, "featurizer_file": "featurizer.json",
    "reference_file": "reference.json", "kf_file": "kf.json", "cnn": cnn_manifest,
    "uncertainty": {"rul_q05_file": "lightgbm_rul_q05.txt", "rul_q95_file": "lightgbm_rul_q95.txt",
                    "coverage_test": cover},
    "demo_file": "demo_cycles.csv", "demo_samples_file": "demo_samples.npz" if USE_CNN else None,
    "parity_file": "parity_outputs.csv", "model_choice": model_choice, "smoothing": SMOOTH,
    "metrics": metrics, "kf_quality_test": kf_quality, "label_summary": label_summary, "load_log": load_log,
    "notes": ["Health index 1 = healthy, 0 = failure-level damage of THAT component (d_ref per component).",
              "Predictions are causally smoothed per engine (alpha per target in 'smoothing'; RUL tracks -1/cycle).",
              "Virtual sensors (v_*) are display-only; models use W + X_s derived features only."],
}
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
import catboost
import sklearn
versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "xgboost": xgb.__version__, "lightgbm": lgb.__version__, "catboost": catboost.__version__,
            "scikit-learn": sklearn.__version__, "shap": shap.__version__, "joblib": joblib.__version__}
if USE_CNN:
    versions["torch"] = torch.__version__
json.dump(versions, open(os.path.join(OUT, "versions.json"), "w"), indent=2)
json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
zip_path = shutil.make_archive(f"genzone_ncmapss_{RUN}_bundle", "zip", OUT)
print("Bundle:", zip_path, f"({os.path.getsize(zip_path) / 1e6:.1f} MB); total time {(time.time() - T_START) / 60:.1f} min")
if IN_COLAB:
    from google.colab import files
    files.download(zip_path)
