# Training guide (ML team)

This folder trains every model behind the GENZONE turbofan digital twin, from the raw NASA N-CMAPSS files to a
**model bundle** (one zip) that the backend and dashboard load. This page covers what you need to retrain, change, or
evaluate the models.

**TL;DR**
1. Open [`ncmapss_colab_training.ipynb`](ncmapss_colab_training.ipynb) in Google Colab and pick a **T4 GPU** runtime.
2. In the configuration cell, set `RUN = "quick"` (1 file, pipeline check) or `RUN = "full"` (all 10 files, final model).
3. Run all cells. The last cell downloads `genzone_ncmapss_<RUN>_bundle.zip`.
4. Unzip it into `backend/models/`, then run `python -m backend.check_parity`. It must print `PARITY OK`.

---

## 1. Dataset: NASA N-CMAPSS

**What it is:** NASA's *New Commercial Modular Aero-Propulsion System Simulation* dataset (C-MAPSS 2): run-to-failure
simulations of a two-spool turbofan flown on **real recorded flight profiles**, sampled at **1 Hz**. Each engine
starts healthy and degrades until failure. Degradation is set through 10 component health parameters
(efficiency/flow modifiers of fan, LPC, HPC, HPT, LPT).

- **Source we download from:** Kaggle mirror [`bishals098/nasa-cmapss-2-engine-degradation`](https://www.kaggle.com/datasets/bishals098/nasa-cmapss-2-engine-degradation).
  It's public, and kagglehub usually downloads it without credentials.
- **Original source:** NASA Prognostics Center of Excellence data repository ("Turbofan Engine Degradation Simulation-2").
- **Reference paper:** M. Arias Chao, C. Kulkarni, K. Goebel, O. Fink, *"Aircraft Engine Run-to-Failure Dataset under
  Real Flight Conditions for Prognostics and Diagnostics"*, Data 6(1):5, 2021. The PDF
  `Run_to_Failure_Simulation_Under_Real_Flight_Conditions_Dataset.pdf` ships with the Kaggle dataset.
- Check the Kaggle page for licence and citation terms before publishing results.

### Files (10 subsets, ~29 GB total)

| File | Size | Notes |
|---|---|---|
| `N-CMAPSS_DS01-005.h5` | 2.87 GB | |
| `N-CMAPSS_DS02-006.h5` | 2.45 GB | used by `RUN="quick"`; PHM 2021 challenge subset; only HPT/LPT degrade abnormally |
| `N-CMAPSS_DS03-012.h5` | 3.69 GB | |
| `N-CMAPSS_DS04.h5` | 3.75 GB | largest file (sets peak RAM) |
| `N-CMAPSS_DS05.h5` | 2.60 GB | |
| `N-CMAPSS_DS06.h5` | 2.55 GB | |
| `N-CMAPSS_DS07.h5` | 2.71 GB | |
| `N-CMAPSS_DS08a-009.h5` | 3.24 GB | |
| `N-CMAPSS_DS08c-008.h5` | 2.41 GB | |
| `N-CMAPSS_DS08d-010.h5` | 2.89 GB | **the Kaggle copy is truncated by 32 bytes**; repaired automatically (see Troubleshooting) |

Different subsets have different failure modes, so **only the full run has abnormal degradation for every
component**. With DS02 alone, fan, LPC and HPC health stay constant at 1.0.

### Inside each `.h5`

Each file has a `dev` (training) and a `test` split with **different engines**. We keep NASA's split:
`dev` engines are used for training and cross-validation, and `test` engines only for the final evaluation.

| Key | Content | Columns (`*_var` holds the names) |
|---|---|---|
| `W_dev/test` | flight condition | `alt` (ft), `Mach`, `TRA` (throttle, %), `T2` (fan inlet temperature, °R) |
| `X_s_dev/test` | 14 measured sensors | `T24 T30 T48 T50` (°R), `P15 P2 P21 P24 Ps30 P40 P50` (psia), `Nf Nc` (rpm), `Wf` (lbm/s) |
| `X_v_dev/test` | virtual (unmeasured) sensors | we only read `T40 P30 P45`, for **display only**; they are never model inputs |
| `T_dev/test` | true health parameters (labels) | `fan/LPC/HPC/HPT/LPT` × `eff_mod / flow_mod` (10 columns) |
| `A_dev/test` | auxiliary | `unit` (engine id), `cycle` (flight number), `Fc` (flight class), `hs` (1 = healthy phase) |
| `Y_dev/test` | label | RUL in flight cycles |

The loader (`load_h5`) maps columns **by name**, not by position. It also drops physically impossible rows
(sensor ≤ 0, unit or cycle < 1).

---

## 2. The training script

| File | Role |
|---|---|
| [`ncmapss_colab_training.py`](ncmapss_colab_training.py) | **The source of truth.** All training code, as `# %%` cells. Edit this file. |
| [`ncmapss_colab_training.ipynb`](ncmapss_colab_training.ipynb) | Generated notebook for Colab. **Don't edit it by hand**; regenerate it. |
| [`build_notebook.py`](build_notebook.py) | Builds the notebook: `python training/build_notebook.py`. It replaces the `# %% [pipeline]` cell with the full source of `backend/pipeline/ncmapss.py`. |
| [`../backend/pipeline/ncmapss.py`](../backend/pipeline/ncmapss.py) | **Shared pipeline code** (Layers 1–3, feature engineering, CNN inputs, smoothing). The same file runs in Colab and in the backend, so training and serving compute features identically. |

So: change `ncmapss.py` or `ncmapss_colab_training.py`, run `python training/build_notebook.py`, run
`python -m pytest backend/tests`, then commit the `.py` and `.ipynb` files together.

### Running it

**Google Colab (recommended)**
1. Runtime → Change runtime type → **T4 GPU**. High-RAM is optional.
2. Upload or open the notebook and edit the configuration cell (`RUN = "full"` for the final model).
3. Run all. Each `.h5` is downloaded (kagglehub), aggregated, then deleted before the next one, so disk use stays at
   about one file.
4. Kaggle credentials are only needed if the anonymous download fails. In that case add Colab secrets
   `KAGGLE_USERNAME` / `KAGGLE_KEY` (kaggle.com → Settings → API → Create New Token).
5. **Without Kaggle:** put the `.h5` files in Google Drive, mount it, and either leave them under
   `/content/drive/MyDrive` (searched automatically) or set the environment variable `NCMAPSS_DIR` to the folder.

**Kaggle notebooks:** these work too. Attach the dataset; the script finds files under `/kaggle/input` and installs
`onnx`/`onnxruntime` itself (turn Internet on in the notebook settings). Don't pick a **P100** GPU: recent PyTorch has
no kernels for it. The script detects this and falls back to CPU, which is slower. Download the zip from the
notebook's **Output** panel.

**Locally:** this needs the `.h5` files and enough RAM (see section 6). It hasn't been tested end to end outside
Colab. From the repo root:
```bash
pip install -r backend/requirements.txt torch onnx kagglehub matplotlib
RUN=quick NCMAPSS_DIR=/path/to/h5_folder PYTHONPATH=. python training/ncmapss_colab_training.py
```

### Configuration (cell 1)

| Setting | Default | Meaning |
|---|---|---|
| `RUN` | `"quick"` | `"quick"` = DS02 only; `"full"` = all 10 files |
| `DATA_DIR` | env `NCMAPSS_DIR` | optional folder with the `.h5` files |
| `DELETE_AFTER_USE` | `True` | delete each downloaded file after aggregation (saves disk) |
| `HEALTHY_SAMPLES_PER_FILE` | 200 000 | 1 Hz healthy rows per file used to fit the healthy-engine map |
| `BASELINE_CYCLES` / `ROLL` | 10 / 5 | per-engine healthy baseline window / rolling window for trend features (cycles) |
| `N_FOLDS` | 5 | GroupKFold **by engine** on the dev engines |
| `BAND_EDGES` | 0.75 / 0.55 / 0.30 | Normal / Warning / Alarm / Emergency health bands (dashboard) |
| `USE_CNN` | `True` | train the 1D-CNN on 0.1 Hz in-flight windows |
| `CNN_EVERY` / `CNN_WINDOW` | 10 / 50 | keep every 10th 1 Hz row (0.1 Hz); window = 50 samples = 500 s of flight |
| `CNN_EPOCHS` / `CNN_BATCH` / `CNN_LR` / `CNN_SAMPLES_PER_EPOCH` | 20 / 1024 / 1e-3 / 300 000 | CNN training |
| `STACK_MIN_GAIN` | 0.03 | keep a blend only if its CV RMSE is ≥ 3% better than the best single model |
| `SMOOTH_GRID` | 1.0 … 0.1 | causal smoothing strengths tried per target (1 = off) |

---

## 3. What the pipeline does (notebook sections)

| § | Step | Details |
|---|---|---|
| 3 | **Layer 1: ingestion** | per file and split: read 1 Hz data → **one row per (engine, flight)** with sensor means/stds, physics proxies, flight descriptors, labels, and mean polynomial features of the flight condition. Also keeps a **0.1 Hz copy** for the CNN and samples healthy 1 Hz rows. Engine ids become `file_index*1000 + unit`. |
| 4 | **Layer 2: physics** | **healthy-engine map**: each sensor as a degree-3 polynomial of (alt, Mach, TRA, T2), fitted on healthy (`hs==1`) dev data. Residual = measured / ideal − 1, which removes the effect of flight conditions. |
| 4 | **Labels** | damage of component *c* = max(\|eff_mod\|, \|flow_mod\|). **Health index HI_c = 1 − damage_c / D_REF_c**, clipped to [0, 1]. `D_REF_c` = 95th percentile of component *c*'s end-of-life damage over dev engines (**per component**). |
| 5 | **Layer 3: Kalman filter** | linearised gas-path model, `residuals = b(w) + H(w)·θ + noise`, with influence coefficients scheduled on alt/Mach/TRA. H and b are identified on dev engines, where θ is known. At run time it estimates the 10 health parameters θ **from sensors only**, with 1-σ uncertainty and innovation gating that flags bad sensors. |
| 6 | **Features** | per engine, causal: z-score, value relative to the engine's own healthy baseline, rolling mean, rolling slope, flight descriptors, cycle number (151 tabular features). CNN inputs: 42 channels per 0.1 Hz sample (flight condition, raw sensors, residuals, that flight's Kalman states), z-scored on dev engines. |
| 7 | **Tree models** | XGBoost, LightGBM, CatBoost, Random Forest **per target**, with out-of-fold predictions from GroupKFold by engine. Targets: `rul`, `hi_fan`, `hi_lpc`, `hi_hpc`, `hi_hpt`, `hi_lpt`. A target that is constant in training (e.g. fan in DS02) becomes a constant. |
| 8 | **1D-CNN** | Arias Chao et al.-style network: 3 conv layers (kernel 10, 10/10/1 channels) → dense 50 → 6 outputs, on 500 s windows. Per-flight prediction = mean over the windows ending in that flight. Out-of-fold predictions use the same engine folds. |
| 9 | **Model choice + smoothing** | per target: non-negative Ridge blend of all models **or** the best single model, decided by **cross-validated** RMSE on dev engines (blend must win by `STACK_MIN_GAIN`). A tree-only fallback is chosen the same way. Then **causal per-engine smoothing** (RUL tracks −1 per flight; health is exponentially smoothed), with strength tuned on dev engines. |
| 10 | **Evaluation** | on the held-out **test engines** only: RMSE/MAE, NASA score for RUL, per-model and Kalman-only baselines, unsmoothed vs smoothed, **degraded-only errors** (RUL ≤ 20, HI < 0.9), and a naive linear baseline on raw sensors. |
| 11 | **Uncertainty** | LightGBM quantile models give a RUL 90% interval and its coverage on test engines. |
| 12 | **Explainability** | TreeSHAP of the tree models, weighted by the blend weights and grouped by physical source; saved as CSV + `shap_summary.png`. |
| 13 | **Cost** | timings per stage and inference latency. |
| 14 | **Export** | writes the bundle (below), zips it and downloads it. |

**Leakage rules (please keep them):** test engines are never used for fitting, tuning, model choice or smoothing.
Everything that is fitted (healthy-engine map, `D_REF`, Kalman filter, z-score statistics, models, blend weights,
smoothing strength) uses dev engines only. All features are **causal**: a flight's prediction only uses that
flight and earlier ones, so the same code runs in real time.

**Team rule:** never train on synthetic or fake data. Unit tests use stub models with fixed formulas; the real
models are only ever fitted on N-CMAPSS.

---

## 4. Output: the model bundle

| File | Content |
|---|---|
| `manifest.json` | targets, base models, blend weights / chosen model (`model_choice`), `smoothing`, per-component `d_ref`, feature list, metrics, label summary |
| `reference.json` / `kf.json` / `featurizer.json` | healthy-engine map, Kalman filter, feature engineering |
| `<model>_<target>.<ext>` | XGBoost `.json`, LightGBM `.txt`, CatBoost `.cbm`, Random Forest `.joblib` |
| `cnn.onnx` (+ `cnn.pt`) | the CNN; the backend runs it with onnxruntime |
| `lightgbm_rul_q05.txt` / `q95.txt` | RUL interval models |
| `demo_cycles.csv`, `demo_samples.npz` | test engines (per flight and 0.1 Hz) shown in the dashboard |
| `parity_outputs.csv` | the notebook's own predictions for those engines (for `check_parity`) |
| `metrics.json`, `versions.json`, `shap_*` | test metrics and timings, library versions, SHAP importance |

**Installing a bundle:** unzip it into `backend/models/`, then:
```bash
python -m backend.check_parity      # backend must reproduce the notebook's predictions -> "PARITY OK"
python -m backend                   # dashboard at http://localhost:8000
```
**Library versions:** load the bundle with the versions in its `versions.json`. Colab currently uses Python 3.13,
scikit-learn 1.6.1, xgboost 3.4.1, lightgbm 4.6.0 and catboost 1.2.10. Random Forest pickles in particular warn
across scikit-learn versions.

---

## 5. Current results

The bundle currently in `backend/models/` comes from a **full run with DS01–DS07, before the CNN, per-component
health scale and smoothing were added**: 29 test engines, 2,318 test flights.

| RUL (test engines) | RMSE (flights) | MAE |
|---|---|---|
| Blend (trees + GRU, since removed) | 7.49 | 5.49 |
| Random Forest alone | 7.51 | 5.49 |
| Naive linear on raw sensor means | 15.07 | 12.07 |
| Blend, last 20 flights only | **1.98** | 1.42 |
| Blend, RUL > 60 (early life) | 9.16 | 7.46 |

| Health on **degraded flights** (HI < 0.9), in points out of 100 | ML RMSE | Kalman filter alone |
|---|---|---|
| HPT | 2.3 | 4.5 |
| HPC | 3.3 | 4.9 |
| LPT | 5.4 | 9.2 |
| Fan | 7.1 | 6.3 |
| LPC | 24.7 | 17.1 |

The RUL 90% interval covers 83% (too narrow). The run took about 31 minutes on Colab T4 (download + aggregate 9 min,
trees 18 min, SHAP 2.5 min).

**Reference point:** Arias Chao et al. (RESS 2022, "Fusing physics-based and deep learning models for prognostics")
report RUL RMSE 4.95 (CNN, sensors only) and 4.14 (hybrid, perfect calibration) on the **same DS02 test engines
(units 11, 14, 15)** that `RUN="quick"` evaluates. Our quick run scored 8.21 before the CNN. Comparing against those
numbers is the main purpose of the CNN.

---

## 6. Runtime and resources

| | Quick (DS02) | Full (10 files) |
|---|---|---|
| Download + aggregation | a few minutes | ~13–15 min (9 min for 7 files) |
| RAM peak | ~4 GB | ~5 GB per file load + ~3–4 GB for 0.1 Hz CNN data (Colab standard 12.7 GB is enough) |
| Disk | ~3 GB | ~4 GB at a time (each file deleted after use) |
| Trees (6 targets × 4 models × 6 fits) | minutes | ~20 min |
| CNN (6 trainings) | minutes on T4 | estimated 10–20 min on T4, much longer on CPU |

Full-run times with the CNN and DS08 are estimates; the first such run will give the real numbers in `timings_s`.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `OSError: Unable to synchronously open file (truncated file: eof = 2885034848 ... stored_eof = 2885034880)` | Kaggle's `DS08d-010.h5` is 32 bytes short; re-downloading doesn't help. The current script repairs a copy automatically (`repair_truncated_h5`, at most 4 KB missing) and drops any zeroed rows. Update to the latest notebook. |
| `OSError: [Errno 30] Read-only file system: '/kaggle/input/...'` | old notebook on Kaggle tried to delete the input; fixed in the current version. |
| `Could not download ... from Kaggle` | add `KAGGLE_USERNAME` / `KAGGLE_KEY` Colab secrets, or use Drive + `NCMAPSS_DIR`. |
| `CUDA GPU not usable with this PyTorch build` | old GPU (e.g. Kaggle P100); it trains on CPU. Switch to T4. |
| `WARNING: CNN ONNX export failed` | `onnx`/`onnxruntime` missing; the bundle then serves the tree-only blend. Install them and re-run the export cell. |
| `NOTE: <component> health barely varies` | expected with `RUN="quick"`; use `RUN="full"`. |
| Out of memory | use High-RAM, or lower `HEALTHY_SAMPLES_PER_FILE` / raise `CNN_EVERY`. |
| `check_parity` reports MISMATCH | library versions differ from `versions.json`, or the bundle and code are from different commits. |
| Colab disconnects | free Colab drops idle sessions after ~90 min; keep the tab active. |

---

## 8. Open work (good next tasks)

1. **First run with the CNN.** Do `RUN="quick"` first and compare with the paper (4.95 / 4.14), then `RUN="full"`.
2. **Physics ablation:** the same Random Forest on raw sensors → + residuals → + Kalman states → full features, to
   quantify what the physics layers add (the project's main claim).
3. **Kalman health as a candidate model per component.** For fan and LPC, the Kalman filter alone currently beats
   the ML.
4. **LPC accuracy:** fan and LPC are confused (both upstream of T24/P24). Try better proxies and per-flight-phase
   (steady high-power) residuals.
5. **Calibrate the RUL interval** (83% → 90% coverage), e.g. on cross-validated errors.
6. **Diagnosis metric:** how often the weakest predicted component is the truly degrading one.
7. **Sensor-fault robustness test:** perturb one sensor on real test engines and show that the Kalman filter's gating
   keeps health estimates stable.

---

## 9. Tests

```bash
python -m pytest backend/tests        # pipeline maths, Kalman filter, encoder, smoothing, bundle loading, API
```
The tests use **stub** models (fixed formulas, no training), so they check plumbing, not accuracy. Accuracy is
measured only by the notebook's test-engine evaluation.
