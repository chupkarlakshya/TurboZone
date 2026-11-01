# Training guide (ML team)

This folder trains the GENZONE turbofan digital twin **exactly as specified in [`pipeline.md`](../pipeline.md)**:
a physics-informed Extended Kalman Filter plus a stacking ensemble of four tree models, trained on NASA N-CMAPSS. The
output is one versioned **model bundle** (a zip) that the backend and dashboard load.

```text
N-CMAPSS HDF5 -> schema inspection + validation -> flight-end aggregation -> engine split (train / validation / test)
  -> healthy-condition baseline g(W), cross-fitted residuals -> reduced-order gas-path model + EKF
  -> causal features -> Random Forest, XGBoost, LightGBM, CatBoost -> out-of-fold predictions -> scaler + Ridge
  -> RUL + empirical interval -> evaluation, ablation E0-E7, SHAP + Ridge contributions -> bundle
```

> **Extensions.** On top of the spec, the notebook adds two optional sections: **component-health forecasts** (section 14;
> health 10, 25 and 50 flights ahead, `RUN_FORECAST`) and **component RUL** (section 14b; flights until a component
> reaches its failure level, `RUN_COMPONENT_RUL`). See section 6.

> **Status.** The first `RUN = "quick"` (DS02) ran end to end on Colab (45 min) and its bundle passes `check_parity`.
> Results are summarised in section 10. Three fixes were made afterwards and have **not been run on Colab yet**: early
> stopping for LightGBM/CatBoost (speed), EKF state selection (never-moving parameters dropped, jointly collinear states
> merged) and an interval per predicted-RUL range. Re-run `quick` once before the long `full` run.

## Quick start

1. Open [`ncmapss_colab_training.ipynb`](ncmapss_colab_training.ipynb) in Google Colab (a **CPU** runtime is enough).
2. Configuration cell: `RUN = "quick"` for the first run (DS02 only), later `RUN = "full"` (9 files; DS08d excluded).
3. **Run all** and accept the Google Drive prompt (the aggregate cache lives in `MyDrive/genzone/`).
4. When it finishes, `genzone_ncmapss_<RUN>_bundle.zip` downloads (a copy is saved in `MyDrive/genzone/`).
5. Unzip it into `backend/models/`, then:
   ```bash
   python -m backend.check_parity      # must print PARITY OK
   python -m backend                   # dashboard at http://localhost:8000
   ```
   `backend/models/` currently holds the first quick-run bundle (DS02 test engines 1011, 1014, 1015); the old pipeline's
   bundle is kept only in `docs/genzone_ncmapss_full_bundle/` for reference.

## 1. Dataset: NASA N-CMAPSS

Run-to-failure simulations of a two-spool turbofan flown on **real recorded flight profiles**, sampled at **1 Hz**
(C-MAPSS 2). Each engine starts healthy and degrades until failure; degradation is imposed through 10 hidden health
parameters (efficiency and flow modifiers of fan, LPC, HPC, HPT, LPT).

- **Downloaded from:** Kaggle mirror [`bishals098/nasa-cmapss-2-engine-degradation`](https://www.kaggle.com/datasets/bishals098/nasa-cmapss-2-engine-degradation) (public).
- **Original:** NASA Prognostics Center of Excellence data repository.
- **Cite:** M. Arias Chao, C. Kulkarni, K. Goebel, O. Fink, *Aircraft Engine Run-to-Failure Dataset under Real Flight
  Conditions for Prognostics and Diagnostics*, Data 6(1):5, 2021, https://doi.org/10.3390/data6010005. Check the Kaggle
  page for licence terms.

| File | Size | Notes |
|---|---|---|
| `N-CMAPSS_DS01-005.h5` | 2.87 GB | |
| `N-CMAPSS_DS02-006.h5` | 2.45 GB | `RUN="quick"`; PHM 2021 challenge subset; only HPT/LPT degrade abnormally |
| `N-CMAPSS_DS03-012.h5` | 3.69 GB | |
| `N-CMAPSS_DS04.h5` | 3.75 GB | largest file (sets peak RAM) |
| `N-CMAPSS_DS05.h5` … `DS07.h5` | 2.55–2.71 GB | |
| `N-CMAPSS_DS08a-009.h5` | 3.24 GB | |
| `N-CMAPSS_DS08c-008.h5` | 2.41 GB | |
| `N-CMAPSS_DS08d-010.h5` | 2.89 GB | **not used**: Kaggle's copy is truncated inside HDF5's internal index and cannot be read (zero-padding lets it open, then reading fails with "free block size is zero?") |

There is no `DS08b`. Inside each file (`dev` = development engines, `test` = NASA's held-out engines):

| Key | Content | Use in this pipeline |
|---|---|---|
| `W_*` | `alt` (ft), `Mach`, `TRA` (%), `T2` (°R) | flight condition: baseline, EKF scheduling, features |
| `X_s_*` | `T24 T30 T48 T50` (°R), `P15 P2 P21 P24 Ps30 P40 P50` (psia), `Nf Nc` (rpm), `Wf` (lbm/s) | measured sensors |
| `X_v_*` | virtual sensors (`T40 P30 P45` read) | **display only**, never model inputs |
| `T_*` | 10 hidden health parameters | **offline only**: EKF calibration and validating the health mapping |
| `A_*` | `unit`, `cycle` (flight), `Fc` (flight class), `hs` (1 = healthy phase) | grouping, ordering, healthy-row selection |
| `Y_*` | RUL in flight cycles | **target only** |

## 2. Files

| File | Role |
|---|---|
| [`ncmapss_colab_training.py`](ncmapss_colab_training.py) | **Source of truth**: the training code as `# %%` cells. Edit this. |
| [`ncmapss_colab_training.ipynb`](ncmapss_colab_training.ipynb) | Generated notebook for Colab. Never edit by hand. |
| [`build_notebook.py`](build_notebook.py) | `python training/build_notebook.py`: inlines the shared modules into the `[pipeline:…]` cells and stamps the build (git commit + code hash). |
| [`../backend/pipeline/ncmapss.py`](../backend/pipeline/ncmapss.py) | Shared: ingestion, schema validation, aggregation, healthy baseline, EKF, features, leakage audit. |
| [`../backend/pipeline/stacking.py`](../backend/pipeline/stacking.py) | Shared: four base models, tuning, engine splits/folds, OOF stacking, Ridge, metrics, interval. |

**Changing the code:** edit the `.py` file(s) → `python -m pytest backend/tests` → `python training/build_notebook.py` →
commit the `.py` and the `.ipynb` together. If you change `load_h5` / `aggregate_cycles`, bump `AGG_VERSION` in
`ncmapss.py` (this invalidates the Drive cache).

## 3. Running on Colab

| Step | What happens |
|---|---|
| Setup | installs xgboost, lightgbm, catboost, shap, h5py, kagglehub, pyarrow; reads Kaggle secrets if present |
| Configuration | **edit `RUN`**; other settings in section 4 |
| Drive | mounts Drive for the cache; no Drive → set `USE_DRIVE = False` (works, nothing is reused) |
| Pre-flight | lists the files on Kaggle with sizes and checks free disk |
| Layer 1 | per file: cache hit → seconds; else download (kagglehub), inspect, validate, aggregate, cache, delete the `.h5` |
| Engine split | train / validation / test engines fixed **before anything is fitted** |
| Layers 2–3 | healthy baseline, cross-fitted residuals, EKF calibration, process-noise tuning, diagnostics, health-mapping validation |
| Features → models → stack | causal features, tuning on validation, out-of-fold predictions, Ridge |
| Evaluation, ablation | test engines (once), E0–E7 |
| Forecasting (extension) | health forecasts, baselines, intervals |
| SHAP | explanations |
| Export | bundle zip, copy to Drive, download |

**Kaggle download.** One file at a time via `kagglehub.dataset_download(DATASET_HANDLE, path=<file>)`; each is deleted after
aggregation (disk stays at ~one file). Local copies (`NCMAPSS_DIR`, `/kaggle/input`, `MyDrive/genzone/`, `/content`,
`MyDrive/`) are used instead and never deleted. Anonymous download usually works; otherwise add the Colab secrets
`KAGGLE_USERNAME` and `KAGGLE_KEY` (kaggle.com → Settings → API → Create New Token). Without Kaggle, put the `.h5` files in
Drive.

**Resuming after a disconnect:** re-run all cells. Files already aggregated load from the Drive cache in seconds; the
rest of the pipeline is recomputed. Delete `MyDrive/genzone/agg_*` (or set `CACHE_AGGREGATES = False`) for a clean run.

## 4. Configuration

| Setting | Default | Meaning |
|---|---|---|
| `RUN` | `"quick"` | `"quick"` = DS02; `"full"` = 9 files (DS08d excluded) |
| `DATA_DIR` | env `NCMAPSS_DIR` | optional folder with the `.h5` files |
| `USE_DRIVE` / `DRIVE_DIR` / `CACHE_AGGREGATES` | `True` / `MyDrive/genzone` / `True` | aggregate cache + bundle copy |
| `HEALTHY_SAMPLES_PER_FILE` | 200 000 | healthy 1 Hz rows per file for the baseline g(W) |
| `SEED` | 42 | engine split, sampling, models |
| `VAL_FRACTION` | 0.2 | share of development engines held out for validation (stratified by subset) |
| `N_FOLDS` | 5 | engine-grouped folds for out-of-fold predictions and cross-fitting |
| `ROLL` | 5 | trailing window (flights) of the temporal features |
| `MERGE_COS` | 0.95 | EKF: merge two states whose sensor signatures are at least this similar |
| `MERGE_UNIQUE` | 0.15 | EKF: merge a state whose signature is mostly a combination of the others (unique share below this); also the identifiability threshold |
| `EKF_GATE` | 9.0 | EKF: per-sensor NIS gate (a sensor is rejected for at most 3 flights in a row) |
| `Q_GRID` | 0.1 … 10 | EKF: process-noise scales tried on validation engines |
| `TUNE` | `True` | small validation grids for the four models; XGBoost, LightGBM and CatBoost always early-stop on validation |
| `INTERVAL_LEVEL` | 0.90 | empirical RUL interval from validation residual quantiles |
| `RELIABLE_MAX_RMSE` / `RELIABLE_MIN_CYCLES` | 0.10 / 30 | health of a component is "ok" only if identifiable and its validation error on degraded flights ≤ 10 points |
| `EKF_IN_RUL` | `"auto"` | RUL feature set: `"auto"` (validation engines choose), `True` (the spec's full list incl. EKF features), `False` (without EKF features) |
| `EKF_SELECT_MARGIN` / `EKF_SELECT_MIN_VAL_ENGINES` | 0.02 / 5 | `auto` keeps the full list unless "no EKF" wins on validation by more than the margin, and only chooses at all with at least this many validation engines |
| `RUN_ABLATION` | `True` | E0–E7 |
| `RUN_FORECAST` / `FORECAST_HORIZONS` | `True` / `(10, 25, 50)` | EXTENSION: component-health forecasts (section 6) |
| `FORECAST_DEGRADED_BELOW` | 0.9 | forecasts are scored on flights whose current true health is below this |
| `RUN_COMPONENT_RUL` | `True` | EXTENSION: component RUL (section 6) |
| `COMP_FAIL_HI` / `COMP_RUL_CAP` / `COMP_RUL_SHOW_BELOW` | 0.30 / 100 / 0.9 | component failure level (the Emergency band edge); cap in flights; the dashboard shows it only while the EKF health is below this |
| `SHAP_ROWS` | 300 | training rows explained per model (Random Forest ≤ 100) |

## 5. What each step does (and which `pipeline.md` section it implements)

| Step | Implementation | Spec |
|---|---|---|
| **Schema inspection & validation** | `inspect_h5` lists keys/shapes/dtypes/attributes; `load_h5` maps columns **by name**, requires row-aligned groups and drops corrupt rows (non-finite, sensor ≤ 0, unit/cycle < 1); `validate_split` checks engine contiguity and cycle order (hard errors), RUL monotonicity, `hs` values, missing values and ranges (reported). Everything is saved in `schema.json`. | 0.1–0.2, 2.5 |
| **Flight-end aggregation** | one row per completed flight: per sensor **mean, std, min, max, last value**; flight condition; physics proxies; labels. Telemetry rows are never treated as cycles; flights are never mixed. | 0.3, 3.1–3.2 |
| **Engine split** | official test engines untouched; validation = 20 % of development engines (fixed seed, stratified by subset); composite ids `file*1000 + unit`. | 0.4, 2.4 |
| **Healthy baseline g(W)** | degree-3 ridge polynomial of (alt, Mach, TRA, T2) per sensor, fitted on healthy (`hs==1`) rows of **training** engines; residual = measured / healthy − 1. Training-engine residuals are **cross-fitted** (baseline refitted without that engine's fold). Residuals are inspected by altitude band on validation engines. | 3.4, 4.5 |
| **Physics proxies** | pressure ratios, isentropic efficiency proxies, corrected speeds/flow from measured stations only (HPC uses static Ps30 because P30 is not measured; nothing invented). | 4.3–4.4 |
| **Reduced-order observation model** | residuals = b(w) + B(w)·x, b and B linear in [1, alt, Mach, TRA], identified on training engines from the 10 hidden parameters (offline use of `T`, 4.7). | 4.6–4.8 |
| **State selection** | parameters that never move in the training engines are **not estimated** (held at their training value; e.g. 7 of 10 on DS02). The rest: states whose whitened sensor signatures have \|cosine\| ≥ `MERGE_COS` are merged, then any state whose signature is mostly a combination of the others (unique share < `MERGE_UNIQUE`) is merged with its most similar partner. A merged state moves its members along their dominant joint direction. | 4.2, 4.8, 5.4 |
| **EKF** | random-walk state advanced once per flight; Joseph covariance update; symmetrised; per-sensor NIS gating; R = robust variance of healthy-phase training residuals; Q from training state increments × scale tuned on validation engines (one-step predictive likelihood, no labels); x0/P0 from first flights. Outputs per flight: states + σ, parameter estimates + σ, prior innovations, per-sensor and joint NIS, minimum covariance eigenvalue. | 5.1–5.5 |
| **Health mapping** | HI = 1 − max(\|eff\|,\|flow\|)/d_ref per component (d_ref = 95th percentile of end-of-life damage on training engines), computed from the **EKF** estimates. Validated against the truth on validation and test engines; components get `reliability` = ok / low / insufficient (decided on validation, needs identifiability). Not an ML target. | 1.1, 4.2 |
| **Causal features** | blocks: sensor summaries; flight condition; healthy residuals; physics proxies; EKF states; EKF σ; innovations; trailing-window mean/std/min/max/slope of residuals, proxies and states; cycle count, cumulative flight hours, history length. Minimum-history rule: windows use the flights available so far. Missing values → training medians. **Leakage audit** blocks target, hidden parameters, identifiers and `hs`. | 6.1–6.3 |
| **EKF features in the RUL model?** | the whole RUL pipeline (tuning, out-of-fold, Ridge) is fitted for **two feature sets** and both are reported: **full** (the spec's list) and **no_ekf** (without EKF states, uncertainty and innovations). `EKF_IN_RUL` decides which ships: `"auto"` lets the **validation** engines choose (spec 7.5: tune on validation only) but keeps the spec's full list unless no_ekf wins by more than 2 % and there are at least 5 validation engines (one engine cannot choose reliably); `True` / `False` force it. Test engines never take part. Both results are in `metrics.ekf_in_rul` and the ablation (E4 = full, E5 = no_ekf). | 7.5, rule 9 |
| **Base models** | Random Forest, XGBoost, LightGBM, CatBoost with the spec's starting configurations; small grids tuned on validation engines; XGBoost, LightGBM and CatBoost early-stop on validation (the number of trees is then frozen for all refits); LightGBM `subsample_freq=1` so `subsample` takes effect. | 7 |
| **Stacking** | engine-grouped out-of-fold predictions on training engines → StandardScaler fitted on those only → Ridge with alpha chosen on validation → base models refitted on the full training partition. Coefficients unconstrained; alpha, intercept, coefficients, base-prediction correlations and meta residuals are saved. | 8 |
| **Uncertainty** | EKF covariance (state uncertainty); empirical RUL interval **per predicted-RUL range** (≤ 20, 20–60, > 60) from residual quantiles of the validation engines plus engine-held-out residuals of the whole stack on the training engines (ranges with < 30 residuals use the pooled quantiles); labelled empirical. RUL error by validation engine and degradation phase. Non-negative RUL in the output, raw value kept. | 9 |
| **Evaluation** | test engines, used once: RMSE, MAE, R², NASA score (late predictions penalised more: e^(d/10)−1 for d>0, e^(−d/13)−1 for d<0); each base model vs the stack; by degradation phase, subset, flight class and engine. | 10.1, 10.3 |
| **Ablation E0–E7** | section 7 below. | 10.2 |
| **Explainability** | TreeSHAP of the four models weighted by the Ridge effective weights; importance by feature block; Ridge contributions per model; physics/EKF diagnostics. The backend adds per-prediction SHAP, contributions and occlusion. | 11 |
| **Bundle** | fingerprints, schema, feature list, engine splits, baseline, EKF (Q/R, x0/P0, states), featurizer (fill medians), four models, Ridge + scaler, interval, diagnostics, tuning table, versions, code commit + hash, metrics. | 13.1 |

## 6. Component-health forecasts (extension beyond `pipeline.md`)

`pipeline.md` makes RUL the only ML target. This optional section (`RUN_FORECAST = True`) also forecasts every component's
health **h flights ahead** (default 10, 25, 50), one LightGBM model per component and horizon:

- **Target:** the true health h flights later **minus the current EKF health estimate** (so a model that learns nothing
  reproduces "no change"). The future labels come from the hidden parameters `T`, used as targets only.
- **Inputs:** the same causal features as RUL plus today's EKF health of all five components. Hyperparameters are the
  ones tuned for the RUL LightGBM model.
- **Forecast:** current EKF health + predicted change, clipped to [0, 1], with an **empirical interval** from the
  validation residuals.
- **Excluded rows:** the last h flights of every engine have no label h flights ahead, so they are neither trained nor
  scored (long horizons therefore never see an engine's final flights).
- **Baselines and the "useful" flag:** each forecast is compared with **persistence** (health stays as today) and a
  **straight-line trend** of the EKF health over the last `ROLL` flights, on flights that are degraded *now*
  (current true health < `FORECAST_DEGRADED_BELOW`; chosen on the present, never on the outcome). A forecast is marked
  `useful` only if, on the **validation** engines, it has at least `RELIABLE_MIN_CYCLES` degraded flights and beats both
  baselines. The dashboard should show only `useful` forecasts.
- **Limits:** a forecast starts from today's estimate, so components whose EKF reliability is `low` have weaker forecasts.
  With `RUN="quick"` (DS02) only HPT and LPT degrade, so the other components will mostly be marked not useful. Degradation
  is slow, so "no change" is hard to beat at short horizons; report what the table shows.
- **Output:** `forecast_<component>_h<h>.txt` files and a `forecast` entry in the manifest (horizons, features, intervals,
  `useful`, metrics); the backend adds `fc_<component>_h<h>` (+ `_lo`, `_hi`) columns to every engine payload.

### Component RUL (extension)

N-CMAPSS has one RUL per engine and **no per-component remaining life**, so a component RUL is *derived*: from the hidden
parameters `T` (labels only, offline) we compute when each component's **true** health first falls to `COMP_FAIL_HI`
(0.30, the Emergency band edge) and train one LightGBM model per component on "flights until that happens", from the same
causal features as the engine RUL plus today's EKF health of all components.

- **Censoring:** engines whose component never reaches the level have no label and are excluded from training and scoring
  (the notebook prints how many engines reach it per partition). A prediction only makes sense for a degrading component,
  so the backend/dashboard should show it only while the EKF health is below `show_below` (0.9).
- **Baselines and `useful`:** scored on validation and test engines against (1) the **engine RUL used as the component
  RUL** and (2) a **straight-line extrapolation of the EKF health** to the failure level. A model is `useful` only if, on
  the validation engines, it has enough flights (≥ `RELIABLE_MIN_CYCLES`), at least 2 engines, and beats both baselines.
- **Limits:** components that never degrade in the training engines get no model (on DS02 only HPT and LPT degrade). If the
  EKF cannot separate HPT from LPT (see the observability table), their separate remaining lives are weak: check
  `ekf_reliability` before presenting a component RUL. Expect it to be less accurate than the engine RUL.
- **Output:** `component_rul_<component>.txt` files and a `component_rul` manifest entry (level, cap, intervals, `useful`,
  metrics); the backend adds `crul_<component>` (+ `_lo`, `_hi`) columns to every engine payload.

## 7. Ablation E0–E7

Same engines, folds and hyperparameters; only the features change. "cond" (flight condition and exposure) is in every
experiment.

| ID | Features | Model | Question |
|---|---|---|---|
| E0 | sensor summaries (static) | Random Forest | baseline |
| E1 | sensor summaries (static) | 4-model stack | does stacking help? |
| E2 | + healthy residuals (static) | stack | does condition compensation help? |
| E3 | + physics proxies, EKF states/σ, innovations (static) | Random Forest | physics/EKF features with one model |
| E4 | everything incl. temporal features | stack | **the full pipeline** |
| E5 | E4 without EKF features | stack | does the EKF help? (rule 9) |
| E6 | E4 without physics residuals/proxies/innovations | stack | do physics residuals help? |
| E7 | E4 without temporal features | stack | do temporal features help? |

**Only claim what this table shows** (`pipeline.md` rule 9): the EKF helps if E5 is clearly worse than E4, and so on.

## 8. Output: the bundle

| File | Content |
|---|---|
| `manifest.json` | version 4; files, fingerprints, splits, features, d_ref, base models + parameters, interval, observability, EKF diagnostics, health mapping, reliability, tuning, metrics, config, versions, build |
| `schema.json` | per-file HDF5 inspection and validation reports, variable units, feature blocks |
| `reference.json` / `ekf.json` / `featurizer.json` / `stack.json` | healthy baseline, EKF, features, scaler + Ridge |
| `randomforest_rul.joblib`, `xgboost_rul.json`, `lightgbm_rul.txt`, `catboost_rul.cbm` | the four base models |
| `forecast_<component>_h<h>.txt` | health-forecast models (extension; 15 files with the defaults) |
| `component_rul_<component>.txt` | component-RUL models (extension; one per degrading component) |
| `demo_cycles.csv`, `parity_outputs.csv` | test engines for the dashboard; the notebook's predictions for `check_parity` |
| `per_engine_metrics.csv`, `per_subset_metrics.csv`, `ablation.csv`, `healthy_residuals_by_altitude.csv` | report tables |
| `metrics.json`, `versions.json`, `shap_importance.csv`, `shap_summary.png` | all metrics and timings, library versions, explanations |

Load the bundle with the library versions in its `versions.json` (Colab currently: Python 3.13, scikit-learn 1.6.1).

## 9. Time and resources

**Measured** on the first quick run (Colab CPU, DS02, before early stopping for LightGBM/CatBoost): total **45 min** -
download + aggregation 3.5 min, baseline + EKF 0.1 min, four models (tuning + out-of-fold + final) **14 min**, ablation
**27 min**, forecasts 0.2 min, SHAP 0.1 min. CatBoost (800 trees) took ~1 min per fit and dominated.

Early stopping on validation now cuts the trees of all three boosted models (XGBoost already stopped at 83 of 800 on
DS02), so the times below are **estimates** until the next run prints `timings_s`:

| | Quick (DS02) | Full (9 files) |
|---|---|---|
| Download + validation + aggregation | ~3.5 min (seconds from cache) | ~13–15 min (~1 min from cache) |
| Baseline + EKF + features | < 1 min | ~3–6 min |
| Four models: tuning + out-of-fold + final | ~3–6 min | ~20–45 min |
| Ablation (5 stacks + 2 RF) | ~6–12 min | ~45–90 min |
| Forecasts + component RUL (extensions) + SHAP + export | ~1 min | ~5–10 min |
| **Total** | **~15–25 min** | **~1.5–2.5 h** |

RAM: peak ~5 GB while aggregating the largest file (Colab standard is enough). Disk: ~4 GB at a time. Drive: < 1 GB.
If time is short: `RUN_ABLATION = False` (report it), or `TUNE = False`. Free Colab disconnects idle sessions; keep the tab
active (the aggregation cache survives a disconnect, the rest is recomputed).

## 10. Reading the results

- **RUL:** RMSE in cycles, plus the split by degradation phase; end-of-life error matters most, early-life error is
  largely irreducible. Compare the stack with each base model: `pipeline.md` 7.5 says not to assume the ensemble wins.
- **Ablation:** decides which claims are allowed.
- **Component health:** read the health-mapping table and `reliability`; components marked `low` / not identifiable must
  be shown as low-confidence (or merged), not as a component health percentage (`pipeline.md` 1.1).
- **Forecasts:** read the forecast table (RMSE in health points vs persistence and trend, skill, `useful`). A forecast
  that does not beat persistence on validation is not shown; say so rather than present it.
- **EKF consistency:** mean joint NIS on validation engines near the number of sensors (14), small gated share,
  covariance eigenvalues ≥ 0.

### EKF features and RUL (read this before claiming anything about the EKF)

On the DS02 quick runs the RUL model was **better without the EKF features** (E5 beat E4 by ~1.1 flights, R² 0.90 vs
0.865). `pipeline.md` rule 9 says the EKF may only be claimed to help RUL if E5 is worse than E4, so on DS02 that claim
cannot be made. The EKF's role in this project is **component health, its uncertainty and sensor checks**, not RUL
accuracy. Likely cause: the EKF states are functions of the same residuals the models already see, so they add features
(265 vs 237) without information, and with 5 training engines extra features overfit. Whether this holds on the full
dataset is decided by the full run: `metrics.ekf_in_rul` and ablation E4 vs E5 report both versions, `EKF_IN_RUL = "auto"`
ships the one the validation engines prefer (the spec's full list when in doubt). Do not choose by test results.

### First quick run (DS02, before the three fixes)

| | Test RUL RMSE (flights) |
|---|---|
| Ridge stack (pipeline) | **7.10** (R² 0.875) |
| best single model (Random Forest) | 6.41 |
| old pipeline, same engines | 8.21 |
| Arias Chao et al.: CNN sensors only / hybrid | 4.95 / 4.14 |

- **Ablation:** healthy-baseline residuals are the big gain (raw sensors 18.1 → 7.6; removing residuals → 14.0). Removing
  the EKF features *improved* RUL (6.47) and so did removing temporal features (6.71): on DS02 the EKF does not help RUL,
  so that claim cannot be made from this run.
- **Bias:** predictions are pulled toward the middle (+6 flights at end of life, −10 early). The stack is ≈ 2.9 + 0.915 ×
  the average model, which explains part of it.
- **Validation:** DS02 has only 6 development engines → 1 validation engine; it chose settings poorly (the raw-sensor
  model had the best validation and worst test score) and the interval covered only 61 %. The full run has ~10.
- **EKF:** 7 of 10 parameters never move in DS02 (now excluded); HPT and LPT are not separable on these engines (now
  merged and shown low-confidence). Health error on degrading test flights: HPT 18.5, LPT 23 points.
- **Forecasts:** none marked useful (no degradation for fan/LPC/HPC; too few degraded validation flights for HPT/LPT).

## 11. Deviations from `pipeline.md` (documented)

| Spec | What we do | Why |
|---|---|---|
| "EKF" with a candidate 4-state vector | observation model linear in the state (EKF update exact); state vector = the 10 N-CMAPSS parameters, merged by observability analysis | data-driven state selection instead of guessing which states are identifiable |
| Repeated engine-level splits (10.3, "where feasible") | one fixed split | Colab time budget; the validation fraction and seed are configurable |
| (none: RUL is the only ML target) | **component-health forecasts** and **component RUL** added as optional extensions (section 6) | requested for the dashboard; kept separate so the spec pipeline is unchanged; `RUN_FORECAST = False` / `RUN_COMPONENT_RUL = False` remove them |
| Backend API names (`/health`, `/ready`, `/api/v1/...`), frontend reconnect/stale indicator (12) | not part of training | separate backend/frontend work |

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not download ... from Kaggle` | add `KAGGLE_USERNAME` / `KAGGLE_KEY` Colab secrets, or put the files in Drive |
| `truncated file: eof = 2885034848 ...` / `free block size is zero?` | Kaggle's DS08d-010 is damaged; it is excluded from `FILES` - do not add it back |
| `ValueError: ... not row-aligned` / `not contiguous` / `cycle numbers go backwards` | the file violates an assumption the pipeline depends on; send the message (do not bypass) |
| `NOTE [...]: dropped N corrupt rows` | informational; corrupt rows were removed |
| Drive prompt fails | `USE_DRIVE = False` |
| Out of memory | High-RAM runtime or lower `HEALTHY_SAMPLES_PER_FILE` |
| `check_parity` MISMATCH | library versions differ from `versions.json`, or bundle and code are from different commits |
| Dashboard: "Unsupported bundle version 3" | that is the old pipeline's bundle; install a bundle from this notebook |

## 13. Tests

```bash
python -m pytest backend/tests
```
Hand-set parameters and stub learners (nothing is fitted on synthetic data) check the plumbing: schema validation,
aggregation, baseline maths, EKF (tracking, gating, PSD, Jacobian vs finite differences, state merging, diagnostics),
features (window statistics, causality, leakage audit), stacking (engine splits, out-of-fold correctness, Ridge
algebra, metrics, interval, forecast labels/baselines/metrics), the backend and API (including forecast columns), and the notebook's Layer 1 and engine-split cells on a tiny synthetic
file. Accuracy is measured only by the notebook's evaluation on real test engines.
