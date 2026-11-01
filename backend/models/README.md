# Model bundle (put the unzipped Colab export here)

`training/ncmapss_colab_training.ipynb` produces `genzone_ncmapss_<run>_bundle.zip`. Unzip it **into this folder**
(so `backend/models/manifest.json` exists) and restart the backend. Then run `python -m backend.check_parity` once:
it re-computes the bundled test engines and compares them with the notebook's own predictions.

Only version-4 bundles (pipeline.md pipeline) load. The previous pipeline's bundle is kept for reference in
`docs/genzone_ncmapss_full_bundle/` and is not loadable by the current backend.

## Files

| File | Content |
|---|---|
| `manifest.json` | version 4: files + fingerprints, engine splits, features, d_ref, base models + parameters, interval, observability, EKF diagnostics, health mapping, reliability, tuning, metrics, config, versions, build |
| `schema.json` | per-file HDF5 inspection/validation reports, variable units, feature blocks |
| `reference.json` | healthy-condition baseline g(W) (polynomial coefficients per sensor) |
| `ekf.json` | EKF: observation model b, B, state groups/loadings, R, Q (+ scale), x0, P0, gate |
| `manifest.json` → `model_features` | the columns the RUL models use (a subset of `features` when the EKF features were left out, see `metrics.ekf_in_rul`) |
| `featurizer.json` | causal features (temporal bases, static columns, window, training-median fill values) |
| `stack.json` | Ridge meta-learner + the scaler fitted on out-of-fold predictions |
| `randomforest_rul.joblib`, `xgboost_rul.json`, `lightgbm_rul.txt`, `catboost_rul.cbm` | the four base models |
| `forecast_<component>_h<h>.txt` | optional health-forecast models (extension; the manifest's `forecast` entry lists horizons, intervals, `useful`) |
| `component_rul_<component>.txt` | optional component-RUL models (extension; the manifest's `component_rul` entry lists the failure level, cap, intervals, `useful`) |
| `demo_cycles.csv` | per-flight aggregates of the held-out test engines (shown in the dashboard) |
| `parity_outputs.csv` | the notebook's predictions for those engines (used by `check_parity`) |
| `per_engine_metrics.csv`, `per_subset_metrics.csv`, `ablation.csv`, `healthy_residuals_by_altitude.csv` | result tables |
| `metrics.json`, `versions.json`, `shap_importance.csv`, `shap_summary.png` | metrics + timings, library versions, explanations |

ML target: `rul` (flight cycles). Component health (`hi_*`, 1 = healthy, 0 = failure-level damage) comes from the EKF.
Library versions on the backend should match `versions.json`.
