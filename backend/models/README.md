# Model bundle (put the unzipped Colab export here)

`training/ncmapss_colab_training.ipynb` produces `genzone_ncmapss_<run>_bundle.zip`. Unzip it **into this folder**
(so `backend/models/manifest.json` exists) and restart the backend. Then run `python -m backend.check_parity` once:
it re-computes the bundled test engines and compares them with the notebook's own predictions.

## Files

| File | Content |
|---|---|
| `manifest.json` | version 3: targets, base models, blend weights / chosen model, smoothing, per-component d_ref, features, band edges, metrics |
| `reference.json` | Layer 2 healthy-engine map (polynomial coefficients per sensor) |
| `kf.json` | Layer 3 Kalman filter (influence coefficients H, bias b, noise R/Q, initial state) |
| `featurizer.json` | causal feature engineering (z-score stats, baseline window, feature order) |
| `<model>_<target>.<ext>` | XGBoost `.json`, LightGBM `.txt`, CatBoost `.cbm`, RandomForest `.joblib` per target |
| `cnn.onnx` (+ `cnn.pt`) | multi-output 1D-CNN (RUL + 5 component health indices) on 500 s windows of 0.1 Hz data |
| `demo_samples.npz` | 0.1 Hz flight condition + sensors of the test engines (CNN input for the dashboard) |
| `lightgbm_rul_q05.txt`, `q95.txt` | RUL quantile models (90% interval) |
| `demo_cycles.csv` | per-cycle aggregates of the held-out test engines (shown in the dashboard) |
| `parity_outputs.csv` | the notebook's predictions for those engines (used by `check_parity`) |
| `shap_*.csv`, `shap_summary.png` | global TreeSHAP importance per target |
| `metrics.json`, `versions.json` | test metrics, timings, library versions |

Targets: `rul` (cycles) and `hi_fan`, `hi_lpc`, `hi_hpc`, `hi_hpt`, `hi_lpt` (1 = healthy, 0 = failure-level
damage). Library versions on the backend should match `versions.json`.
