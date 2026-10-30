# GENZONE backend (AI Manthan) - NASA N-CMAPSS digital twin

FastAPI service that runs the physics-informed, interpretable pipeline from a trained bundle and serves the dashboard.

```bash
pip install -r backend/requirements.txt
python -m backend                     # http://localhost:8000  (dashboard at /, API under /api)
python -m backend.check_parity        # after installing a new bundle in backend/models/
python -m pytest backend/tests
```

Without a bundle the dashboard shows "No trained model bundle loaded" with instructions (no fake data).

## Pipeline (same code in Colab and here: `backend/pipeline/ncmapss.py`)

| Layer | What | Where |
|---|---|---|
| 1 Ingestion | N-CMAPSS `.h5` (1 Hz flight data) -> one row per flight cycle; causal per-engine healthy baseline | `load_h5`, `aggregate_cycles` |
| 2 Physics | measured-only ratios/efficiency proxies; healthy-engine map gives the ideal sensor value at the same flight condition; residual = measured/ideal - 1 | `physics_1hz`, `HealthyReference` |
| 3 Kalman filter | linearised gas-path model (influence coefficients scheduled on alt/Mach/TRA); estimates the 10 fan/LPC/HPC/HPT/LPT efficiency & flow modifiers with 1-sigma; innovation gating flags bad sensors | `GasPathKF` |
| 4 ML | XGBoost + LightGBM + CatBoost + RandomForest per target + 1D-CNN on 0.1 Hz in-flight windows; per target a non-negative Ridge blend or the best single model (chosen by CV); causal smoothing; RUL 90% interval from quantile models | `bundle.py` |
| 4 XAI | TreeSHAP (trees, weighted by blend weights) + occlusion per sensor through the full stack incl. CNN and Kalman filter | `ModelBundle.explain` |
| 5 Dashboard | component health, KF states, residuals, Brayton core cycle, alerts, explanations | `service.py`, `public/` |
| 6 Continuous learning | not implemented (retrain = re-run the notebook) | - |

Endpoints: `GET /api/health`, `/api/model`, `/api/engines`, `/api/engine/{id}`,
`/api/engine/{id}/explain/{cycle}?target=rul|hi_fan|hi_lpc|hi_hpc|hi_hpt|hi_lpt`, `POST /api/analyze`
(CSV: raw 1 Hz N-CMAPSS columns `alt, Mach, TRA, T2, T24 ... Wf, unit, cycle[, Fc]`, or per-cycle aggregates).

## Honest limits

* Efficiencies are proxies: total P30 and T40 are not measured (HPC uses static Ps30; turbine efficiency and the
  Brayton burner-exit point use the *virtual* T40 and are display-only).
* N-CMAPSS has no burner health parameter, so the burner is not health-tracked.
* The 3D model is the original turbojet geometry: zones are mapped to fan / compressor (LPC+HPC) / turbine (HPT+LPT).
* Few engines per subset (about 6 train / 3 test): metrics on a single subset are noisy; use RUN="full".
* Cross-validation inside the notebook uses Kalman/featurizer parameters fitted on all training engines; the
  held-out test-engine metrics are the clean ones.
* Tests use a stub bundle (fixed formulas, synthetic data); the real model-loading path is verified with
  `check_parity` on the first real bundle.
