# GENZONE | Turbofan Digital Twin (AI Manthan)

Physics-informed, interpretable health monitoring for a two-spool turbofan, trained on **NASA N-CMAPSS**
(C-MAPSS 2, Kaggle mirror `bishals098/nasa-cmapss-2-engine-degradation`). It predicts component-wise health
indices (fan, LPC, HPC, HPT, LPT) and remaining useful life, and explains every prediction.

## Pipeline
1. **Ingestion & preprocessing** – 1 Hz flight data aggregated to one row per flight cycle; per-engine healthy baseline.
2. **Physics model** – efficiency proxies and pressure ratios from measured sensors; a healthy-engine map predicts the
   ideal sensor values at the same flight condition (measured vs ideal residuals).
3. **Kalman filter** – linearised gas-path model estimating the 10 component efficiency/flow modifiers with uncertainty
   and flagging inconsistent sensors.
4. **Interpretable ML** – XGBoost, LightGBM, CatBoost, Random Forest on per-cycle physics features, plus a 1D-CNN on
   0.1 Hz in-flight windows enriched with residuals and Kalman states (hybrid design of Arias Chao et al.). Per target
   a non-negative Ridge blend is kept only if it beats the best single model in cross-validation; predictions are
   causally smoothed per engine; health is scaled per component; TreeSHAP plus occlusion; RUL 90% interval.
5. **Dashboard** – component health, Kalman states, residuals, core Brayton cycle, alerts, explanations (`public/`).
6. **Continuous learning** – retraining = re-running the notebook (not automated yet).

## Running
1. Train on Google Colab: open `training/ncmapss_colab_training.ipynb`, run all cells (`RUN="quick"` for DS02,
   `RUN="full"` for all 10 files, DS01–DS08d). It downloads a bundle zip.
2. Unzip the bundle into `backend/models/`.
3. `pip install -r backend/requirements.txt` then `python -m backend` and open http://localhost:8000.
4. Optional check: `python -m backend.check_parity`.

Details, limits and API: `backend/README.md`. The notebook is generated from `training/ncmapss_colab_training.py`
(`python training/build_notebook.py`) so it always contains the same pipeline code as the backend.
