# GENZONE | Turbofan Digital Twin (AI Manthan)

Physics-informed, explainable health monitoring for a two-spool turbofan, implemented as specified in
**[pipeline.md](pipeline.md)** and trained on **NASA N-CMAPSS** (Kaggle mirror `bishals098/nasa-cmapss-2-engine-degradation`).
It predicts remaining useful life (RUL) with a stacking ensemble, estimates component health with an Extended Kalman
Filter, and explains every prediction.

> **ML team:** the full training guide (dataset, training script, configuration, every pipeline step, outputs, times,
> troubleshooting) is in **[training/README.md](training/README.md)**.

## Pipeline (`pipeline.md`)
1. **Ingestion** – HDF5 schema inspection and validation (columns by name, row alignment, ordering), one row per
   completed flight (sensor mean/std/min/max/last), engine split train / validation / test before anything is fitted.
2. **Physics** – healthy-condition baseline g(W) fitted on healthy training rows (cross-fitted residuals), pressure-ratio
   and efficiency proxies from measured stations only.
3. **EKF** – reduced-order gas-path model `residual = b(W) + B(W)·x`; states selected by observability analysis;
   Joseph-form Kalman update with innovation gating; Q/R estimated from training data, Q scale tuned on validation.
4. **ML** – Random Forest, XGBoost, LightGBM, CatBoost on causal physics/EKF features → engine-grouped out-of-fold
   predictions → scaler + Ridge → RUL with an empirical interval. Ablation E0–E7 tests what each layer adds.
5. **Health** – component health = EKF estimates mapped by each component's failure-level damage, validated against
   the truth; components that the sensors cannot identify are flagged low-confidence.
6. **Component RUL (extension beyond the spec)** – flights until each degrading component reaches its failure level, scored
   against the engine RUL and an EKF-health trend; only components where it beats both are marked useful.
7. **Forecasts (extension beyond the spec)** – component health 10/25/50 flights ahead, compared against "no change" and
   a straight-line trend; only forecasts that beat both are marked useful.
8. **Explainability & dashboard** – SHAP, Ridge contributions, occlusion per sensor, EKF diagnostics; FastAPI backend +
   the dashboard (`public/`).

## Running
1. Train on Google Colab (a CPU runtime is enough): open `training/ncmapss_colab_training.ipynb`, run all cells
   (`RUN="quick"` for DS02, ~15–25 min; `RUN="full"` for 9 files (DS08d excluded), ~65–110 min; estimates). The script downloads the
   data from Kaggle; a Drive cache makes re-runs fast. It downloads a bundle zip.
2. Unzip the bundle into `backend/models/`.
3. `pip install -r backend/requirements.txt`, then `python -m backend` and open http://localhost:8000.
4. Check: `python -m backend.check_parity` (must print `PARITY OK`).

No bundle is installed right now: the previous one came from the old (pre-`pipeline.md`) pipeline and is kept only in
`docs/genzone_ncmapss_full_bundle/`. Until the new notebook is run, the dashboard shows "No trained model bundle loaded".

## Repository layout
| Path | Content |
|---|---|
| `training/` | Colab training notebook + its source script, notebook builder, **training guide** |
| `backend/pipeline/ncmapss.py`, `stacking.py` | shared pipeline code (used by training and serving) |
| `backend/` | FastAPI service, bundle loader, parity check, tests |
| `backend/models/` | the trained model bundle served by the dashboard |
| `public/` | dashboard: `index.html` + `app.js` + `app.css` (fleet queue, RUL, gas-path health, physics, explanations, maintenance log); the previous 3D dashboard is kept at `legacy.html` |
| `docs/` | project documents, reference papers, the old pipeline's bundle (reference only) |
| `pipeline.md` | the specification this project implements |

Details, limits and API: `backend/README.md`. The notebook is generated from `training/ncmapss_colab_training.py`
(`python training/build_notebook.py`) so it always contains the same pipeline code as the backend.
