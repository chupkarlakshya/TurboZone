# GENZONE backend (AI Manthan) - NASA N-CMAPSS digital twin

FastAPI service that runs the physics-informed, interpretable pipeline from a trained bundle and serves the dashboard.

```bash
pip install -r backend/requirements.txt
python -m backend                     # http://localhost:8000  (dashboard at /, API under /api)
python -m backend.check_parity        # after installing a new bundle in backend/models/
python -m pytest backend/tests
```

Without a bundle the dashboard shows "No trained model bundle loaded" with instructions (no fake data).

## Pipeline (`pipeline.md`; same code in Colab and here: `backend/pipeline/ncmapss.py`, `stacking.py`)

| Layer | What | Where |
|---|---|---|
| 1 Ingestion | N-CMAPSS `.h5` (1 Hz) -> one row per completed flight (sensor mean/std/min/max/last); raw 1 Hz CSV uploads are aggregated the same way | `load_h5`, `aggregate_cycles` |
| 2 Physics | healthy-condition baseline g(W) -> residual = measured / healthy - 1; measured-only ratios and efficiency proxies | `HealthyReference`, `physics_1hz` |
| 3 EKF | reduced-order gas-path model; states (merged by observability), parameter estimates with 1-sigma, prior innovations, per-sensor and joint NIS, gating flags | `GasPathEKF` |
| 4 ML | causal features -> Random Forest, XGBoost, LightGBM, CatBoost -> scaler + Ridge -> RUL; empirical interval | `CausalFeaturizer`, `StackModel`, `bundle.py` |
| 4 Health | component health = EKF estimate mapped by d_ref (not an ML target); `model.reliability` says which components are identifiable | `kf_component_health` |
| 5 XAI | TreeSHAP of the four models weighted by the Ridge effective weights; per-model Ridge contributions; occlusion per sensor through the whole pipeline | `ModelBundle.explain` |
| 4 Forecast (extension) | health h flights ahead per component: current EKF health + LightGBM-predicted change, empirical interval; `fc_<component>_h<h>[_lo/_hi]` payload columns, `model.forecast.useful` says which to show | `ModelBundle._forecast` |
| 4 Component RUL (extension) | flights until a component's health reaches the failure level (LightGBM per degrading component; labels derived offline from the hidden parameters); `crul_<component>[_lo/_hi]` payload columns, `model.component_rul.useful` says which to show | `ModelBundle._component_rul` |
| 6 Dashboard | component health, EKF states, residuals, Brayton core cycle, alerts, explanations; payload has `status`, `model_version`, `prediction_timestamp` | `service.py`, `public/` |

Endpoints: `GET /api/health`, `/api/model`, `/api/engines`, `/api/engine/{id}`,
`/api/engine/{id}/explain/{cycle}?target=rul|hi_fan|hi_lpc|hi_hpc|hi_hpt|hi_lpt`, `POST /api/analyze`
(CSV: raw 1 Hz N-CMAPSS columns `alt, Mach, TRA, T2, T24 ... Wf, unit, cycle[, Fc]`, or per-flight aggregates).
Only version-4 bundles (from the current notebook) load; older bundles are refused with a clear message.

## Combustor and nozzle: physics monitors (`backend/pipeline/physics_monitors.py`)

N-CMAPSS degrades only the fan, LPC, HPC, HPT and LPT, so the Kalman filter has no combustor or nozzle state and
there is nothing to learn them from. They are tracked by first-principles monitors instead, using measured sensors only.
Each formula runs on the measured flight and on the healthy-engine reference at the same condition, and the ratio
of the two is the deviation, so unknown constants (guide-vane area, cooling air, bleeds) cancel.

| Monitor | Formula | Health limits (engineering assumptions) |
|---|---|---|
| Combustion efficiency | T40 from the HP-spool power balance h(T40) = h(T48) + k[h(T30) − h(T24)]; core flow through the choked HPT guide vanes m ∝ (1 + EKF HPT flow modifier)·P40/√T40; η_b ∝ m·[h(T40) − h(T30)] / (Wf·LHV) | tolerance 1 % (fuel-meter accuracy), failure at a 4 % loss |
| Burner pressure ratio | P40 / Ps30 | tolerance 0.5 %, failure at a 3 % drop (about double the design loss) |
| Nozzle effective area | A ∝ m·√T50 / (P50·f(P50/p_amb)), compressible flow function, p_amb from P2 and Mach | tolerance 1 %, failure at ±6 % |

Air properties use the variable-cp fit of Çengel Table A-2c. Monitors are smoothed per engine (5-flight EWMA). Combustor
health is the worse of its two monitors. The columns are `hi_burner`, `hi_nozzle`, `dev_<monitor>[_smooth]`,
`phys_t40_K` and `phys_npr`; triage, alerts, the database and the dashboard treat both parts like any other module.

### Deep physics check

    python -m backend.physics_check        # also served at GET /api/physics/check and on the Gas path tab

This runs 19 checks on every flight of the bundled test engines, in three groups:
* **Conservation and limits:** HP-spool energy balance against the simulator's T40, burner pressure loss (the
  simulator holds it at exactly 4.5 %), burner energy balance across flight conditions, entropy rise across every
  uncooled component, isentropic efficiencies, the Carnot limit, the standard atmosphere and the nozzle regime.
* **Monitors at rest:** flat on healthy flights, and unmoved while the compressor and turbines wear (no false alarms).
* **Fault injection:** a 3 % fuel excess, a 2 % P40 drop and a 5 % larger nozzle are each read back at the injected size.

The film-cooled HPT is excluded from the single-stream entropy check because cooling air joins between stations 40
and 48; the spool energy balance covers it instead.

## Maintenance workflow (SQLite, `backend/pipeline/store.py`)

The engineer's loop on top of the pipeline: triage the fleet -> open an engine -> read the advisory -> handle alerts
-> log the inspection finding. State lives in SQLite (standard library, no server): `backend/data/genzone.db`, or
`GENZONE_DB=<path>`. The folder is git-ignored.

| Table | Contents | Source |
|---|---|---|
| `engines` | latest summary per engine: RUL, planning RUL, health band, urgency, localisation | derived from the bundle, rebuilt on start |
| `flight_predictions` | RUL, interval and component health for every flight shown | derived from the bundle, rebuilt on start |
| `alerts` | alerts from `derive_alerts`, with status `open` / `acknowledged` / `dismissed`, who handled them and when | derived, engineer-edited |
| `maintenance_events` | inspections: type, module, finding (`confirmed` / `not_found` / `inconclusive`), observed health, action, who, when | engineer input |

When `SCHEMA_VERSION` changes, the derived tables are dropped and rebuilt. Alerts and maintenance events are kept.

| Endpoint | Purpose |
|---|---|
| `GET /api/fleet?as_of=0.5` | ranked triage list. `as_of` replays the fleet at that share of each engine's recorded flights (the demo engines are run-to-failure histories, so their last flight is always end of life) |
| `GET /api/engine/{id}/alerts[?status=open]`, `POST /api/alerts/{alert_id}` `{status, user, note}` | alert handling with an audit trail |
| `GET`/`POST /api/engine/{id}/maintenance` | inspection log. `performed_by` is required. Linking `alert_id` acknowledges the alert |
| `GET /api/learning/summary`, `/api/learning/export.csv` | findings joined with the prediction shown at that flight (flag precision, health-estimate error), for offline recalibration. The deployed model is never updated automatically |

The engine payload carries two additions:
* `advisory`: planning RUL = lower (5th-percentile) interval bound; urgency = the worse of the health band and a
  planning-RUL level (<= 30 / 15 / 5 flights); localisation reports the EKF-merged group (e.g. "Fan / HPT / LPT")
  when the most degraded component is not identifiable.
* `forecast_display`: only forecasts that beat both baselines on validation engines. `confident` if the component
  is reliable, `indicative` (group trend only) otherwise.

Dashboard side (`public/index.html`, `app.js`, `app.css`, no build step, no framework):
* Left rail: the fleet queue from `/api/fleet`, with the replay slider.
* Engine tabs: **Overview** (RUL chart with interval and planning RUL; the decision panel; the 3D engine
  `public/models/Turbojet Engine.glb` tinted section by section with health, station temperatures at the section
  boundaries and leader lines to the module cards), **Gas path** (Brayton T–s cycle, every sensor against the healthy baseline with the filter
  check, EKF parameters), **Why this prediction** (TreeSHAP groups, sensor occlusion, the four-model blend, test
  metrics), and **Maintenance** (alerts, inspection form, record, findings export).
* The URL hash `#/<engine>/<flight>/<tab>` deep-links a view. Arrow keys step through flights; space plays.
* 3D sections: the model is an STL conversion whose 16 meshes each span the whole engine, so `SECTIONS` in `app.js`
  assigns vertices by axial position (model X, inlet at +X): spinner and first stage, front compressor stages (LPC),
  rear compressor stages (HPC), combustor, two turbine-disc groups (HPT, LPT), exhaust nozzle. The boundaries were
  read off a side section of the geometry. The model is a single-spool turbojet, so the mapping is illustrative; if
  WebGL or the model is unavailable the page shows a 2D schematic instead.
* The previous dashboard remains at `/legacy.html`.

## Honest limits

* Efficiencies are proxies: total P30 and T40 are not measured (HPC uses static Ps30; turbine efficiency and the
  Brayton burner-exit point use the *virtual* T40 and are display-only).
* N-CMAPSS has no burner health parameter, so the burner is not health-tracked.
* The 3D model is the original turbojet geometry: zones are mapped to fan / compressor (LPC+HPC) / turbine (HPT+LPT).
* Few engines per subset (about 6 train / 3 test): metrics on a single subset are noisy; use RUN="full".
* The EKF and the healthy baseline are calibrated on all training engines; out-of-fold predictions use cross-fitted
  baseline residuals, but the EKF calibration is shared across folds. The held-out test-engine metrics are the clean ones.
* Not yet implemented from `pipeline.md` section 12: `/ready`, versioned `/api/v1` routes, and a reconnecting frontend
  client with a stale-data indicator.
* Tests use a stub bundle (fixed formulas, synthetic data); the real model-loading path is verified with
  `check_parity` on the first real bundle. Tests run against an in-memory database (`tests/conftest.py`).
* The maintenance store has no authentication: `performed_by` is a free-text name, so the audit trail is only as good
  as what users type in. It is a single-process SQLite file, fine for a demo but not for a multi-user deployment.
