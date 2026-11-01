# N-CMAPSS Physics-Informed EKF + Stacking Ensemble Pipeline

**Project:** Aircraft-engine Remaining Useful Life (RUL) prediction and
explainable digital twin\
**Dataset:** NASA N-CMAPSS (New Commercial Modular Aero-Propulsion
System Simulation)\
**Model family:** Reduced-order thermodynamic / physics-informed state
model + Extended Kalman Filter (EKF) + Random Forest, XGBoost, LightGBM,
CatBoost + Ridge stacking\
**Frontend:** Existing 3D engine model and frontend (retain; do not
rebuild)\
**Backend:** Python inference service connected continuously to the
existing frontend\
**Status:** Implementation specification. Items marked *calibrate*,
*verify* or *measure* must be established from the selected HDF5 subset
and experiments. No unmeasured performance or calibration values are
assumed.

------------------------------------------------------------------------

## 0. Non-negotiable engineering rules

1.  **Use the N-CMAPSS schema, not the older C-MAPSS FD001--FD004
    text-file schema.** N-CMAPSS is distributed as HDF5 data and has
    richer operating-condition, measured-sensor, virtual-sensor,
    degradation and auxiliary groups.
2.  **Never assume sensor column order or names.** Read the dataset
    Check shapes and units before as    README/metadata and build a verified column map for the exact file.
signing names.
3.  **Do not call each 1 Hz telemetry row an engine cycle.** N-CMAPSS
    contains high-rate telemetry within flights and a separate
    cycle/flight identifier. RUL is expressed in cycles. Preserve
    timestamps/row order and aggregate or sample at a clearly defined
    prediction cadence.
4.  **Split by engine unit/trajectory before fitting any preprocessing,
    healthy baseline, physics calibration, EKF noise parameters, feature
    transforms, models or meta-model.** Do not randomly split rows from
    the same trajectory across train and test.
5.  **Keep hidden degradation parameters `T` out of online model
    inputs.** They may be used for offline physics
    calibration/diagnostics only when their definitions are verified,
    and for post-hoc validation. Do not use them as inference features.
6.  **Treat `Y` as the RUL target, not an input feature.** Never use
    future RUL or future observations to create current-time features.
7.  **Do not claim a full high-fidelity thermodynamic simulator.** This
    project implements a reduced-order, dataset-calibrated
    gas-path/thermodynamic observation model. Its equations and
    calibration quality must be documented.
8.  **Do not hard-code guessed EKF values, physical efficiencies, sensor
    noise, degradation rates, or model scores.** Estimate them from
    training data and record the fitted values in versioned
    configuration/artifacts.
9.  **Do not claim that EKF improves RUL unless the ablation study
    demonstrates it.** EKF state estimation, physics residuals, ML
    prediction and RUL are different outputs.
10. **The existing frontend must remain connected to the backend.** No
    silent switch to mock values. Show backend status, model version,
    timestamp and stale-data state. Reconnect automatically after
    transient failures.

------------------------------------------------------------------------

## 1. System objective and architecture

At each permitted prediction time for an engine, estimate its current
latent degradation/health state and predict remaining operating cycles
to end-of-life.

``` text
NASA N-CMAPSS HDF5
       |
       v
Schema inspection + metadata validation
       |
       v
Trajectory reconstruction (dataset + unit + flight/cycle + timestamp)
       |
       +-------------------------------+
       |                               |
       v                               v
Measured sensors X_s              Operating conditions W
       |                               |
       +---------------+---------------+
                       |
                       v
         Healthy-condition baseline model
         + reduced-order thermodynamic model
                       |
                       v
          Physics-predicted sensor outputs
                       |
           actual - predicted residuals
                       |
                       v
        Extended Kalman Filter (EKF)
        latent degradation-state estimate
        covariance + innovations
                       |
                       v
           Causal feature construction
                       |
          +------------+------------+
          |            |            |
          v            v            v
       XGBoost      LightGBM      CatBoost       Random Forest
          \            |            |             /
           \-----------+------------+------------/
                       |
             Out-of-fold base predictions
                       |
                       v
              Ridge meta-learner
                       |
                       v
          Final RUL + uncertainty/diagnostics
                       |
             SHAP + feature importance
                       |
                       v
              Python backend API
                       |
             persistent connection
                       |
                       v
        EXISTING frontend + 3D engine
```

### 1.1 Outputs

The backend should expose:

-   `rul_cycles`: predicted remaining operating cycles.
-   `health_state`: estimated latent degradation state(s), with
    units/definition.
-   `health_index`: optional 0--100 display index, only after defining
    and validating the mapping. It is not automatically a physical
    percentage.
-   `state_uncertainty`: EKF covariance or a documented uncertainty
    summary.
-   `innovation_summary`: sensor residual/innovation diagnostics.
-   `top_features`: SHAP explanation for the current prediction.
-   `model_contributions`: base-model predictions and Ridge contribution
    breakdown.
-   `status`: healthy, warning, degraded, stale, unavailable or error.
-   `model_version`, `dataset_id`, `unit_id`, `prediction_timestamp`.

Do not display a component-specific health percentage unless that
component state is identifiable and calibrated. If not identifiable,
display a latent/system degradation indicator with an explicit label.

------------------------------------------------------------------------

## 2. N-CMAPSS dataset specification

### 2.1 Dataset facts

N-CMAPSS contains synthetic run-to-failure turbofan-engine trajectories
under realistic flight conditions. The dataset paper describes eight
subsets and multiple degradation modes affecting flow and/or efficiency
of rotating components. The files are HDF5. The dataset is not
interchangeable with the older C-MAPSS FD001--FD004 benchmark.

Primary references:

-   Chao, Kulkarni, Goebel and Fink (2021), *Aircraft Engine
    Run-to-Failure Dataset under Real Flight Conditions for Prognostics
    and Diagnostics*, Data, 6(1), 5. https://doi.org/10.3390/data6010005
-   NASA PCoE dataset repository:
    https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
-   Darrah et al. (2022), *Developing Deep Learning Models for System
    Remaining Useful Life Predictions: Application to Aircraft Engines*,
    NASA technical report / PHM paper:
    https://ntrs.nasa.gov/citations/20220009583

### 2.2 HDF5 groups

The common N-CMAPSS data organisation includes:

  -------------------------------------------------------------------------
  Group                   Meaning                 Pipeline treatment
  ----------------------- ----------------------- -------------------------
  `W`                     Flight/operating        Model inputs and
                          conditions              condition variables

  `X_s`                   Measured sensor         Primary sensor
                          telemetry               observations for physics
                                                  model and EKF

  `X_v`                   Virtual/model-derived   Optional feature source
                          sensors                 only after checking
                                                  availability and
                                                  provenance

  `T`                     Unobservable            Offline
                          degradation parameters  calibration/diagnostics
                                                  only; never online input

  `A`                     Auxiliary data such as  Grouping, ordering,
                          unit, cycle, flight     healthy-data selection
                          class and health-state  and audit
                          fields                  

  `Y`                     RUL labels              Supervised training
                                                  target only
  -------------------------------------------------------------------------

**Important:** Exact array dimensions, channel order, names and some
auxiliary layouts can vary by file/release. Inspect every selected HDF5
file and its accompanying README. Do not rely on a guessed `X_s` column
index.

### 2.3 Known variable semantics to verify against the selected file's metadata

NASA's N-CMAPSS technical literature documents the following
operating-condition fields:

  Symbol   Meaning                   Reported unit
  -------- ------------------------- ---------------
  `alt`    Altitude                  ft
  `Mach`   Mach number               dimensionless
  `TRA`    Throttle resolver angle   \%
  `T2`     Fan inlet temperature     °R

A documented measured telemetry set includes:

  Symbol   Meaning                  Reported unit
  -------- ------------------------ ---------------
  `Wf`     Fuel flow                pps
  `Nf`     Physical fan speed       rpm
  `Nc`     Physical core speed      rpm
  `T24`    LPC outlet temperature   °R
  `T30`    HPC outlet temperature   °R
  `T48`    HPT outlet temperature   °R
  `T50`    LPT outlet temperature   °R
  `P15`    Bypass-duct pressure     psia
  `P2`     Fan inlet pressure       psia
  `P21`    Fan outlet pressure      psia
  `P24`    LPC outlet pressure      psia
  `Ps30`   HPC outlet pressure      psia
  `P40`    Burner outlet pressure   psia
  `P50`    LPT outlet pressure      psia

This list is a **metadata cross-check**, not permission to assign these
names to columns by position. Confirm that each symbol is present in the
exact subset and group. Some station variables needed for a complete
cycle calculation may be absent from measured sensors; do not invent
them.

### 2.4 Initial subset selection

-   Make the subset configurable using `NC_MAPSS_DATASET_ID`.
-   Start with **one subset only** for the first working prototype.
-   Use the official development/train and test partition supplied with
    the selected release wherever available.
-   Keep the official test trajectories untouched until final
    evaluation.
-   If a development partition must be split further, split by complete
    engine trajectories using a fixed seed.
-   Do not pool subsets until the pipeline has passed schema and unit
    tests separately for each subset.
-   Unit identifiers may repeat across files/subsets. Use a composite
    identifier such as `(dataset_id, unit_id)`.

### 2.5 Data loading and validation

Use `h5py`, `numpy` and `pandas`/`polars`.

At ingestion:

1.  List HDF5 keys and inspect shapes, dtypes and attributes.
2.  Load the accompanying metadata/README.
3.  Assert that row-aligned groups have compatible record counts.
4.  Build an explicit column-name-to-index mapping from metadata.
5.  Validate `A` fields and determine how unit, cycle/flight, flight
    class and health-state values are encoded.
6.  Verify that `Y` is aligned with the correct telemetry rows and
    determine its exact units/definition.
7.  Check missing values, infinities, duplicates and trajectory
    ordering.
8.  Save a schema manifest and dataset fingerprint.

Example inspection code:

``` python
import h5py

def inspect_h5(path):
    with h5py.File(path, "r") as f:
        for key in f.keys():
            obj = f[key]
            if hasattr(obj, "shape"):
                print(key, "shape=", obj.shape, "dtype=", obj.dtype)
            else:
                print(key, "group")
                for subkey in obj.keys():
                    item = obj[subkey]
                    print("  ", subkey, getattr(item, "shape", None),
                          getattr(item, "dtype", None))
```

The loader must be adapted after inspection; do not assume all keys are
datasets rather than groups.

------------------------------------------------------------------------

## 3. Preprocessing and trajectory construction

### 3.1 Preserve the correct hierarchy

A record must be traceable to:

-   dataset subset;
-   engine/unit;
-   flight or cycle identifier;
-   within-flight sample/time index, if supplied;
-   original row index.

Construct a stable key:

``` text
trajectory_id = dataset_id + ":" + unit_id
sample_id = trajectory_id + ":" + flight_or_cycle + ":" + within_flight_index
```

If the source does not provide a within-flight timestamp, preserve
original row order and do not fabricate clock times.

### 3.2 Do not confuse sample rows with cycles

N-CMAPSS contains telemetry sampled within flights. A single
flight/cycle may contribute many sensor rows. RUL is in operating
cycles, not raw 1 Hz samples.

Choose and document one inference cadence:

-   **Flight-end cadence (recommended for the first prototype):**
    aggregate the telemetry of a completed flight/cycle and issue one
    RUL estimate per completed flight/cycle.
-   **Within-flight cadence:** only if the intended use requires it;
    keep sample-level timestamps and ensure the RUL target and
    validation protocol are aligned to the same point in time.

For flight-end aggregation, calculate physically meaningful summaries
per flight: mean, standard deviation, minimum, maximum, last value, and
optionally robust quantiles. Keep the last measurement separately. Do
not average across different flights.

### 3.3 Cleaning

-   Preserve real degradation-related extremes.
-   Remove a row only when it is corrupt or invalid according to
    documented rules.
-   Impute missing values using training-only fitted imputers.
-   Add missingness indicators when missingness itself is informative.
-   Never interpolate using future rows when generating a current-time
    prediction.
-   Check sensor ranges against metadata and engineering plausibility.

### 3.4 Operating-condition normalisation

Flight conditions vary continuously. Do not use the old C-MAPSS
FD001--FD004 six-operating-condition normalisation recipe blindly.

Use one of these controlled approaches:

1.  Healthy-condition baseline model `g_j(W)` for each sensor `j`,
    trained only on healthy rows in the training engines; or
2.  A condition-aware regression/smoothing model trained on healthy
    training data.

For sensor `j`:

``` text
baseline_j = g_j(W)
residual_j = measured_sensor_j - baseline_j
```

Use cross-fitting or a held-out healthy validation group when generating
residuals for training rows. Do not allow the baseline model to predict
its own training observations in a way that creates unrealistically
small residuals.

The health-state flag `hs` may be used to select healthy rows during
**offline training/calibration**, if the field is present and its
encoding is verified. It is not assumed to be available in live
deployment.

### 3.5 Scaling

-   Tree models do not require standardisation, but can use consistent
    raw/derived features.
-   EKF inputs and state variables need sensible numerical scales and
    unit consistency.
-   Ridge meta-features should be standardised using a scaler fitted
    only on out-of-fold training predictions.
-   Fit all scalers on training data only.
-   Persist scalers with the model artifact.

### 3.6 Target

Use the provided `Y` only after verifying its meaning, shape, alignment
and units. The supervised target should represent remaining operating
cycles at the selected prediction cadence.

Do not apply a capped RUL target unless explicitly justified and
documented. If using a cap, report both capped-target and uncapped-cycle
evaluation, and compare only like-for-like protocols.

------------------------------------------------------------------------

## 4. Reduced-order physics-informed engine model

### 4.1 What this model is

Implement a **reduced-order gas-path/thermodynamic model**, not a full
C-MAPSS simulator and not a claim of high-fidelity CFD.

The model has two jobs:

1.  Represent physically meaningful relationships between flight
    conditions, component performance and measured gas-path variables.
2.  Predict nominal/expected sensor behaviour so that deviations can be
    used as degradation evidence.

N-CMAPSS data were generated using a turbofan engine simulation and
imposed degradation. Reconstructing a compact physical representation is
an independent modelling layer; it is not necessary to re-simulate the
entire engine.

### 4.2 Component states: start small

Do not begin with four arbitrary health percentages. The exact
degradation modes differ across subsets and can affect flow capacity and
efficiency of rotating components.

Start with a compact state vector whose variables have identifiable
sensor support. Suggested first prototype:

``` text
x = [
    d_LPC_efficiency,
    d_HPC_efficiency,
    d_turbine_efficiency,
    d_flow_capacity
]
```

This is a **candidate reduced state**, not a guaranteed identifiable
state for every subset. If observability analysis shows that a state
cannot be distinguished from another, merge it or remove it. Expand to
component-specific flow/efficiency states only when the selected
subset's degradation metadata and measured channels support that
resolution.

Define each degradation variable with a documented convention. Example:
`d = 0` is the calibrated healthy reference; increasing `d` means
increasing degradation. Do not call `1-d` a health percentage until the
scale has been calibrated and validated.

### 4.3 Thermodynamic relationships

Use verified sensor stations and metadata. The following relationships
are candidates when the necessary inputs are present:

**Compressor pressure ratio**

``` text
PR_LPC = P24 / P2
PR_HPC = Ps30 / P24
```

Check exact station definitions and use consistent absolute pressure
units.

**Compressor isentropic efficiency**

For a compressor with known inlet and outlet total temperatures and
pressure ratio:

``` text
T_out_ideal = T_in * PR ** ((gamma - 1) / gamma)
eta_c = (T_out_ideal - T_in) / (T_out_actual - T_in)
```

Use the correct station temperatures for the component being modelled.
If the exact inlet/outlet temperature or pressure is not measured, do
not calculate a supposedly measured efficiency from incomplete channels.
Instead, estimate a reduced-order relationship or use a calibrated
surrogate.

**Turbine efficiency**

A turbine efficiency calculation requires appropriate inlet/outlet total
temperatures and pressure ratio. Use the correct station definitions and
thermodynamic convention. Do not infer turbine inlet temperatures from
an unrelated station or silently substitute missing channels.

**Flow and speed relationships**

Use available fuel flow, corrected/physical speeds and pressures as
observation variables only after checking their exact definitions and
units. Corrected flow/speed calculations require inlet total pressure
and temperature and appropriate reference conditions.

### 4.4 Unit conversion

Use a single internal SI convention for physical calculations:

-   °R to K: `K = °R * 5/9`
-   psia to Pa: `Pa = psia * 6894.757293`
-   pounds per second to kg/s: `kg/s = pps * 0.45359237`
-   rpm to rad/s: `rad/s = rpm * 2*pi/60`
-   altitude ft to m: `m = ft * 0.3048`

Mach number is dimensionless. TRA must retain its documented
percentage/angle convention; do not reinterpret it without metadata.

Retain original units in raw storage and convert only in the physics
layer. Unit tests must verify each conversion.

### 4.5 Healthy reference model

Fit a condition-dependent healthy sensor model on healthy training
observations:

``` text
z_healthy = g(W)
```

For each selected measured channel, `g` may be a regularised regression,
spline, Gaussian process, or another lightweight model. Start with a
transparent model and inspect residuals by operating-condition range.

This gives the expected healthy measurement under the current flight
condition.

### 4.6 Degradation-aware observation model

Build a calibrated reduced-order mapping:

``` text
z_pred = h(W, x)
```

where:

-   `W` is the current flight condition;
-   `x` is the latent degradation state;
-   `z_pred` is the predicted measured sensor vector.

A practical implementation is:

``` text
z_pred = z_healthy(W) + B(W) @ x
```

or a nonlinear calibrated form:

``` text
z_pred = h_theta(W, x)
```

`B` or `theta` must be estimated from training trajectories and checked
on held-out engines. Constrain the model so that the direction and
magnitude of state effects are physically plausible. Do not assert a
universal sign for every sensor without validating the component/station
relationship.

If the physics equations cannot be calibrated from the available
channels, use a clearly labelled **physics-informed sensor-response
surrogate** rather than inventing thermodynamic parameters.

### 4.7 Offline use of `T`

The `T` group contains degradation parameters described as unobservable
in the operational telemetry. It can be useful for:

-   checking whether the latent state tracks the simulator's
    degradation;
-   calibrating a mapping from simulator degradation parameters to the
    reduced state;
-   testing whether proposed state variables are identifiable;
-   offline component-state validation.

It must not be supplied to the online EKF or RUL model if it would not
be available at inference time. Record all `T`-based experiments
separately from deployable results.

### 4.8 Physics calibration procedure

1.  Select one subset and inspect its metadata.
2.  Map verified `W`, `X_s`, `A`, `Y`, and (for offline analysis only)
    `T` fields.
3.  Select a healthy training-only sample using verified `hs`.
4.  Fit `z_healthy = g(W)`.
5.  Fit the degradation-response mapping using training data only.
6.  Compare predicted and measured sensor values by engine, flight
    class, operating-condition range and degradation phase.
7.  Check monotonic/physical constraints where justified.
8.  Perform observability/sensitivity analysis.
9.  Remove or merge poorly identifiable states.
10. Freeze and version the calibrated parameters before validation/test
    evaluation.

**Calibration artifacts:** `healthy_baseline.joblib`,
`physics_parameters.json`, `sensor_map.json`, `unit_map.json`,
calibration plots and a calibration report.

------------------------------------------------------------------------

## 5. Extended Kalman Filter (EKF)

### 5.1 State-space formulation

Use the selected latent degradation state vector:

``` text
x_k = [d_LPC_efficiency, d_HPC_efficiency,
       d_turbine_efficiency, d_flow_capacity]^T
```

The exact state dimension is configurable after observability analysis.

**State transition:**

``` text
x_k = f(x_(k-1), u_k, delta_exposure_k) + w_k
```

where `u_k` is operating condition and `delta_exposure_k` is the elapsed
operating exposure between prediction points. Since degradation is
associated with engine use/flight progression, do not advance the
degradation state once per raw telemetry row as if each row were a
flight cycle.

For the first version, use a constrained random-walk or monotonic drift
model:

``` text
x_k = x_(k-1) + a(u_k) * delta_exposure_k + w_k
```

Only impose monotonicity if supported by the degradation definition and
data. Sensor noise or a temporary performance change must not be
interpreted as recovery of physical damage.

**Observation:**

``` text
z_k = h(W_k, x_k) + v_k
```

Use selected measured `X_s` channels and the calibrated physical
observation function.

### 5.2 EKF equations

Prediction:

``` text
x_minus = f(x_previous, u_k)
P_minus = F_k @ P_previous @ F_k.T + Q_k
```

where `F_k` is the Jacobian of `f` with respect to the state.

Measurement update:

``` text
innovation = z_k - h(W_k, x_minus)
S_k = H_k @ P_minus @ H_k.T + R_k
K_k = P_minus @ H_k.T @ inv(S_k)
x_updated = x_minus + K_k @ innovation
```

where `H_k` is the Jacobian of `h` with respect to the state.

Use the Joseph covariance update for numerical stability:

``` text
I_KH = I - K_k @ H_k
P_updated = I_KH @ P_minus @ I_KH.T + K_k @ R_k @ K_k.T
```

Symmetrise covariance after update and check positive semidefiniteness.

### 5.3 Q and R calibration

Do **not** copy arbitrary values from a tutorial.

-   `R`: estimate from healthy training residuals, sensor
    repeatability/noise and/or robust residual covariance after
    condition compensation.
-   `Q`: tune using training trajectories and held-out validation
    engines to represent unmodelled degradation dynamics.
-   Use diagonal matrices only as an initial baseline; inspect
    cross-correlations before adding off-diagonal terms.
-   Use robust covariance estimates if outliers exist.
-   Keep Q/R fixed for the first controlled experiment; only introduce
    adaptive covariance if justified and separately evaluated.

Persist the calibrated Q/R and their estimation method.

### 5.4 Observability and diagnostics

Check:

-   numerical rank/conditioning of the local observability matrix or
    sensitivity/Jacobian stack;
-   correlation between state sensitivities;
-   innovation mean and variance;
-   normalised innovation squared (NIS);
-   covariance positivity and numerical stability;
-   state bounds and physically plausible progression;
-   performance across different flight classes and degradation modes.

If two component states cannot be separated from the measured channels,
do not show separate component health values. Merge them into a combined
latent state.

### 5.5 EKF outputs

For every prediction point, store:

-   state estimate `x_hat`;
-   covariance `P`;
-   innovation vector;
-   innovation covariance;
-   NIS/diagnostic flags;
-   physics-predicted sensors;
-   actual measured sensors;
-   state timestamp/flight/cycle.

The EKF does **not** directly produce RUL. It provides estimated
condition/state information to the downstream RUL model.

------------------------------------------------------------------------

## 6. Causal feature construction

All features at time `k` must use only current and past information
available at `k`.

### 6.1 Feature blocks

1.  **Current measured sensors:** selected `X_s`.
2.  **Condition variables:** verified `W`.
3.  **Healthy-reference residuals:** `X_s - z_healthy(W)`.
4.  **Physics residuals/innovations:** `X_s - h(W, x_hat_prior)`.
5.  **EKF state estimates:** `x_hat`.
6.  **EKF uncertainty:** state standard deviations from `diag(P)`.
7.  **Temporal features:** trailing-window mean, standard deviation,
    min/max, slope and last value.
8.  **Operating exposure:** cycle/flight count and elapsed exposure,
    only if available at prediction time.
9.  **Optional `X_v`:** include only if provenance confirms it is an
    available inference-time signal and it does not encode target/hidden
    simulator information.

### 6.2 Windowing

For a window length `N` at the selected prediction cadence:

``` text
rolling_mean_k = mean(x[k-N+1 : k])
rolling_std_k  = std(x[k-N+1 : k])
rolling_slope_k = causal_linear_slope(x[k-N+1 : k])
```

Handle trajectory starts with explicit minimum-history rules. Do not pad
using future values.

### 6.3 Feature leakage audit

Explicitly exclude:

-   `Y` / RUL;
-   hidden degradation group `T`;
-   future rows or future flight aggregates;
-   test-engine statistics used during training;
-   raw unit ID as a predictive feature unless a justified deployment
    use is proven (default: exclude);
-   health-state flag `hs` from online model inputs if unavailable at
    deployment.

`unit_id`, `cycle`, `hs` can still be used for grouping, ordering,
training-only healthy-row selection or audit where appropriate. Their
use must be explicit.

Persist the ordered feature schema and verify it at inference.

------------------------------------------------------------------------

## 7. Four base regressors

Train all four models on the **same engine-level split and same feature
schema**.

### 7.1 Random Forest Regressor

Purpose: bagged tree baseline, nonlinear interactions, variance
reduction.

Starting configuration (tune on validation only):

``` python
RandomForestRegressor(
    n_estimators=400,
    max_features=0.7,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
```

This is a starting search point, not a calibrated optimum.

### 7.2 XGBoost Regressor

Purpose: gradient-boosted trees with regularised objective.

Starting configuration:

``` python
XGBRegressor(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42
)
```

Use early stopping on the validation set. Tune depth, learning rate,
estimators, row/column subsampling and regularisation.

### 7.3 LightGBM Regressor

Purpose: efficient histogram-based gradient boosting.

Starting configuration:

``` python
LGBMRegressor(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    random_state=42,
    verbosity=-1
)
```

Tune leaves, depth, minimum child samples, learning rate and
regularisation. Verify the installed LightGBM version and subsampling
semantics.

### 7.4 CatBoost Regressor

Purpose: symmetric-tree boosting and a different inductive bias.

Starting configuration:

``` python
CatBoostRegressor(
    iterations=800,
    depth=7,
    learning_rate=0.03,
    loss_function="RMSE",
    l2_leaf_reg=5.0,
    random_seed=42,
    verbose=False
)
```

Tune depth, iterations, learning rate and regularisation on validation
engines.

### 7.5 Model selection

-   Tune each model on validation engines only.
-   Keep a fixed training/validation/test split across all experiments.
-   Record random seeds, library versions and hardware.
-   Compare individual models before stacking.
-   Do not assume the ensemble will outperform the best individual
    model.
-   Retain the existing raw-sensor Random Forest result as a historical
    baseline, but re-evaluate it under the leakage-safe split before
    comparing it to new results.

------------------------------------------------------------------------

## 8. Stacking with Ridge Regression

### 8.1 Why stacking

The four base models may make different errors. Ridge Regression learns
a regularised linear combination of their predictions.

For base predictions `p1, p2, p3, p4`:

``` text
RUL_hat = b + w1*p1 + w2*p2 + w3*p3 + w4*p4
```

The Ridge objective is:

``` text
min_w ||y - X_meta*w||^2 + alpha * ||w||^2
```

Weights are not automatically percentages of model trust. Correlated
base predictions and scaling affect coefficient interpretation.

### 8.2 Correct out-of-fold procedure

1.  Split the training trajectories into grouped folds by engine.
2.  For each fold, train each base model on the other folds.
3.  Predict the held-out fold.
4.  Repeat so every training row receives a prediction from a base model
    that did not train on that row's engine.
5.  Build the meta-training matrix from these four out-of-fold
    prediction columns.
6.  Fit the meta-feature scaler on the meta-training matrix only, if
    scaling is used.
7.  Fit Ridge on the scaled out-of-fold predictions and true training
    RUL.
8.  Refit each base model on the full training partition.
9.  For validation/test/inference, obtain predictions from those
    refitted base models and pass them through the saved scaler and
    Ridge model.

For time-ordered validation within engines, use a split strategy that
respects chronology as well as engine grouping. Do not use ordinary
shuffled K-fold across correlated samples.

### 8.3 Ridge diagnostics

Save and inspect:

-   alpha selected on validation data;
-   intercept;
-   coefficients;
-   base-model prediction correlations;
-   meta-model residuals.

Do not force coefficients to sum to one unless a non-negative convex
combination is an explicit design choice and is trained accordingly.

------------------------------------------------------------------------

## 9. Uncertainty and RUL safety logic

Ridge stacking gives a point estimate, not a calibrated probability
distribution.

For the first release:

-   expose EKF state covariance as state-estimation uncertainty;
-   calculate empirical RUL error by validation engine and
    operating/degradation group;
-   optionally use validation residual quantiles for a clearly labelled
    empirical prediction interval;
-   calibrate interval coverage on a separate calibration split if
    sufficient data is available.

Do not present EKF covariance as RUL prediction uncertainty. They are
different quantities.

Apply output safeguards:

-   RUL cannot be negative in the user-facing output; preserve raw
    prediction separately for diagnostics.
-   If predictions are implausible or out of training support, return a
    warning.
-   If sensor inputs are missing or out of range, return
    degraded/unavailable status rather than a fabricated prediction.
-   Do not make safety-critical aircraft maintenance decisions from this
    research prototype.

------------------------------------------------------------------------

## 10. Evaluation protocol

### 10.1 Metrics

Report:

**RMSE in cycles**

``` text
RMSE = sqrt(mean((RUL_true - RUL_pred)^2))
```

**MAE in cycles**

``` text
MAE = mean(abs(RUL_true - RUL_pred))
```

**R²**

Report as goodness of fit, not "accuracy percentage".

**NASA scoring function**

Implement the exact scoring convention from the chosen benchmark/paper.
Document the asymmetric early/late prediction penalty and units. Do not
compare scores from different conventions.

### 10.2 Required experiments

  -----------------------------------------------------------------------
  ID                      Experiment              Purpose
  ----------------------- ----------------------- -----------------------
  E0                      Raw-sensor Random       Existing baseline
                          Forest                  

  E1                      Raw-sensor four-model   Ensemble benefit
                          stack                   

  E2                      Healthy residual        Condition compensation
                          features + stack        benefit

  E3                      Physics residual/state  Physics/EKF feature
                          features + RF           benefit

  E4                      Physics + EKF +         Full proposed pipeline
                          four-model stack        

  E5                      Full pipeline without   EKF ablation
                          EKF features            

  E6                      Full pipeline without   Physics feature
                          physics residuals       ablation

  E7                      Full pipeline without   Temporal feature
                          temporal features       ablation
  -----------------------------------------------------------------------

All experiments must use the same engine-level evaluation protocol.

### 10.3 Report by subset and operating/degradation group

Do not combine all N-CMAPSS subsets into one score without reporting
per-subset results. Report per-engine and aggregate results. Where
possible, stratify by flight class and degradation mode.

Report mean and spread across repeated engine-level splits where
feasible. Keep the official test set untouched until final model
selection.

### 10.4 Acceptance gates

The system is not considered validated until:

-   HDF5 schema and units are verified;
-   no train/test engine overlap exists;
-   feature leakage audit passes;
-   physics residuals are evaluated on held-out engines;
-   EKF numerical/observability diagnostics pass;
-   each base model is evaluated independently;
-   stacking uses out-of-fold predictions;
-   full pipeline is compared with all ablations;
-   frontend receives real inference results from the backend;
-   API reconnect and stale-data behaviour are tested.

------------------------------------------------------------------------

## 11. Explainability

### 11.1 SHAP

Use SHAP for per-prediction and global feature explanations for tree
models.

Show:

-   current prediction's strongest positive/negative feature
    contributions;
-   global feature importance;
-   feature values alongside contributions;
-   explanation timestamp and model version.

Do not interpret a SHAP value as proof of physical causation. It
explains the fitted model's prediction under the SHAP assumptions.

### 11.2 Ridge model contributions

Expose the four base predictions and each linear contribution:

``` text
contribution_i = coefficient_i * scaled_base_prediction_i
```

Include intercept and scaler details in internal diagnostics. Do not
label coefficients as reliability percentages.

### 11.3 Physics/EKF diagnostics

Display:

-   measured sensor;
-   physics-predicted sensor;
-   residual/innovation;
-   estimated state;
-   state uncertainty;
-   NIS/diagnostic flag.

This makes the digital twin explain what the estimator observed, not
just the final RUL.

------------------------------------------------------------------------

## 12. Backend and persistent frontend connection

**This section is mandatory. The existing 3D model and frontend already
exist. Do not rebuild them. Keep the frontend connected to the Python
backend throughout development and deployment.**

### 12.1 Backend stack

Recommended:

-   Python 3.11 (or a version compatible with selected ML libraries);
-   FastAPI;
-   Uvicorn;
-   Pydantic;
-   NumPy/SciPy;
-   h5py;
-   scikit-learn;
-   xgboost;
-   lightgbm;
-   catboost;
-   shap;
-   joblib.

Pin exact package versions in `requirements.txt` or `pyproject.toml`
after a clean environment test.

### 12.2 Required API endpoints

  ----------------------------------------------------------------------------------------
  Endpoint                                 Method                  Purpose
  ---------------------------------------- ----------------------- -----------------------
  `/health`                                GET                     Backend liveness

  `/ready`                                 GET                     Model artifacts loaded
                                                                   and inference ready

  `/api/v1/engine/{unit_id}/latest`        GET                     Latest engine
                                                                   prediction

  `/api/v1/engine/{unit_id}/predict`       POST                    Submit current
                                                                   sensor/condition
                                                                   payload

  `/api/v1/engine/{unit_id}/history`       GET                     Historical RUL and
                                                                   health estimates

  `/api/v1/engine/{unit_id}/explanation`   GET                     SHAP and
                                                                   model-contribution data

  `/api/v1/engine/{unit_id}/telemetry`     GET                     Latest
                                                                   sensor/physics/EKF
                                                                   telemetry

  `/api/v1/models/version`                 GET                     Active model and
                                                                   dataset version
  ----------------------------------------------------------------------------------------

Use WebSocket or Server-Sent Events if the frontend needs push updates.
Otherwise use controlled polling. Do not open a new connection on every
UI render.

### 12.3 Prediction payload

Use a versioned schema. Example structure (values are placeholders, not
actual sensor data):

``` json
{
  "schema_version": "1.0",
  "dataset_id": "DS02",
  "unit_id": "example-unit",
  "cycle": 120,
  "timestamp": "2026-01-01T00:00:00Z",
  "operating_conditions": {
    "alt": 0.0,
    "Mach": 0.0,
    "TRA": 0.0,
    "T2": 0.0
  },
  "sensors": {
    "T24": 0.0,
    "T30": 0.0,
    "T48": 0.0,
    "T50": 0.0,
    "P2": 0.0,
    "P24": 0.0,
    "Ps30": 0.0,
    "Wf": 0.0,
    "Nf": 0.0,
    "Nc": 0.0
  }
}
```

Only include fields that are verified in the selected dataset schema.
For a real deployment, do not send synthetic placeholder values.

### 12.4 Prediction response

``` json
{
  "schema_version": "1.0",
  "unit_id": "example-unit",
  "prediction_timestamp": "2026-01-01T00:00:01Z",
  "rul_cycles": null,
  "health_state": {},
  "state_uncertainty": {},
  "innovation_summary": {},
  "base_predictions": {},
  "ridge_contributions": {},
  "top_features": [],
  "status": "unavailable",
  "model_version": "model-version",
  "dataset_id": "DS02"
}
```

`null` is intentional in this example. Replace it only with a real model
prediction. Never send a fake RUL as if it were an inference result.

### 12.5 Keep the frontend connected

Implement all of the following:

1.  Store backend URL in frontend environment configuration, not
    scattered hard-coded strings.
2.  Create one reusable API client/service.
3.  Keep the connection alive using WebSocket/SSE or a single controlled
    polling loop.
4.  Reconnect automatically after transient disconnects using
    exponential backoff and jitter.
5.  Send heartbeat/ping where the transport supports it.
6.  Show a visible connection indicator: `Connected`, `Reconnecting`,
    `Disconnected`, `Stale data`.
7.  Show the timestamp of the last successful prediction.
8.  Do not replace failed API responses with hard-coded mock RUL values.
9.  If backend is unavailable, retain the last confirmed result but mark
    it stale; do not imply it is live.
10. Validate payload schema and model version before updating the 3D
    engine.
11. Avoid duplicate requests caused by component re-rendering.
12. Clean up WebSocket/SSE/polling subscriptions when the relevant view
    is unmounted.
13. Add CORS configuration for the actual frontend origin.
14. Use HTTPS/WSS in deployment.
15. Add structured backend logs for request ID, unit ID, inference
    duration, model version and errors.

### 12.6 Frontend update contract

The existing 3D engine should consume a stable response object, not
internal Python objects.

Map:

-   `rul_cycles` -\> RUL display;
-   `health_state` -\> component/latent health visualisation;
-   `state_uncertainty` -\> uncertainty indicator;
-   `innovation_summary` -\> sensor anomaly/residual panel;
-   `top_features` -\> SHAP panel;
-   `ridge_contributions` -\> model contribution chart;
-   `status` -\> warning/health state;
-   `prediction_timestamp` -\> last update indicator.

The 3D model must not infer or invent health values from animation. Its
visual state must be driven by backend outputs.

------------------------------------------------------------------------

## 13. Repository structure

``` text
project/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── health.py
│   │   │   ├── predict.py
│   │   │   ├── telemetry.py
│   │   │   └── explain.py
│   │   ├── schemas/
│   │   │   ├── request.py
│   │   │   └── response.py
│   │   ├── services/
│   │   │   ├── inference_service.py
│   │   │   └── connection_state.py
│   │   └── config.py
│   ├── requirements.txt
│   └── tests/
├── ml/
│   ├── data_loader.py
│   ├── schema_validator.py
│   ├── preprocessing.py
│   ├── trajectory_builder.py
│   ├── healthy_baseline.py
│   ├── engine_physics.py
│   ├── physics_calibration.py
│   ├── ekf.py
│   ├── feature_builder.py
│   ├── train_base_models.py
│   ├── train_stacker.py
│   ├── evaluate.py
│   ├── explainability.py
│   └── inference.py
├── configs/
│   ├── dataset.yaml
│   ├── sensor_map.yaml
│   ├── physics.yaml
│   ├── ekf.yaml
│   └── model.yaml
├── artifacts/
│   ├── preprocessing/
│   ├── physics/
│   ├── ekf/
│   ├── base_models/
│   ├── stacker/
│   └── metadata/
├── frontend/                 # Existing frontend; retain it
├── reports/
│   ├── calibration/
│   ├── evaluation/
│   └── ablations/
├── data/                     # Do not commit large NASA files
├── .env.example
├── .gitignore
└── README.md
```

### 13.1 Artifact versioning

Every trained model bundle must include:

-   dataset subset and file fingerprint;
-   feature schema and ordered feature names;
-   sensor mapping and units;
-   train/validation/test engine IDs;
-   preprocessing/scaler objects;
-   healthy baseline parameters;
-   physics calibration parameters;
-   EKF Q/R and initial state/covariance;
-   base-model artifacts;
-   Ridge scaler and meta-model;
-   SHAP/explanation configuration;
-   package versions;
-   training code commit/hash;
-   metrics and evaluation report.

The backend must load one internally consistent artifact bundle. Do not
mix a new feature builder with old model weights.

------------------------------------------------------------------------

## 14. Configuration template

Example `configs/dataset.yaml`:

``` yaml
dataset:
  name: N-CMAPSS
  subset: ${NC_MAPSS_DATASET_ID}
  file_path: ${NC_MAPSS_H5_PATH}
  metadata_path: ${NC_MAPSS_METADATA_PATH}
  use_official_split: true
  composite_unit_key: [dataset_id, unit_id]
  inference_cadence: flight_end
  target_group: Y
  measured_sensor_group: X_s
  operating_condition_group: W
  auxiliary_group: A
  hidden_degradation_group: T
  virtual_sensor_group: X_v
  use_hidden_degradation_as_feature: false
```

Example `configs/physics.yaml`:

``` yaml
physics:
  model_type: reduced_order_gas_path
  internal_units: SI
  healthy_reference:
    use_training_healthy_rows: true
    health_flag_source: A
    health_flag_name: hs
  state:
    initial_dimension: 4
    state_names:
      - d_LPC_efficiency
      - d_HPC_efficiency
      - d_turbine_efficiency
      - d_flow_capacity
    constrain_state: true
  calibration:
    fit_on_training_engines_only: true
    validate_on_held_out_engines: true
    use_T_group_for_online_features: false
```

The state dimension and names are provisional. The calibration report
must justify the final state vector. If the selected subset cannot
support these four states, reduce the dimension.

Example `configs/ekf.yaml`:

``` yaml
ekf:
  type: extended_kalman_filter
  initial_state: calibrated_from_training
  initial_covariance: calibrated_from_training
  process_noise_Q: calibrated_from_training
  measurement_noise_R: calibrated_from_healthy_residuals
  covariance_update: joseph
  use_future_measurements: false
  prediction_cadence: flight_end
  diagnostics:
    compute_innovation: true
    compute_nis: true
    check_covariance_psd: true
```

Do not put arbitrary numeric Q/R values in source code. Save fitted
matrices in the versioned artifact bundle.

------------------------------------------------------------------------

## 15. Testing checklist

### Dataset and preprocessing

-   [ ] HDF5 groups and shapes inspected.
-   [ ] Sensor names mapped from official metadata.
-   [ ] Units verified.
-   [ ] Row alignment verified.
-   [ ] Unit IDs made unique across subsets.
-   [ ] Telemetry rows distinguished from flight/cycle count.
-   [ ] RUL target alignment verified.
-   [ ] No train/test engine overlap.
-   [ ] No future information in features.
-   [ ] Hidden `T` group excluded from online features.

### Physics and EKF

-   [ ] SI conversions unit-tested.
-   [ ] Healthy baseline trained on training healthy rows only.
-   [ ] Physics observation model fitted on training engines only.
-   [ ] Sensor prediction residuals evaluated on held-out engines.
-   [ ] Component-state identifiability checked.
-   [ ] Jacobians numerically checked against finite differences.
-   [ ] Q/R estimated using documented methods.
-   [ ] EKF covariance remains numerically valid.
-   [ ] Innovations and NIS inspected.
-   [ ] State estimates compared with offline reference information
    where available.

### Machine learning

-   [ ] Raw-sensor RF baseline re-evaluated.
-   [ ] Four base models trained on identical splits.
-   [ ] Hyperparameters selected using validation only.
-   [ ] OOF stacking predictions generated by engine-grouped folds.
-   [ ] Ridge meta-model trained only on OOF predictions.
-   [ ] Test set used once for final evaluation.
-   [ ] RMSE, MAE, R² and documented NASA score reported.
-   [ ] Ablations completed.

### Explainability

-   [ ] SHAP per-prediction explanations available.
-   [ ] Global feature importance available.
-   [ ] Ridge contributions available.
-   [ ] Physics residuals displayed.
-   [ ] EKF uncertainty labelled correctly.
-   [ ] No causal claim made from SHAP alone.

### Backend and frontend

-   [ ] `/health` and `/ready` work.
-   [ ] Frontend API URL configured through environment.
-   [ ] Existing 3D frontend connected to real backend outputs.
-   [ ] Reconnection tested after backend restart.
-   [ ] Stale data visibly labelled.
-   [ ] No silent mock-data fallback.
-   [ ] Schema/version mismatch handled.
-   [ ] Inference latency logged.
-   [ ] Backend errors surfaced in frontend.
-   [ ] End-to-end prediction tested.

------------------------------------------------------------------------

## 16. 24-hour implementation priority

The full specification is a research-grade target. For a 24-hour
prototype, implement in this order:

1.  Inspect one N-CMAPSS subset and verify the HDF5 schema.
2.  Reproduce the existing Random Forest baseline with engine-wise
    separation.
3.  Implement the healthy-condition sensor baseline and residual
    generation.
4.  Implement a low-dimensional calibrated observation model.
5.  Implement and test the EKF on a small number of trajectories.
6.  Build causal features from sensors, physics residuals and EKF
    states.
7.  Train the four tree models.
8.  Generate out-of-fold predictions and train Ridge.
9.  Compare baseline, ensemble and full pipeline.
10. Add SHAP for the final prediction.
11. Connect the model service to the existing frontend and 3D engine.
12. Test reconnect, stale data and real prediction updates.

**If full thermodynamic calibration is not completed in 24 hours, label
the result a reduced-order physics-informed prototype. Do not claim a
fully calibrated component-level simulator.**

------------------------------------------------------------------------

## 17. Definition of done

The pipeline is ready for demonstration when:

-   A selected N-CMAPSS HDF5 subset loads correctly.
-   A new engine observation can pass through preprocessing, physics
    model and EKF.
-   The EKF returns a valid state estimate and uncertainty.
-   The four regressors return RUL predictions.
-   Ridge returns the final RUL estimate.
-   SHAP and model-contribution explanations are generated.
-   The backend returns a schema-valid response.
-   The existing frontend remains connected and updates the 3D engine
    from real backend outputs.
-   The evaluation report clearly distinguishes baseline, ensemble and
    physics/EKF results.
-   All limitations and uncompleted calibration steps are disclosed.

**Final principle:** The physics model estimates what the engine should
do, the EKF estimates the engine's hidden condition from observed
measurements, the ML ensemble estimates remaining useful life, and the
existing digital twin communicates the result. Each layer must be
independently tested and connected through a versioned, persistent
backend contract.
