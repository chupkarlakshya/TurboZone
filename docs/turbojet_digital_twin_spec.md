# Physics-Informed Digital Twin — Four-Stage Turbojet Health Monitoring
### Master Build Specification (for an agentic IDE — e.g. Antigravity, Claude Code, Cursor)

**Source:** Official Problem Statement, IIT Indore x Hindustan Aeronautics Limited (HAL)
**Track:** Physics-Informed Digital Twin for Real-Time Four-Stage Turbojet Health Monitoring
*(Note: a separate, unrelated HAL/IIT Indore track — "Hybrid-Electric Propulsion Optimization for a Fixed-Wing UAV" — exists in the same document bundle but is out of scope for this build.)*

---

## 1. Project Definition (verbatim scope, paraphrased for clarity)

Build a **Physics-Informed Digital Twin** that reconstructs the operational and health state of a **single-spool, four-stage turbojet engine** using a **fixed, limited set of sensor measurements** (defined in §2 — no vibration, no oil debris, no N1/N2, no direct efficiency readouts). The system must:

- Estimate **hidden** subsystem health indicators that are not directly measured.
- Predict engine performance (thrust, fuel efficiency, degradation trajectory).
- Maintain a **continuously updated virtual representation** across the engine's operational life.
- Combine **engineering/physics principles** with **data-driven surrogate modeling** — not a black-box model, and not a full CFD/high-fidelity sim (explicitly ruled out as too slow for real-time use).
- Be **interpretable** and **computationally efficient** — these are graded criteria, not nice-to-haves.

**"Four-stage" clarification (important, physics-informed note for the agent):** the provided dataset does *not* give per-stage sensor data — it gives lumped station measurements (compressor exit, combustor exit/turbine inlet, turbine exit). The four-stage architecture is *physical context* for how the engine degrades internally (e.g., which compressor/turbine stage might be fouling), but the model must infer stage-level or component-level health from lumped, station-level pressure/temperature data. Do not assume access to per-stage instrumentation that isn't in §2.

---

## 2. Data Schema (fixed — this is the actual sensor set, confirmed against the official spec)

Synthetic but physics-based dataset: multiple virtual engines, varying flight conditions, progressive degradation across cycles (run-to-failure style, like C-MAPSS but single-spool).

| Parameter | Symbol | Unit |
|---|---|---|
| Engine ID | — | – |
| Cycle | c | – |
| Altitude | h | m |
| Mach Number | M | – |
| Ambient Temperature | T_amb | K |
| Ambient Pressure | P_amb | Pa |
| Shaft Speed | RPM | rev/min |
| Fuel Flow Rate | ṁ_f | kg/s |
| Compressor Exit Pressure | P2 | Pa |
| Compressor Exit Temperature | T2 | K |
| Combustor Exit Pressure (= Turbine Inlet Pressure) | P3 | Pa |
| Turbine Inlet Temperature | T3 | K |
| Turbine Exit Pressure | P4 | Pa |
| Turbine Exit Temperature | T4 | K |

**Do not** build pipelines that assume N1/N2 (twin-spool), vibration, oil debris, or nozzle-exit (P5/T5) data — none of that exists in this dataset. Any physics formula requiring air mass flow (ṁ_a) must either estimate it from RPM/altitude/Mach via a compressor-map correlation, or be reformulated to avoid needing it (see §4).

---

## 3. Physics Layer (lumped-parameter, single-spool, single-stream)

Use γ = 1.4 for the inlet/ambient air stream and γ ≈ 1.33, c_p ≈ 1150 J/kg·K for hot gas past the combustor (standard engineering approximation; document this assumption).

**3.1 Ambient/inlet total conditions from flight state**
```
T_t,amb = T_amb * (1 + (γ-1)/2 * M²)
P_t,amb = P_amb * (1 + (γ-1)/2 * M²)^(γ/(γ-1))
```

**3.2 Corrected parameters (normalize across altitude/Mach so degradation isn't confused with operating-point changes)**
```
θ = T_amb / 288.15
δ = P_amb / 101325
N_corrected  = RPM / sqrt(θ)
Wf_corrected = ṁ_f / (δ * sqrt(θ))
```
This step is the single most important physics-informed preprocessing move: without it, the model will confuse "engine is at high altitude/throttle" with "engine is degrading."

**3.3 Compressor module**
```
π_c   = P2 / P_t,amb                                  (pressure ratio)
T2s   = T_t,amb * π_c^((γ-1)/γ)                        (ideal/isentropic exit temp)
η_c   = (T2s - T_t,amb) / (T2 - T_t,amb)                (isentropic efficiency — KEY health signal)
```

**3.4 Combustor module**
```
ΔP_comb = (P2 - P3) / P2                                (pressure loss fraction; healthy ≈ 2-6%)
```
Energy balance (requires an estimate of air mass flow ṁ_a — see note below):
```
η_b = [(ṁ_a + ṁ_f) * c_p * (T3 - T2)] / (ṁ_f * LHV)     where LHV ≈ 43,000 kJ/kg
```
*ṁ_a estimation note:* if not directly derivable from this dataset, either (a) approximate via a simple corrected-flow-vs-corrected-speed compressor map fit trained on early/healthy cycles per engine, or (b) skip η_b entirely and use the pressure-loss-only + T3 anomaly signal as the combustor health proxy — safer given the "physics consistency" grading criterion rewards not overclaiming unmeasurable quantities.

**3.5 Turbine module**
```
π_t   = P3 / P4
T4s   = T3 * π_t^(-(γ-1)/γ)
η_t   = (T3 - T4) / (T3 - T4s)                          (isentropic efficiency — KEY health signal)
```

**3.6 Thrust estimate (no nozzle station in dataset → approximate)**
Physically-grounded ideal expansion estimate:
```
V_jet ≈ sqrt( 2*cp*T4*(1 - (P_amb/P4)^((γ-1)/γ)) )
F ≈ ṁ_gas * (V_jet - V_amb),   ṁ_gas ≈ ṁ_a + ṁ_f,   V_amb = M*sqrt(γ*R*T_amb)
```
Report this as an **estimate with explicit assumptions**, not a measured value — this is exactly the kind of "physics consistency" the grading rewards being honest about.

**3.7 Health indices per module (0–1 normalized against each engine's own early-life/healthy baseline)**
```
HI_comp  = η_c / η_c,baseline
HI_comb  = combustor proxy (η_b or pressure-loss-based) / baseline
HI_turb  = η_t / η_t,baseline
```
Baseline = rolling average of the first N cycles (e.g., first 10–15% of that engine's life), computed **per engine**, not globally — this matters because different virtual engines may have different manufacturing tolerances/starting points.

**3.8 Overall Health Index**
Two options, use both and let the model/report justify the choice:
- Weighted composite: `OHI = w1*HI_comp + w2*HI_comb + w3*HI_turb` (weights justified by sensitivity/failure-mode dominance, document the reasoning — turbine is usually weighted highest).
- Data-fit form (from published aero-engine DT literature, power-law degradation): `HI(i) = a * RUL(i)^b + c`, fit per engine by least squares on your own RUL predictions. Useful for producing a smooth trend curve for the dashboard even when raw efficiency estimates are noisy.

---

## 4. Feature Engineering for the Surrogate Model

Per-cycle feature vector (in addition to raw sensors), all physics-derived and computable directly from §2/§3:
- Corrected RPM, corrected fuel flow
- π_c, π_t, η_c, η_t, ΔP_comb
- T2−T_t,amb (compressor work), T3−T4 (turbine work extracted)
- Rolling deltas/slopes of η_c, η_t, T4 over a short window (captures degradation *rate*, not just level)
- Cycle number itself (monotonic degradation proxy, used carefully to avoid the model just memorizing cycle→RUL without learning physics)

Standardize (Z-score) all inputs before feeding to the model; fit scalers on training engines only.

---

## 5. Surrogate Model Architecture

Given the grading explicitly weights **Computational Efficiency (10%)** and **Physics Consistency (15%)** alongside accuracy, avoid over-engineering (skip a full custom Transformer unless justified). Two reasonable tiers:

**Baseline (recommended starting point):** GRU or small LSTM over a sliding window of the engineered feature vector (§4), predicting RUL/HI directly. Cheap, fast to train, easy to explain in the report.

**Stretch goal (if time allows, referencing the TMR-style improvements from literature):** lightweight Transformer encoder with learnable positional encoding + one or two self-attention layers + small MLP head (skip the decoder entirely, matches the "improved Transformer" pattern from the digital-twin literature reviewed earlier) — but only if the baseline's computational-efficiency numbers are logged first, so you can show the trade-off honestly in the report (this scores well on both "innovation" narrative *and* the efficiency criterion, since you're not hiding the cost).

**Physics-informed loss (this is what actually earns "Physics Consistency" points, not just using physics-derived features):**
```
Loss_total = Loss_data (MSE on RUL/HI) + λ * Loss_physics
```
where `Loss_physics` penalizes predictions that violate known constraints, e.g.:
- Monotonicity: HI should not increase over a cycle window (degradation is one-directional, barring maintenance events — check if the dataset includes any).
- Consistency: predicted η_c, η_t implied by the model's internal state should stay within physically plausible bounds (e.g., η ∈ (0,1)).

---

## 6. Performance Prediction Outputs

- **Thrust** — via §3.6 estimate, trended over cycles.
- **Fuel efficiency metrics** — corrected SFC proxy: `Wf_corrected / F_estimate`, trended over cycles.
- **Degradation trajectory** — HI(cycle) curve per engine, both actual-so-far and forward-extrapolated.
- **RUL** — cycles remaining until HI crosses a failure threshold (piecewise-linear degradation-labeling approach, consistent with standard C-MAPSS-style RUL labeling: RUL decreases linearly to zero from a "healthy plateau" cutoff).

---

## 7. Uncertainty Quantification (graded implicitly under accuracy/generalization, and explicitly required in Task 6)

Pick ONE primary method and implement it properly rather than bolting on several superficially:
- **MC Dropout** (cheapest to add to a GRU/Transformer — keep dropout active at inference, run N forward passes, report mean ± std) — best fit given the efficiency constraint.
- **Quantile regression head** (predict 10th/50th/90th percentile RUL directly) — good alternative if you want calibrated intervals without extra inference cost.
- Report confidence as a 90% prediction interval on RUL/HI, and as a scalar "confidence score" for the dashboard: `Confidence(%) = 100 * (1 - σ_pred/μ_pred)`, clipped to [0,100].

---

## 8. Digital Twin Dashboard (exact required panels, per official spec)

1. Engine operating conditions (Altitude, Mach, T_amb, P_amb, RPM, Fuel Flow — live/selected-cycle view)
2. Compressor health (HI_comp trend + current value)
3. Combustor health (HI_comb trend + current value)
4. Turbine health (HI_turb trend + current value)
5. Overall health index (OHI, with normal/warning/alarm/emergency color banding)
6. Predicted thrust (trend)
7. Degradation trends (HI vs cycle, RUL vs cycle)
8. Prediction confidence (interval or % score, per §7)

Keep this **honest and interpretable** over flashy — the grading criterion is literally "Dashboard **and Interpretability**," not visual spectacle. A clean line-chart dashboard with clear health banding beats a 3D animated engine with no uncertainty shown. (3D/CAD visualization was part of the *other*, unrelated hackathon doc — not required here; skip it unless you have time budget left after the core deliverables.)

---

## 9. Deliverables Checklist (official)

- [ ] **Technical Report** — methodology, feature engineering strategy, physics integration approach, model architecture, validation methodology
- [ ] **Source Code** — complete implementation
- [ ] **Digital Twin Dashboard** — §8 panels
- [ ] **Presentation** — engineering rationale, surrogate modeling strategy, health estimation methodology, key results/insights

---

## 10. Evaluation Criteria — build priority should match these weights

| Criterion | Weight | Build implication |
|---|---|---|
| Health Estimation Accuracy | 30% | Get η_c/η_t/HI baselines and RUL fitting right first — this is the largest single chunk. |
| Surrogate Model Performance | 20% | Benchmark against a naive baseline (linear regression on raw sensors) to prove the surrogate adds value. |
| Physics Consistency | 15% | Use §3 formulas honestly, document every assumption (γ, c_p, ṁ_a estimation), don't fabricate unmeasurable quantities. |
| Generalization Capability | 15% | Train/test split by **engine ID**, not by cycle — test on entirely unseen engines/flight-condition combinations. |
| Computational Efficiency | 10% | Log training time and inference latency; justify model size choice explicitly in the report. |
| Dashboard & Interpretability | 10% | Clean, correct, all 8 required panels — not more, not fancier. |

---

## 11. Suggested Repo Structure (for the IDE agent to scaffold)

```
/data           - raw + processed dataset, per-engine train/test split
/physics        - §3 formulas as a standalone, testable module (no ML dependencies)
/features       - §4 feature engineering pipeline
/models         - baseline (GRU) and stretch (transformer) surrogate models
/uncertainty    - MC dropout / quantile regression wrapper
/dashboard      - visualization app (Streamlit/Plotly Dash recommended — fast to build, easy to demo)
/report          - technical report source
/notebooks      - exploratory analysis, baseline vs physics-informed comparisons
```

**Suggested stack:** Python, PyTorch (or scikit-learn for the baseline), Streamlit or Plotly Dash for the dashboard (far faster to stand up than a full React/Three.js stack for this scope, and the grading doesn't reward 3D visualization). Keep it lean — the efficiency and interpretability criteria actively punish over-engineering.

---

## 12. Open Assumptions to State Explicitly in the Report

- γ and c_p values used for each gas stream, and why.
- How ṁ_a (air mass flow) was estimated, since it's not in the raw dataset.
- How the healthy-baseline window was chosen per engine.
- How the RUL failure threshold / HI grading boundaries (normal/warning/alarm/emergency) were set.

These are exactly the kind of "assumptions must be clearly documented and justified" statements HAL/IIT Indore explicitly ask for — treat this section as a report checklist, not an afterthought.
