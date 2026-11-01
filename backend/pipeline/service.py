"""Turns pipeline outputs into the dashboard payload (Layer 5): display units, alerts, physics checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ncmapss as nc
from . import physics_monitors as pm

BAND_NAMES = ["Normal", "Warning", "Alarm", "Emergency"]
RUL_ALERTS = [(30, "warning"), (15, "alarm"), (5, "emergency")]
DISAGREE_CYCLES = 20.0
ALL_COMPONENTS = nc.COMPONENTS + pm.AUX_COMPONENTS      # EKF-tracked modules + physics-monitored combustor, nozzle
NAMES = {**nc.COMP_NAMES, **pm.AUX_NAMES}          # tree-only vs full-stack RUL gap that flags low confidence


def band_index(v: float, edges) -> int:
    return int(sum(v < e for e in edges))


def _num(v):
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    v = float(v)
    return v if np.isfinite(v) else None


def to_display(s: str, v: np.ndarray) -> np.ndarray:
    kind = nc.SENSOR_INFO[s][1]
    if kind == "T":
        return v * nc.DEG_R_TO_K
    if kind == "P":
        return v * nc.PSIA_TO_KPA
    if kind == "W":
        return v * nc.LBM_TO_KG
    return v


def display_columns(e: pd.DataFrame, targets: list[str]) -> dict:
    cols = {
        "cycle": e["cycle"], "fc": e.get("fc"), "duration_min": e["duration_s"] / 60.0,
        "alt_m": e["alt_mean"] * nc.FT_TO_M, "alt_max_m": e.get("alt_max", e["alt_mean"]) * nc.FT_TO_M,
        "mach": e["Mach_mean"], "tra": e["TRA_mean"], "t2_K": e["T2_mean"] * nc.DEG_R_TO_K,
    }
    for s in nc.XS_VARS:
        cols[f"x_{s}"] = to_display(s, e[f"m_{s}"].to_numpy())
        cols[f"ideal_{s}"] = to_display(s, e[f"ref_{s}"].to_numpy())
        cols[f"res_{s}"] = e[f"res_{s}"] * 100.0                       # percent vs healthy engine
        cols[f"nis_{s}"] = e[f"nis_{s}"]
        cols[f"gated_{s}"] = e[f"gated_{s}"]
    for v in nc.XV_DISPLAY:
        if f"v_{v}" in e:
            cols[f"v_{v}"] = to_display(v, e[f"v_{v}"].to_numpy())
    for p in nc.PROXIES:
        cols[f"p_{p}"] = e[f"p_{p}"]
    if "v_T40" in e:      # display-only turbine efficiency estimate (needs the virtual T40)
        t40, t50 = e["v_T40"].to_numpy(), e["m_T50"].to_numpy()
        pr = np.clip(e["m_P50"].to_numpy() / e["m_P40"].to_numpy(), 1e-6, None)
        cols["eta_turb_est"] = (t40 - t50) / np.where(np.abs(t40 * (1 - pr ** nc.K_HOT)) < 1e-9, np.nan,
                                                      t40 * (1 - pr ** nc.K_HOT))
    for t in nc.T_VARS:
        cols[f"kf_{t}"] = e[f"kf_{t}"]
        cols[f"kf_{t}_sigma"] = e[f"kf_{t}_sigma"]
    for c in nc.COMPONENTS:
        cols[f"kf_hi_{c}"] = e[f"kf_hi_{c}"]
        cols[f"kf_hi_{c}_sigma"] = e[f"kf_hi_{c}_sigma"]
    for c in pm.AUX_COMPONENTS:                                     # physics monitors (physics_monitors.py)
        for k in (f"hi_{c}", f"hi_{c}_sigma"):
            if k in e:
                cols[k] = e[k]
    for m in pm.LIMITS:
        for k in (f"hi_{m}", f"dev_{m}", f"dev_{m}_smooth"):
            if k in e:
                cols[k] = e[k]
    for k in ("phys_t40_K", "phys_npr"):
        if k in e:
            cols[k] = e[k]
    for t in targets:
        for suf in ("", "_trees_only", "_spread", "_raw"):
            cols[f"{t}{suf}"] = e[f"{t}{suf}"]
    for q in ("rul_q05", "rul_q95", "rul_spread", "nis_total"):
        if q in e:
            cols[q] = e[q]
    for c in e.columns:                                             # per-model predictions / Ridge contributions
        if c.startswith(("base_", "contrib_", "fc_", "crul_")):
            cols[c] = e[c]
    comps = [c for c in ALL_COMPONENTS if f"hi_{c}" in e]
    his = np.column_stack([e[f"hi_{c}"].to_numpy() for c in comps])
    cols["ohi"] = his.min(axis=1)
    cols["ohi_component"] = his.argmin(axis=1)
    for c in nc.COMPONENTS:                                            # ground truth, demo engines only
        if f"truth_hi_{c}" in e:
            cols[f"truth_hi_{c}"] = e[f"truth_hi_{c}"]
    for k in ("truth_rul", "truth_hs"):
        if k in e:
            cols[k] = e[k]
    return {k: [_num(x) for x in np.asarray(v)] for k, v in cols.items() if v is not None}


def derive_alerts(e: pd.DataFrame, edges) -> list[dict]:
    alerts = []
    for c in [c for c in ALL_COMPONENTS if f"hi_{c}" in e]:
        prev = 0
        for cyc, v in zip(e["cycle"], e[f"hi_{c}"]):
            b = band_index(v, edges)
            if b > prev:
                alerts.append({"cycle": int(cyc), "type": "health_band", "component": c,
                               "severity": BAND_NAMES[b].lower(),
                               "message": f"{NAMES[c]} health {v:.2f}: {BAND_NAMES[prev]} -> {BAND_NAMES[b]}"})
                prev = b
    done = set()
    for cyc, r in zip(e["cycle"], e["rul"]):
        for thr, sev in RUL_ALERTS:
            if r <= thr and thr not in done:
                done.add(thr)
                alerts.append({"cycle": int(cyc), "type": "rul", "severity": sev,
                               "message": f"Predicted RUL {r:.0f} cycles (<= {thr})"})
    for s in nc.XS_VARS:
        g = e[f"gated_{s}"].to_numpy(dtype=bool)
        run = np.convolve(g.astype(int), np.ones(5, dtype=int))[: len(g)]
        start = np.flatnonzero((run >= 3) & (np.concatenate([[0], run[:-1]]) < 3))
        for i in start:
            alerts.append({"cycle": int(e["cycle"].iloc[i]), "type": "sensor_health", "severity": "warning",
                           "message": f"{s} ({nc.SENSOR_INFO[s][0]}): repeated Kalman-filter innovation rejections"})
    gap = (e["rul"] - e["rul_trees_only"]).abs().to_numpy()
    for i in np.flatnonzero((gap > DISAGREE_CYCLES) & (np.concatenate([[0], gap[:-1]]) <= DISAGREE_CYCLES)):
        alerts.append({"cycle": int(e["cycle"].iloc[i]), "type": "model_disagreement", "severity": "warning",
                       "message": f"Tree-only and full-stack RUL differ by {gap[i]:.0f} cycles: low confidence"})
    return sorted(alerts, key=lambda a: (a["cycle"], a["type"]))


def constraint_checks(e: pd.DataFrame, targets) -> dict:
    hi_raw = np.column_stack([e[f"hi_{c}_raw"].to_numpy() for c in nc.COMPONENTS])
    his = np.column_stack([e[f"hi_{c}"].to_numpy() for c in nc.COMPONENTS])
    rise = (pd.DataFrame(his).rolling(5, min_periods=1).max().to_numpy() - pd.DataFrame(his).rolling(
        5, min_periods=1).min().to_numpy()) if len(his) else np.zeros((0, 5))
    increasing = np.diff(his, axis=0) > 0.05 if len(his) > 1 else np.zeros((0, 5), dtype=bool)
    eff = np.column_stack([e[f"kf_{t}"].to_numpy() for t in nc.T_VARS if t.endswith("eff_mod")])
    return {
        "health_raw_within_0_1": bool(((hi_raw > -0.05) & (hi_raw < 1.05)).all()),
        "monotonic_degradation": bool(not increasing.any()),
        "kf_efficiency_not_improving": bool((eff <= 0.005).all()),
        "rul_non_negative": bool((e["rul_raw"] >= -1).all()),
        "_max_5cycle_health_swing": float(rise.max()) if rise.size else 0.0,
    }


def status_from(band: int, rul: float) -> str:
    return "healthy" if band == 0 and rul > RUL_ALERTS[0][0] else "warning" if band <= 1 and rul > RUL_ALERTS[1][0]         else "degraded"


def triage_from_values(his: dict, rul: float, q05: float, reliability: dict | None, edges) -> dict:
    """Triage summary of one flight from component health values, RUL and the lower interval bound."""
    worst = min(his, key=his.get)
    band = band_index(his[worst], edges)
    loc, reliable = _group_label(worst, reliability)
    planning = max(float(q05), 0.0)
    rul_level = sum(planning <= thr for thr, _ in RUL_ALERTS)       # 0..3 from the planning RUL
    urgency = max(band, rul_level)                                  # health and RUL can disagree: take the worse
    return {"planning_rul": planning, "rul": float(rul), "band": band, "band_name": BAND_NAMES[band],
            "urgency": urgency, "urgency_name": BAND_NAMES[urgency],
            "urgency_driver": "health" if band >= rul_level else "rul",
            "status": status_from(band, rul), "worst_component": worst, "localisation": loc,
            "localisation_reliable": reliable, "ohi": float(his[worst])}


def status_fields(bundle, e: pd.DataFrame, edges) -> dict:
    """pipeline.md 1.1 / 12.4: status of the latest flight, model version, prediction timestamp."""
    import datetime

    last = e.iloc[-1]
    worst = min(float(last[f"hi_{c}"]) for c in ALL_COMPONENTS if f"hi_{c}" in e)
    band = band_index(worst, edges)
    rul = float(last["rul"])
    status = "healthy" if band == 0 and rul > RUL_ALERTS[0][0] else "warning" if band <= 1 and rul > RUL_ALERTS[1][0] \
        else "degraded"
    build = bundle.manifest.get("build") or {}
    return {"status": status, "model_version": f"{build.get('commit') or 'unknown'}@{build.get('trained_utc') or '?'}",
            "dataset_id": bundle.manifest.get("run"),
            "prediction_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


def _group_label(c: str, reliability: dict) -> tuple[str, bool]:
    """Localisation text for component c. Components the EKF cannot separate are reported as one group."""
    if c in pm.AUX_COMPONENTS:                       # physics monitor, measured directly: always attributable
        return NAMES[c], True
    r = (reliability or {}).get(c) or {}
    if r.get("status") == "ok" and r.get("identifiable", True):
        return nc.COMP_NAMES[c], True
    members = sorted({c, *r.get("merged_with", [])}, key=nc.COMPONENTS.index)
    return " / ".join(nc.COMP_NAMES[m] for m in members) + " (not separable by the EKF)", False


def advisory(bundle, e: pd.DataFrame, edges) -> dict:
    """What a maintenance engineer should take from the latest flight: planning RUL (lower interval bound),
    triage band, and a localisation that does not claim more than the EKF can identify."""
    last = e.iloc[-1]
    his = {c: float(last[f"hi_{c}"]) for c in ALL_COMPONENTS if f"hi_{c}" in e}
    q05 = float(last["rul_q05"]) if "rul_q05" in e else float(last["rul"])
    t = triage_from_values(his, last["rul"], q05, bundle.manifest.get("reliability"), edges)
    return {**t, "guidance": GUIDANCE_PREFIX + ("" if t["localisation_reliable"] else GUIDANCE_ISOLATION) + GUIDANCE_SUFFIX}


GUIDANCE_PREFIX = "Plan the shop visit from the planning RUL (5th-percentile bound), not the point estimate. "
GUIDANCE_ISOLATION = ("Component isolation is not reliable here: confirm by borescope / EGT margin / vibration before "
                      "acting on a single module. ")
GUIDANCE_SUFFIX = "Advisory decision support only; the approved maintenance programme governs."


def forecast_display(bundle, e: pd.DataFrame) -> list[dict]:
    """Component-health forecasts with a trust level. 'confident' only where the forecast beat both baselines on
    validation engines AND the component's EKF health is reliable; 'indicative' where useful but the component is
    not separable (shown as a group trend only); everything else is hidden."""
    fc = getattr(bundle, "forecast", None) or {}
    useful, rel, last = fc.get("useful") or {}, bundle.manifest.get("reliability") or {}, e.iloc[-1]
    out = []
    for c in nc.COMPONENTS:
        for h in fc.get("horizons", []):
            col = f"fc_{c}_h{h}"
            if col not in e or not (useful.get(c) or {}).get(str(h)):
                continue
            ok = (rel.get(c) or {}).get("status") == "ok"
            label, _ = _group_label(c, rel)
            out.append({"component": c, "label": label, "horizon": int(h), "trust": "confident" if ok else "indicative",
                        "health_now": _num(last[f"kf_hi_{c}"]), "health_forecast": _num(last[col]),
                        "lo": _num(last.get(col + "_lo")), "hi": _num(last.get(col + "_hi"))})
    return out


def engine_payload(bundle, e: pd.DataFrame, engine_id: str, source: str) -> dict:
    e = e.sort_values("cycle").reset_index(drop=True)
    edges = bundle.manifest.get("band_edges", [0.75, 0.55, 0.30])
    meta = {}
    for k in ("dataset", "unit_local"):
        if k in e:
            meta[k] = _num(e[k].iloc[0]) if k != "dataset" else str(e[k].iloc[0])
    return {
        "engine_id": engine_id, "source": source, "n_cycles": int(len(e)), **meta,
        "components": nc.COMPONENTS, "aux_components": pm.AUX_COMPONENTS, "component_names": NAMES,
        "monitors": {m: {"label": pm.MONITOR_LABELS[m], "direction": d, "tolerance": t, "failure_limit": l}
                     for m, (d, t, l) in pm.LIMITS.items()}, "sensors": nc.XS_VARS, "theta": nc.T_VARS,
        "sensor_info": {s: nc.SENSOR_INFO[s][0] for s in nc.XS_VARS}, "targets": bundle.targets,
        "band_edges": edges, "columns": display_columns(e, bundle.targets),
        "alerts": derive_alerts(e, edges), "advisory": advisory(bundle, e, edges),
        "forecast_display": forecast_display(bundle, e), "constraint_checks": constraint_checks(e, bundle.targets),
        "model": bundle.info(), **status_fields(bundle, e, edges),
        "assumptions": [
            "RUL = Ridge stack of Random Forest, XGBoost, LightGBM and CatBoost on physics/EKF features (pipeline.md).",
            "Component health = EKF parameter estimate mapped as 1 - damage / failure-level damage of that component; "
            "see model.reliability / model.observability for which components are identifiable.",
            "Combustor and nozzle health = physics monitors (spool and burner energy balance, nozzle continuity) "
            "against a healthy engine, with engineering limits; N-CMAPSS itself never degrades these parts.",
            "Ideal = healthy-condition baseline at the same flight condition; residual = measured / ideal - 1.",
            "RUL interval = empirical (validation residual quantiles), not a calibrated probability.",
            "Research prototype: not for maintenance decisions.",
        ],
    }


def parse_upload(df: pd.DataFrame) -> pd.DataFrame:
    """Accept raw 1 Hz flight data (W + X_s + unit + cycle [+ Fc]) or this pipeline's per-flight aggregates."""
    df = df.rename(columns=lambda c: str(c).strip())
    if all(f"m_{s}" in df for s in nc.XS_VARS) and "phi_0" in df:
        cyc = df.copy()
    elif all(c in df for c in nc.W_VARS + nc.XS_VARS):
        if "unit" not in df:
            df["unit"] = 1
        if "cycle" not in df:
            raise ValueError("1 Hz upload needs a 'cycle' column (flight number per engine)")
        df = df.sort_values(["unit", "cycle"], kind="stable")
        cyc = nc.aggregate_cycles(df)
    else:
        missing = [c for c in nc.W_VARS + nc.XS_VARS if c not in df]
        raise ValueError(f"Upload must contain N-CMAPSS columns; missing {missing}")
    for c in ("fc",):
        if c not in cyc:
            cyc[c] = 0
    return cyc


def truth_columns(demo: pd.DataFrame) -> pd.DataFrame:
    """Rename label columns of the demo file so they never reach the feature pipeline."""
    d = demo.copy()
    ren = {f"hi_{c}": f"truth_hi_{c}" for c in nc.COMPONENTS}
    ren.update({"rul": "truth_rul", "hs": "truth_hs"})
    d = d.rename(columns={k: v for k, v in ren.items() if k in d})
    return d.drop(columns=[t for t in nc.T_VARS if t in d])
