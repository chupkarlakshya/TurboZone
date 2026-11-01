"""Deep physics check of the pipeline output on the bundled test engines (real N-CMAPSS flights).

    python -m backend.physics_check          # prints the report; exit code 1 if any check fails

Three groups of checks:
  1. Conservation laws and thermodynamic limits on every flight (energy balance of the HP spool and the burner, second
     law across compressor and turbines, isentropic efficiencies, Carnot limit, nozzle flow regime, atmosphere).
     The simulator's virtual T40 / P30 are used here as the reference they are, never as monitor inputs.
  2. The combustor and nozzle monitors at rest: they stay at their healthy value on healthy flights, and do not move
     while the compressor and turbines wear (N-CMAPSS never degrades the combustor or nozzle).
  3. Sensitivity: a known fault injected into the measured inputs (extra fuel for the same heat release, a burner
     pressure drop, a larger nozzle area) is recovered by the monitor at the injected size.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .pipeline import physics_monitors as pm

K = pm.DEG_R_TO_K
PA = pm.PSIA_TO_PA


def _s0(t):
    return (pm.CP_A * np.log(t) + pm.CP_B * t + pm.CP_C * t ** 2 / 2 + pm.CP_D * t ** 3 / 3) / pm.M_AIR * 1e3


def _t_isentropic(t1, p1, p2):
    """Exit temperature of an isentropic process from (t1, p1) to p2 with variable cp (Newton on s0)."""
    target = _s0(t1) + pm.R_AIR * np.log(p2 / p1)
    t = t1 * (p2 / p1) ** ((pm.GAMMA_COLD - 1.0) / pm.GAMMA_COLD)        # constant-cp first guess
    for _ in range(10):
        t = t - (_s0(t) - target) / (pm.cp_air(t) / t)
    return t


def _check(cid, name, value, expect, ok, detail=""):
    return {"id": cid, "name": name, "value": value, "expect": expect, "pass": bool(ok), "detail": detail}


def _pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def run_checks(out: pd.DataFrame) -> dict:
    """out: ModelBundle.run() output for engines with truth columns (truth_hs, truth_hi_*)."""
    t = lambda s: out["m_" + s].to_numpy(np.float64) * K
    p = lambda s: out["m_" + s].to_numpy(np.float64) * PA
    t24, t30, t48, t50, t2 = t("T24"), t("T30"), t("T48"), t("T50"), out["T2_mean"].to_numpy(np.float64) * K
    p24, p40, p50, ps30 = p("P24"), p("P40"), p("P50"), p("Ps30")
    wf = out["m_Wf"].to_numpy(np.float64) * pm.LBM_TO_KG
    t40v = out["v_T40"].to_numpy(np.float64) * K               # simulator T40 (reference only)
    p30v = out["v_P30"].to_numpy(np.float64) * PA              # simulator total P30 (reference only)
    healthy = out["truth_hs"].to_numpy() == 1 if "truth_hs" in out else np.ones(len(out), bool)
    checks = []

    # ---- 1. conservation laws and limits ------------------------------------------------------------------
    t40s = out["phys_t40_K"].to_numpy(np.float64)
    corr = float(np.corrcoef(t40s, t40v)[0, 1])
    k_impl = (pm.h_air(t40v) - pm.h_air(t48)) / (pm.h_air(t30) - pm.h_air(t24))
    k_cv = float(np.std(k_impl) / np.mean(k_impl))
    checks.append(_check(
        "spool_balance", "HP-spool energy balance closes",
        f"corr {corr:.4f}; implied work ratio k = {np.median(k_impl):.3f} (CV {_pct(k_cv)})", "corr > 0.99, CV < 2%",
        corr > 0.99 and k_cv < 0.02,
        "T40 from HPT work = HPC work tracks the simulator's T40 flight by flight. The constant k > 1 is turbine "
        "cooling-air dilution, power off-take and hot-gas cp; it cancels in the monitors' measured/healthy ratio."))

    loss = 1.0 - p40 / p30v
    checks.append(_check(
        "burner_loss", "Burner total-pressure loss in the physical range",
        f"median {_pct(np.median(loss))}, spread {_pct(np.percentile(loss, 95) - np.percentile(loss, 5), 3)}",
        "2% to 8% of P30", 0.02 <= np.median(loss) <= 0.08,
        "Measured P40 against the simulator's total P30. A constant value also shows N-CMAPSS never degrades the burner."))

    m_core = p40 / np.sqrt(t40v)
    eb = m_core * (pm.h_air(t40v) - pm.h_air(t30)) / (wf * pm.LHV_JET_A)
    eb_cv = float(np.std(eb[healthy]) / np.mean(eb[healthy]))
    checks.append(_check(
        "burner_energy", "Burner energy balance is consistent across flight conditions",
        f"CV of m·Δh / (Wf·LHV) on healthy flights {_pct(eb_cv)}", "< 2%", eb_cv < 0.02,
        "With choked HPT guide vanes (m ~ P40/√T40), fuel energy and the burner's enthalpy rise stay proportional "
        "across altitude, Mach and throttle: the combustion efficiency is constant, as it should be."))

    # second law and isentropic efficiency on the UNCOOLED components. The HPT is film-cooled (compressor air joins
    # between stations 40 and 48), so a single-stream entropy balance across it is not valid; the HP-spool energy
    # balance above covers it instead.
    p2_, p45v = p("P2"), out["v_P45"].to_numpy(np.float64) * PA
    legs = {"fan + LPC": (t2, p2_, t24, p24), "HPC": (t24, p24, t30, p30v), "burner": (t30, p30v, t40v, p40),
            "LPT": (t48, p45v, t50, p50)}
    ds = {k: pm.entropy_change(*v) for k, v in legs.items()}
    checks.append(_check(
        "second_law", "Second law: entropy rises across every uncooled component",
        "; ".join(f"{k} {np.mean(v > 0):.0%} (min {v.min():.1f} J/kg·K)" for k, v in ds.items()), "100% of flights",
        all(bool((v > 0).all()) for v in ds.values()),
        "Fan + LPC, HPC and LPT are adiabatic with losses and the burner adds heat, so each must generate entropy. "
        "The film-cooled HPT is checked through the spool energy balance instead."))

    def eta_comp(t1, p1, t2_, p2__):
        return (pm.h_air(_t_isentropic(t1, p1, p2__)) - pm.h_air(t1)) / (pm.h_air(t2_) - pm.h_air(t1))

    def eta_turb(t1, p1, t2_, p2__):
        return (pm.h_air(t1) - pm.h_air(t2_)) / (pm.h_air(t1) - pm.h_air(_t_isentropic(t1, p1, p2__)))

    effs = {"fan + LPC": eta_comp(t2, p2_, t24, p24), "HPC": eta_comp(t24, p24, t30, p30v), "LPT": eta_turb(t48, p45v, t50, p50)}
    checks.append(_check(
        "isentropic", "Isentropic efficiencies are physical",
        "; ".join(f"{k} {np.percentile(v, 1):.3f} to {np.percentile(v, 99):.3f}" for k, v in effs.items()),
        "between 0.5 and 1 on every flight", all(bool(((v > 0.5) & (v < 1)).all()) for v in effs.values()),
        "Variable-cp efficiencies from measured temperatures and pressures (P30 and P45 from the simulator). "
        "They also fall as the modules wear, which is the degradation the filter tracks."))

    eta_th = 1.0 - (pm.h_air(t50) - pm.h_air(t2)) / (pm.h_air(t40v) - pm.h_air(t30))
    carnot = 1.0 - t2 / t40v
    checks.append(_check(
        "carnot", "Core cycle stays below the Carnot limit",
        f"core thermal efficiency {np.median(eta_th):.3f} vs Carnot {np.median(carnot):.3f} (median); "
        f"margin min {np.min(carnot - eta_th):.3f}", "below Carnot on every flight", bool((eta_th < carnot).all())))

    p_amb = pm.static_pressure(p("P2"), out["Mach_mean"].to_numpy(np.float64))
    isa = pm.isa_pressure(out["alt_mean"].to_numpy(np.float64))
    ratio = p_amb / isa
    checks.append(_check(
        "atmosphere", "Ambient pressure from P2 and Mach agrees with the standard atmosphere",
        f"median ratio {np.median(ratio):.4f}, 95% within {_pct(np.percentile(np.abs(ratio - 1), 95), 1)}",
        "within 3%", abs(np.median(ratio) - 1) < 0.03,
        "Validates the static pressure the nozzle monitor expands to (inlet recovery assumed ideal)."))

    npr = out["phys_npr"].to_numpy(np.float64)
    choked = out["phys_nozzle_choked"].to_numpy(bool)
    checks.append(_check(
        "nozzle_regime", "Core nozzle flows outward and its regime is known",
        f"pressure ratio {np.min(npr):.2f} to {np.max(npr):.2f}; choked on {choked.mean():.0%} of flights",
        "ratio > 1 on every flight", bool((npr > 1).all()),
        "Below the critical ratio (1.85 for γ = 1.33) the subsonic flow function is used."))

    # ---- 2. monitors at rest ------------------------------------------------------------------------------
    for name, (direction, tol, limit) in pm.LIMITS.items():
        dev = out[f"dev_{name}"].to_numpy(np.float64)
        sm = out[f"dev_{name}_smooth"].to_numpy(np.float64)
        checks.append(_check(
            f"rest_{name}", f"{pm.MONITOR_LABELS[name]} monitor at rest on healthy flights",
            f"mean {_pct(np.mean(dev[healthy]), 3)}, sd {_pct(np.std(dev[healthy]), 3)}", "|mean| < 0.1%, sd < 0.2%",
            abs(np.mean(dev[healthy])) < 0.001 and np.std(dev[healthy]) < 0.002))
        worn = ~healthy
        checks.append(_check(
            f"decouple_{name}", f"{pm.MONITOR_LABELS[name]} monitor ignores compressor and turbine wear",
            f"largest smoothed deviation on degraded flights {_pct(np.max(np.abs(sm[worn])) if worn.any() else 0, 2)} "
            f"(tolerance {_pct(tol, 1)})", "below the tolerance on every flight",
            bool((np.abs(sm[worn]) < tol).all()) if worn.any() else True,
            "N-CMAPSS wears only fan, compressors and turbines, so this monitor must not move."))
    for comp in pm.AUX_COMPONENTS:
        h = out[f"hi_{comp}"].to_numpy(np.float64)
        checks.append(_check(
            f"no_false_{comp}", f"No false alarms on the {pm.AUX_NAMES[comp].lower()}",
            f"health 1.00 on {np.mean(h >= 0.999):.1%} of {len(h)} flights (min {h.min():.3f})", "100%",
            bool((h >= 0.999).all())))

    # ---- 3. sensitivity: inject known faults into the measured inputs -------------------------------------
    sample = out[healthy].groupby("unit").head(3) if healthy.any() else out.head(60)

    def recovered(df, name):
        meas, ref = pm.raw_monitors(df, "m_"), pm.raw_monitors(df, "ref_")
        return float(np.median(meas[name] / ref[name] - 1.0))

    base = {n: recovered(sample, n) for n in pm.LIMITS}
    inj = sample.copy(); inj["m_Wf"] = inj["m_Wf"] * 1.03
    got = recovered(inj, "eta_b") - base["eta_b"]
    exp_ = 1 / 1.03 - 1
    h = pm.health_from_deviation(got, *pm.LIMITS["eta_b"])
    checks.append(_check(
        "inject_eta", "Detects a combustion-efficiency loss",
        f"3% more fuel for the same heat release reads as {_pct(got)} (expected {_pct(exp_)}); combustor health {h:.2f}",
        "within 0.1 points of expected", abs(got - exp_) < 0.001))
    inj = sample.copy(); inj["m_P40"] = inj["m_P40"] * 0.98
    got = recovered(inj, "dp_b") - base["dp_b"]
    h = pm.health_from_deviation(got, *pm.LIMITS["dp_b"])
    checks.append(_check(
        "inject_dp", "Detects a burner pressure-loss increase",
        f"a 2% lower P40 reads as {_pct(got)} (expected −2.00%); combustor health {h:.2f}", "within 0.1 points",
        abs(got + 0.02) < 0.001))
    # nozzle: find the P50 that passes the same core flow through a 5% larger area
    inj = sample.copy()
    t50k = inj["m_T50"].to_numpy(np.float64) * K
    pam = pm.static_pressure(inj["m_P2"].to_numpy(np.float64) * PA, inj["Mach_mean"].to_numpy(np.float64))
    p50 = inj["m_P50"].to_numpy(np.float64) * PA
    target = p50 * pm.nozzle_flow_function(p50 / pam)[0] / 1.05      # m*sqrt(T)/A held, A x 1.05
    lo, hi_ = pam * (1 + 1e-6), p50.copy()
    for _ in range(60):
        mid = (lo + hi_) / 2
        too_high = mid * pm.nozzle_flow_function(mid / pam)[0] > target
        hi_ = np.where(too_high, mid, hi_); lo = np.where(too_high, lo, mid)
    inj["m_P50"] = (lo + hi_) / 2 / PA
    got = recovered(inj, "a8") - base["a8"]
    h = pm.health_from_deviation(got, *pm.LIMITS["a8"])
    checks.append(_check(
        "inject_a8", "Detects a nozzle area change",
        f"a 5% larger nozzle (same flow, lower P50) reads as {_pct(got)} (expected +5.00%); nozzle health {h:.2f}",
        "within 0.1 points", abs(got - 0.05) < 0.001))

    return {"flights": int(len(out)), "engines": int(out["unit"].nunique()), "healthy_flights": int(healthy.sum()),
            "passed": int(sum(c["pass"] for c in checks)), "total": len(checks), "checks": checks,
            "limits": {k: {"direction": d, "tolerance": t_, "failure_limit": l} for k, (d, t_, l) in pm.LIMITS.items()}}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from .app import MODEL_DIR
    from .pipeline.bundle import ModelBundle
    from .pipeline.service import truth_columns
    b = ModelBundle.load(MODEL_DIR)
    rep = run_checks(b.run(truth_columns(b.demo)))
    print(f"Physics check on {rep['engines']} test engines, {rep['flights']} flights ({rep['healthy_flights']} healthy)\n")
    for c in rep["checks"]:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['name']}\n       {c['value']}   (expect {c['expect']})")
    print(f"\n{rep['passed']} of {rep['total']} checks passed")
    return 0 if rep["passed"] == rep["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
