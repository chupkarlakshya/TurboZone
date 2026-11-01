"""Physics monitors for the two gas-path parts N-CMAPSS never degrades: the combustor and the exhaust nozzle.

Nothing here is learned. Each monitor is a first-principles formula evaluated twice per flight: on the measured flight
means (m_<sensor>) and on the healthy-engine reference at the same flight condition (ref_<sensor>, Layer 2). The ratio
measured / reference - 1 is the monitor's deviation; constants that the formulas cannot know (HPT nozzle-guide-vane
area, cooling-air dilution, bleed fractions) cancel in that ratio. Only measured sensors are used: the simulator's
virtual T40 / P30 are never inputs, they are used by `physics_check` to validate the formulas.

  Combustor
    T40     turbine-inlet temperature from the HP-spool power balance  h(T40) = h(T48) + k * [h(T30) - h(T24)]
    m_core  core mass flow through the choked HPT nozzle guide vanes   m ~ (1 + HPT flow modifier) * P40 / sqrt(T40)
            (the guide-vane flow capacity is exactly what the EKF's HPT flow state estimates; without the
            correction a failing HPT would leak into both monitors)
    eta_b   combustion efficiency from the burner energy balance       eta_b ~ m_core * [h(T40) - h(T30)] / (Wf * LHV)
    dp_b    burner pressure ratio                                       P40 / Ps30
  Nozzle
    A8      effective core-nozzle area from compressible continuity     A ~ m_core * sqrt(T50) / (P50 * f(P50 / p_amb))
            p_amb = static pressure from the fan-inlet total pressure P2 and flight Mach number

Health index per monitor: 1 inside a tolerance band, falling linearly to 0 at a failure limit. The limits are
engineering assumptions (documented below), not fitted: N-CMAPSS contains no degraded combustor or nozzle to fit them
on. Monitors are smoothed causally per engine (exponential average) before the mapping.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------------ constants
R_AIR = 287.05                       # J/(kg K)
LHV_JET_A = 43.1e6                   # J/kg, lower heating value of Jet-A
GAMMA_HOT = 1.333                    # core-nozzle gas
GAMMA_COLD = 1.4
# Cengel, Thermodynamics, Table A-2c: ideal-gas cp of air, kJ/(kmol K), 273-1800 K, max error 0.72 %
CP_A, CP_B, CP_C, CP_D = 28.11, 0.1967e-2, 0.4802e-5, -1.966e-9
M_AIR = 28.97
SPOOL_WORK_RATIO = 1.0 / 0.99        # k: HPT enthalpy drop per unit HPC enthalpy rise (mechanical efficiency 0.99).
                                     # Cooling-air dilution and power off-take raise the real value; it cancels in the
                                     # measured / reference ratio (physics_check reports the value the data implies).
DEG_R_TO_K = 5.0 / 9.0
PSIA_TO_PA = 6894.757
LBM_TO_KG = 0.45359237

# monitor -> (direction that means damage, tolerance, failure limit), as fractions of the healthy value
#   eta_b: combustion efficiency is ~99 % when healthy; a 4 % loss (fuel-nozzle coking, liner damage) is treated as
#          failure-level. Tolerance 1 %: engine fuel-flow meters are accurate to about +/-0.5-1 % of reading.
#   dp_b:  design burner loss is ~4.5 % of P30 (the simulator's value); a 3 % drop in P40/P30 roughly doubles it.
#   a8:    effective nozzle area; a 6 % change either way (erosion, flap or liner damage) is treated as failure-level.
LIMITS = {"eta_b": ("down", 0.01, 0.04), "dp_b": ("down", 0.005, 0.03), "a8": ("both", 0.01, 0.06)}
EWMA_SPAN = 5                        # flights
AUX_COMPONENTS = ["burner", "nozzle"]
AUX_NAMES = {"burner": "Combustor", "nozzle": "Nozzle"}
AUX_MONITORS = {"burner": ["eta_b", "dp_b"], "nozzle": ["a8"]}
MONITOR_LABELS = {"eta_b": "Combustion efficiency", "dp_b": "Burner pressure ratio", "a8": "Nozzle effective area"}


# ------------------------------------------------------------------------------------------------ thermodynamics
def cp_air(t):
    """Specific heat of air, J/(kg K), T in K."""
    return (CP_A + CP_B * t + CP_C * t ** 2 + CP_D * t ** 3) / M_AIR * 1e3


def h_air(t):
    """Specific enthalpy of air relative to 0 K, J/kg, T in K (integral of cp_air)."""
    return (CP_A * t + CP_B * t ** 2 / 2 + CP_C * t ** 3 / 3 + CP_D * t ** 4 / 4) / M_AIR * 1e3


def t_from_h(h, t0):
    """Invert h_air by Newton iteration from a first guess t0 (K)."""
    t = np.asarray(t0, dtype=np.float64).copy()
    for _ in range(8):
        t = t - (h_air(t) - h) / cp_air(t)
    return t


def t40_spool_balance(t24, t30, t48, k=SPOOL_WORK_RATIO):
    """Turbine-inlet temperature (K) from the HP-spool power balance; HPC inlet = LPC exit (station 24)."""
    return t_from_h(h_air(t48) + k * (h_air(t30) - h_air(t24)), t48 + (t30 - t24))


def static_pressure(pt, mach, gamma=GAMMA_COLD):
    return pt / (1.0 + (gamma - 1.0) / 2.0 * mach ** 2) ** (gamma / (gamma - 1.0))


def isa_pressure(alt_ft):
    """ISA static pressure (Pa) for a geopotential altitude in feet (troposphere and lower stratosphere)."""
    h = np.asarray(alt_ft, dtype=np.float64) * 0.3048
    return np.where(h <= 11000.0, 101325.0 * (1.0 - 2.25577e-5 * h) ** 5.25588, 22632.06 * np.exp(-(h - 11000.0) / 6341.62))


def nozzle_flow_function(npr, gamma=GAMMA_HOT, r=R_AIR):
    """Mass-flow function m*sqrt(Tt)/(A*Pt) of a convergent nozzle at total-to-ambient pressure ratio npr.
    Returns (flow function, choked flag); above the critical ratio the flow is choked and the function is constant."""
    crit = ((gamma + 1.0) / 2.0) ** (gamma / (gamma - 1.0))
    npr = np.maximum(np.asarray(npr, dtype=np.float64), 1.0 + 1e-9)
    pr = np.minimum(npr, crit)
    f = np.sqrt(2.0 * gamma / ((gamma - 1.0) * r) * (pr ** (-2.0 / gamma) - pr ** (-(gamma + 1.0) / gamma)))
    return f, npr >= crit


def entropy_change(t1, p1, t2, p2):
    """Ideal-gas entropy change, J/(kg K), with variable cp: s2 - s1 = integral(cp/T dT) - R ln(p2/p1)."""
    def s0(t):
        return (CP_A * np.log(t) + CP_B * t + CP_C * t ** 2 / 2 + CP_D * t ** 3 / 3) / M_AIR * 1e3
    return s0(t2) - s0(t1) - R_AIR * np.log(p2 / p1)


# ------------------------------------------------------------------------------------------------ monitors
def raw_monitors(df: pd.DataFrame, prefix: str, hpt_flow=None) -> dict:
    """Monitor values (up to unknown constants) from <prefix><sensor> columns, prefix 'm_' or 'ref_'.
    hpt_flow: relative change of the HPT guide-vane flow capacity (EKF estimate on the measured side, 0 for the
    healthy reference)."""
    k = lambda s: df[prefix + s].to_numpy(dtype=np.float64) * DEG_R_TO_K
    p = lambda s: df[prefix + s].to_numpy(dtype=np.float64) * PSIA_TO_PA
    t24, t30, t48, t50 = k("T24"), k("T30"), k("T48"), k("T50")
    p40, p50, ps30, p2 = p("P40"), p("P50"), p("Ps30"), p("P2")
    wf = df[prefix + "Wf"].to_numpy(dtype=np.float64) * LBM_TO_KG
    t40 = t40_spool_balance(t24, t30, t48)
    flow_cap = 1.0 if hpt_flow is None else 1.0 + np.asarray(hpt_flow, dtype=np.float64)
    m_core = flow_cap * p40 / np.sqrt(t40)               # x unknown NGV throat constant (cancels in the ratio)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta_b = m_core * (h_air(t40) - h_air(t30)) / (wf * LHV_JET_A)
        p_amb = static_pressure(p2, df["Mach_mean"].to_numpy(dtype=np.float64))
        f, choked = nozzle_flow_function(p50 / p_amb)
        a8 = m_core * np.sqrt(t50) / (p50 * f)
    return {"t40": t40, "eta_b": eta_b, "dp_b": p40 / ps30, "a8": a8, "npr": p50 / p_amb, "choked": choked}


def health_from_deviation(dev, direction, tol, limit):
    d = np.asarray(dev, dtype=np.float64)
    loss = -d if direction == "down" else (d if direction == "up" else np.abs(d))
    return np.clip(1.0 - np.maximum(loss - tol, 0.0) / (limit - tol), 0.0, 1.0)


def monitor_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Combustor and nozzle monitors for a pipeline output frame (needs m_*, ref_*, Mach_mean, unit, cycle).

    Columns: phys_t40_K (spool-balance T40), phys_npr, phys_nozzle_choked, dev_<monitor> (raw deviation),
    dev_<monitor>_smooth (causal per-engine EWMA), hi_<monitor>, hi_burner / hi_nozzle (+ _sigma)."""
    hpt_flow = df["kf_HPT_flow_mod"].to_numpy(dtype=np.float64) if "kf_HPT_flow_mod" in df else None
    meas, ref = raw_monitors(df, "m_", hpt_flow), raw_monitors(df, "ref_")
    out = pd.DataFrame(index=df.index)
    out["phys_t40_K"] = meas["t40"]
    out["phys_npr"] = meas["npr"]
    out["phys_nozzle_choked"] = meas["choked"]
    unit = df["unit"].to_numpy() if "unit" in df else np.zeros(len(df))
    for name, (direction, tol, limit) in LIMITS.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = meas[name] / ref[name] - 1.0
        dev = pd.Series(dev, index=df.index)
        smooth = dev.groupby(unit).transform(lambda s: s.ewm(span=EWMA_SPAN, adjust=True).mean())
        spread = dev.groupby(unit).transform(lambda s: s.rolling(EWMA_SPAN, min_periods=2).std()).fillna(0.0)
        out[f"dev_{name}"] = dev.to_numpy()
        out[f"dev_{name}_smooth"] = smooth.to_numpy()
        out[f"hi_{name}"] = health_from_deviation(smooth, direction, tol, limit)
        out[f"hi_{name}_sigma"] = spread.to_numpy() / (limit - tol)
    for comp, mons in AUX_MONITORS.items():
        his = np.column_stack([out[f"hi_{m}"] for m in mons])
        worst = his.argmin(axis=1)
        sig = np.column_stack([out[f"hi_{m}_sigma"] for m in mons])
        out[f"hi_{comp}"] = his.min(axis=1)
        out[f"hi_{comp}_sigma"] = sig[np.arange(len(out)), worst]
    return out
