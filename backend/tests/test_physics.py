"""Physics monitors (combustor, nozzle): formulas against textbook values, and monitor behaviour on known inputs."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import app as appmod
from backend.pipeline import physics_monitors as pm
from backend.tests.stub_bundle import stub_bundle


def test_air_properties_match_tables():
    # Cengel Table A-2b: cp of air 1.005 kJ/kg K at 300 K, 1.142 at 1000 K
    assert pm.cp_air(300.0) == pytest.approx(1005, rel=0.01)
    assert pm.cp_air(1000.0) == pytest.approx(1142, rel=0.01)
    # enthalpy is the integral of cp
    t = np.array([400.0, 900.0])
    assert (pm.h_air(t + 0.5) - pm.h_air(t - 0.5)) == pytest.approx(pm.cp_air(t), rel=1e-6)
    assert pm.t_from_h(pm.h_air(1234.5), 1000.0) == pytest.approx(1234.5, abs=1e-6)


def test_atmosphere_and_static_pressure():
    assert pm.isa_pressure(0.0) == pytest.approx(101325.0, rel=1e-6)
    assert pm.isa_pressure(11000 / 0.3048) == pytest.approx(22632.0, rel=2e-3)
    assert pm.static_pressure(100.0, 0.0) == pytest.approx(100.0)
    assert pm.static_pressure(1.0, 1.0) == pytest.approx(1 / 1.8929, rel=1e-3)      # p/pt at Mach 1, gamma 1.4


def test_nozzle_flow_function():
    f_crit = np.sqrt(1.4 / 287.05) * (2 / 2.4) ** (2.4 / 0.8)
    f, choked = pm.nozzle_flow_function(np.array([1.2, 1.893, 3.0]), gamma=1.4, r=287.05)
    assert f[1] == pytest.approx(f_crit, rel=1e-3) and f[2] == pytest.approx(f_crit, rel=1e-9)   # constant once choked
    assert f[0] < f_crit and list(choked) == [False, True, True]


def test_spool_balance_and_entropy():
    t24, t30, t48 = 400.0, 750.0, 950.0
    t40 = pm.t40_spool_balance(t24, t30, t48, k=1.2)
    assert pm.h_air(t40) - pm.h_air(t48) == pytest.approx(1.2 * (pm.h_air(t30) - pm.h_air(t24)), rel=1e-9)
    assert pm.entropy_change(300.0, 1e5, 300.0, 2e5) < 0                     # isothermal compression lowers s
    t2s = 300.0 * 10 ** (0.4 / 1.4)                                         # constant-cp isentrope: ds ~ 0
    assert abs(pm.entropy_change(300.0, 1e5, t2s, 1e6)) < 15


def _frame(n=30):
    """Measured == healthy reference at a cruise-like point (degR, psia, lbm/s)."""
    base = {"T24": 1100.0, "T30": 1500.0, "T48": 1650.0, "T50": 1250.0, "P2": 8.0, "P40": 220.0, "Ps30": 225.0,
            "P50": 10.0, "Wf": 1.6}
    d = {"unit": np.repeat([1], n), "cycle": np.arange(1, n + 1), "Mach_mean": 0.75}
    for s, v in base.items():
        d["m_" + s] = v
        d["ref_" + s] = v
    return pd.DataFrame(d)


def test_monitors_rest_at_healthy_and_detect_faults():
    out = pm.monitor_columns(_frame())
    for m in pm.LIMITS:
        assert np.allclose(out[f"dev_{m}"], 0, atol=1e-12) and np.allclose(out[f"hi_{m}"], 1.0)
    assert np.allclose(out["hi_burner"], 1.0) and np.allclose(out["hi_nozzle"], 1.0)

    f = _frame(); f["m_Wf"] *= 1.03                                          # same heat release for 3 % more fuel
    out = pm.monitor_columns(f)
    assert out["dev_eta_b"].iloc[-1] == pytest.approx(1 / 1.03 - 1, rel=1e-6)
    _, tol, lim = pm.LIMITS["eta_b"]
    assert out["hi_burner"].iloc[-1] == pytest.approx(1 - (0.0291 - tol) / (lim - tol), abs=0.01)
    assert np.allclose(out["hi_nozzle"], 1.0)                                # fuel does not touch the nozzle

    f = _frame(); f["m_P40"] *= 0.97                                         # 3 % burner pressure loss = failure limit
    assert pm.monitor_columns(f)["hi_dp_b"].iloc[-1] == pytest.approx(0.0, abs=1e-9)

    f = _frame(); f["m_P50"] *= 0.99                                         # lower P50 for the same flow: larger nozzle
    assert pm.monitor_columns(f)["dev_a8"].iloc[-1] > 0


def test_hpt_flow_correction_and_smoothing():
    f = _frame(); f["kf_HPT_flow_mod"] = -0.02
    out = pm.monitor_columns(f)
    assert out["dev_eta_b"].iloc[0] == pytest.approx(-0.02, rel=1e-6)        # flow capacity scales the mass flow
    f = _frame(); f.loc[f.index[-1], "m_Wf"] *= 1.05                          # one-flight spike is smoothed
    out = pm.monitor_columns(f)
    assert abs(out["dev_eta_b_smooth"].iloc[-1]) < abs(out["dev_eta_b"].iloc[-1])


def test_health_mapping_directions():
    assert pm.health_from_deviation(0.02, "down", 0.01, 0.04) == 1.0         # an improvement is not damage
    assert pm.health_from_deviation(-0.025, "down", 0.01, 0.04) == pytest.approx(0.5)
    assert pm.health_from_deviation(0.035, "both", 0.01, 0.06) == pytest.approx(0.5)
    assert pm.health_from_deviation(-0.5, "both", 0.01, 0.06) == 0.0


def test_payload_and_api_carry_combustor_and_nozzle():
    appmod.set_bundle(stub_bundle())
    try:
        c = TestClient(appmod.app)
        eid = c.get("/api/engines").json()[0]["engine_id"]
        p = c.get(f"/api/engine/{eid}").json()
        assert p["aux_components"] == ["burner", "nozzle"] and p["component_names"]["burner"] == "Combustor"
        for k in ("hi_burner", "hi_nozzle", "hi_burner_sigma", "dev_eta_b_smooth", "dev_a8", "phys_t40_K", "phys_npr"):
            assert k in p["columns"], k
        assert all(0.0 <= v <= 1.0 for v in p["columns"]["hi_burner"])
        rep = c.get("/api/physics/check").json()
        assert rep["total"] >= 15 and {ch["id"] for ch in rep["checks"]} >= {"spool_balance", "second_law", "inject_eta"}
        ev = {"event_type": "borescope", "finding": "not_found", "performed_by": "t", "component": "burner"}
        assert c.post(f"/api/engine/{eid}/maintenance", json=ev).status_code == 201
    finally:
        appmod.set_bundle(None)
