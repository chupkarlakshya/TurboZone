"""Tests for the shared N-CMAPSS pipeline. No model is fitted: the healthy reference and the Kalman filter use
hand-set parameters, so these tests only check the maths and data handling."""
import numpy as np
import pandas as pd
import pytest

from backend.pipeline import ncmapss as nc


def _flight_frame(n_units=2, n_cycles=4, n_sec=50, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, n_cycles + 1):
            for _ in range(n_sec):
                w = [rng.uniform(5000, 35000), rng.uniform(0.2, 0.8), rng.uniform(40, 90), rng.uniform(470, 530)]
                x = [600 + rng.normal(), 1500 + rng.normal(), 1800, 1300, 20, 14, 18, 25, 300, 280, 22, 2000,
                     8500 + rng.normal(), 2.0 + 0.01 * rng.normal()]
                t = list(-0.001 * c * np.arange(1, 11) / 10)
                rows.append(w + x + t + [u, c, 1 + (c % 3), 1 if c < 3 else 0, 10 - c])
    cols = nc.W_VARS + nc.XS_VARS + nc.T_VARS + ["unit", "cycle", "Fc", "hs", "RUL"]
    return pd.DataFrame(rows, columns=cols)


def _write_h5(path, df):
    import h5py
    with h5py.File(path, "w") as f:
        groups = {"W": nc.W_VARS, "X_s": nc.XS_VARS[::-1], "T": nc.T_VARS, "A": ["unit", "cycle", "Fc", "hs"]}
        for split in ("dev", "test"):
            for g, names in groups.items():
                f[f"{g}_{split}"] = df[names].to_numpy()
            f[f"Y_{split}"] = df[["RUL"]].to_numpy()
            f[f"X_v_{split}"] = np.column_stack([df["T30"] + 900, df["Ps30"] * 1.05, df["P50"] * 2])
        for g, names in groups.items():
            f[f"{g}_var"] = np.array(names, dtype="S")
        f["X_v_var"] = np.array(["T40", "P30", "P45"], dtype="S")


def test_load_h5_maps_columns_by_name(tmp_path):
    df = _flight_frame()
    p = tmp_path / "N-CMAPSS_DS99.h5"
    _write_h5(p, df)
    got = nc.load_h5(str(p), "dev")
    for c in nc.W_VARS + nc.XS_VARS + nc.T_VARS:            # X_s was stored in reversed order
        assert np.allclose(got[c].to_numpy(), df[c].to_numpy(), rtol=1e-6), c
    assert got["unit"].dtype.kind == "i" and "v_T40" in got and "RUL" in got


def test_aggregate_cycles_matches_pandas_and_ignores_nan():
    df = _flight_frame()
    df.loc[3, "T30"] = np.nan
    agg = nc.aggregate_cycles(df)
    assert len(agg) == 8 and list(agg[["unit", "cycle"]].iloc[0]) == [1, 1]
    expect = df.groupby(["unit", "cycle"])["T30"].mean().to_numpy()   # pandas mean skips NaN
    assert np.allclose(agg["m_T30"].to_numpy(), expect)
    assert np.allclose(agg["sd_Nc"].to_numpy(), df.groupby(["unit", "cycle"])["Nc"].std(ddof=0).to_numpy())
    phi = nc.poly_features(nc.scaled_w(df[nc.W_VARS].to_numpy()))
    phim = pd.DataFrame(phi).groupby([df.unit, df.cycle]).mean().to_numpy()
    assert np.allclose(agg[[f"phi_{j}" for j in range(phi.shape[1])]].to_numpy(), phim)
    assert (agg["hs"].iloc[:4].tolist() == [1, 1, 0, 0]) and agg["duration_s"].eq(50).all()


def test_physics_proxies_known_values():
    t2, p2 = 500.0, 14.0
    p24 = p2 * 2.0
    t24 = t2 * (1 + (2.0 ** nc.K_COLD - 1) / 0.88)                    # eta_core = 0.88 by construction
    ps30 = p24 * 6.0
    t30 = t24 * (1 + (6.0 ** nc.K_COLD - 1) / 0.85)                   # eta_hpc = 0.85
    row = {"T2": t2, "P2": p2, "P21": p2 * 1.5, "P24": p24, "T24": t24, "Ps30": ps30, "T30": t30, "P40": ps30 * 0.95,
           "T48": 1700.0, "T50": 1300.0, "P50": 20.0, "Nf": 2000.0, "Nc": 9000.0, "Wf": 2.0}
    p = nc.physics_1hz(pd.DataFrame([row])).iloc[0]
    assert p["eta_core"] == pytest.approx(0.88, rel=1e-5)
    assert p["eta_hpc"] == pytest.approx(0.85, rel=1e-5)
    assert p["burner_dp"] == pytest.approx(0.05, rel=1e-5)
    assert p["fan_pr"] == pytest.approx(1.5, rel=1e-6)


def test_reference_mean_identity_without_fitting():
    """mean over a cycle of the polynomial map == map applied to the mean polynomial features (exact)."""
    df = _flight_frame()
    ref = nc.HealthyReference()
    ref.coef = np.random.default_rng(1).normal(size=(len(nc.poly_terms(4, 3)), len(nc.XS_VARS)))
    agg = nc.aggregate_cycles(df)
    via_phi = ref.expected_from_phi(agg[[f"phi_{j}" for j in range(ref.coef.shape[0])]].to_numpy())
    direct = pd.DataFrame(ref.predict(df[nc.W_VARS].to_numpy())).groupby([df.unit, df.cycle]).mean().to_numpy()
    assert np.allclose(via_phi, direct)
    r2 = nc.HealthyReference.from_dict(ref.to_dict())
    assert np.allclose(r2.coef, ref.coef)


def test_component_health_formula():
    t = pd.DataFrame([{k: 0.0 for k in nc.T_VARS}])
    t["HPT_eff_mod"], t["HPT_flow_mod"], t["fan_flow_mod"] = -0.02, 0.01, -0.04
    hi = nc.component_health(t, d_ref=0.04).iloc[0]
    assert hi["hi_hpt"] == pytest.approx(0.5) and hi["hi_fan"] == pytest.approx(0.0) and hi["hi_lpc"] == 1.0


def _manual_kf(rng):
    kf = nc.GasPathKF()
    s, p, k = len(nc.XS_VARS), len(nc.T_VARS), 4
    kf.b = np.zeros((k, s))
    kf.H = np.zeros((p, k, s))
    kf.H[:, 0, :] = rng.normal(size=(p, s))                            # operating-point independent part
    kf.R = np.full(s, 1e-6)
    kf.Q = np.full(p, 1e-6)
    kf.x0 = np.zeros(p)
    kf.P0 = np.full(p, 1e-3)
    return kf


def test_kalman_filter_tracks_known_parameters_and_gates_a_spike():
    rng = np.random.default_rng(3)
    kf = _manual_kf(rng)
    n = 60
    theta = np.outer(np.linspace(0, 1, n), -0.02 * rng.uniform(0.2, 1, len(nc.T_VARS)))
    h = kf.H[:, 0, :].T                                               # (S, P)
    res = theta @ h.T + 1e-3 * rng.normal(size=(n, len(nc.XS_VARS)))
    cyc = pd.DataFrame(res, columns=[f"res_{s}" for s in nc.XS_VARS])
    cyc["unit"], cyc["cycle"] = 1, np.arange(1, n + 1)
    cyc["alt_mean"], cyc["Mach_mean"], cyc["TRA_mean"] = 20000.0, 0.5, 60.0
    cyc.loc[40, "res_T48"] += 1.0                                     # faulty reading
    out = kf.run(cyc)
    est = out[[f"kf_{t}" for t in nc.T_VARS]].to_numpy()
    assert np.abs(est[-1] - theta[-1]).max() < 3e-3
    assert out.loc[40, "gated_T48"] and not out["gated_T48"].drop(40).any()
    assert (out[[f"kf_{t}_sigma" for t in nc.T_VARS]].to_numpy() > 0).all()
    hi = nc.kf_component_health(out, d_ref=0.03)
    assert hi.filter(like="kf_hi_").drop(columns=hi.filter(like="sigma").columns).apply(lambda c: c.between(0, 1)).all().all()
    k2 = nc.GasPathKF.from_dict(kf.to_dict())
    assert np.allclose(k2.run(cyc)[f"kf_{nc.T_VARS[0]}"], out[f"kf_{nc.T_VARS[0]}"])


def _manual_featurizer():
    f = nc.CausalFeaturizer(base_cols=["res_T48", "p_eta_hpc"], baseline_cycles=3, roll=2)
    n = len(f.base_cols) + len(f.flight_cols)
    f.mu, f.sd = np.zeros(n), np.ones(n)
    f._names()
    return f


def _cycles(n_units=2, n=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, n + 1):
            rows.append({"unit": u, "cycle": c, "res_T48": 0.001 * c + rng.normal() * 1e-4,
                         "p_eta_hpc": 0.85 - 0.001 * c, "alt_mean": 2e4, "Mach_mean": 0.5, "TRA_mean": 60,
                         "T2_mean": 500, "duration_s": 3000, "fc": 2})
    return pd.DataFrame(rows)


def test_featurizer_is_causal_and_serialisable():
    f = _manual_featurizer()
    cyc = _cycles()
    full = f.transform(cyc)
    cut = f.transform(cyc[(cyc.unit == 1) & (cyc.cycle <= 6)])
    assert np.allclose(cut[f.feature_names].to_numpy(), full[full.unit == 1][f.feature_names].to_numpy()[:6])
    g = nc.CausalFeaturizer.from_dict(f.to_dict())
    assert np.allclose(g.transform(cyc)[f.feature_names].to_numpy(), full[f.feature_names].to_numpy())
    assert set(f.seq_names) <= set(full.columns)


def test_windows_are_causal_and_padded():
    f = _manual_featurizer()
    fr = f.transform(_cycles())
    big, st = nc.build_windows(fr, f.seq_names, 5)
    w = nc.gather_windows(big, st, 5, np.arange(len(fr)))
    seq = fr[f.seq_names].to_numpy(np.float32)
    assert np.allclose(w[:, -1], seq)
    assert np.allclose(w[0], np.repeat(seq[:1], 5, axis=0))              # padding with the first cycle
    assert np.allclose(w[12, -1], seq[12]) and np.allclose(w[12, :-1], np.repeat(seq[12:13], 4, axis=0))  # unit 2 start


def test_component_health_with_per_component_failure_scale():
    t = pd.DataFrame([{k: 0.0 for k in nc.T_VARS}])
    t["HPT_eff_mod"], t["fan_flow_mod"] = -0.01, -0.04
    d_ref = {"fan": 0.08, "lpc": 0.05, "hpc": 0.05, "hpt": 0.02, "lpt": 0.03}
    hi = nc.component_health(t, d_ref).iloc[0]
    assert hi["hi_hpt"] == pytest.approx(0.5) and hi["hi_fan"] == pytest.approx(0.5)   # each on its own scale
    end = pd.DataFrame([{**{k: 0.0 for k in nc.T_VARS}, "unit": u, "cycle": c, "HPT_eff_mod": -0.001 * u * c}
                        for u in (1, 2, 3) for c in (1, 2)])
    d = nc.fit_damage_reference(end, q=1.0)
    assert d["hpt"] == pytest.approx(0.006) and d["fan"] == pytest.approx(1e-3)          # floor for no damage


def test_causal_smoothing():
    units, cycles = np.array([1, 1, 1, 2, 2]), np.array([1, 2, 3, 1, 2])
    v = np.array([10.0, 12.0, 6.0, 5.0, 9.0])
    assert np.allclose(nc.causal_smooth(units, cycles, v, 1.0), v)
    rul = nc.causal_smooth(units, cycles, v, 0.5, "rul")
    assert rul[1] == pytest.approx(0.5 * 12 + 0.5 * (10 - 1)) and rul[3] == 5.0           # resets per engine
    cut = nc.causal_smooth(units[:2], cycles[:2], v[:2], 0.5, "rul")
    assert np.allclose(cut, rul[:2])                                                      # causal
    assert (nc.smooth_target("hi_hpt", units, cycles, np.array([1.2, 0.9, -0.1, 1, 1]), 0.5) <= 1).all()


def _manual_reference():
    ref = nc.HealthyReference()
    ref.coef = np.zeros((len(nc.poly_terms(4, 3)), len(nc.XS_VARS)))
    ref.coef[0] = np.arange(1, len(nc.XS_VARS) + 1) * 100.0                               # ideal = constant
    return ref


def test_downsample_encoder_and_cycle_mean():
    rows = [[1, c, 2e4, 0.5, 60, 500] + list(np.arange(1, 15) * 100.0 * (1.01 if c == 2 else 1.0))
            for c in (1, 2) for _ in range(25)]
    df = pd.DataFrame(rows, columns=["unit", "cycle"] + nc.W_VARS + nc.XS_VARS)
    s = nc.downsample_1hz(df, 10)
    assert list(s["cycle"]) == [1, 1, 1, 2, 2, 2]                                         # rows 0, 10, 20 per flight
    cyc = pd.DataFrame({"unit": [1, 2], "cycle": [2, 1], **{f"kf_{t}": [0.5, 0.0] for t in nc.T_VARS}})
    enc = nc.SampleEncoder()
    raw, units, rows_ = enc.raw(s, cyc, _manual_reference())
    assert len(raw) == 3 and (rows_ == 0).all()                                          # only (unit 1, cycle 2)
    ch = enc.channels
    assert np.allclose(raw[:, ch.index("res_T48")], 0.01, atol=1e-6) and np.allclose(raw[:, ch.index("kf_HPT_eff_mod")], 0.5)
    occ, _, _ = enc.raw(s, cyc, _manual_reference(), occlude="T48")
    assert np.allclose(occ[:, ch.index("res_T48")], 0) and np.allclose(occ[:, ch.index("x_T48")], 300.0)
    m = nc.cycle_mean(np.array([0, 0, 2]), np.array([[1.0], [3.0], [5.0]]), 3)
    assert m[0, 0] == 2.0 and np.isnan(m[1, 0]) and m[2, 0] == 5.0


def test_truncated_h5_is_repaired_and_zeroed_rows_dropped(tmp_path):
    df = _flight_frame()
    p = tmp_path / "N-CMAPSS_DS98.h5"
    _write_h5(p, df)
    data = p.read_bytes()
    p.write_bytes(data[:-32])                                         # like Kaggle's DS08d-010
    fixed, repaired = nc.repair_truncated_h5(str(p), str(tmp_path))
    assert repaired is not None and fixed != str(p)
    got = nc.load_h5(fixed, "test")
    assert 0 < len(got) <= len(df) and (got[nc.XS_VARS].to_numpy() > 0).all()
    ok, none = nc.repair_truncated_h5(fixed, str(tmp_path))           # intact file: used as is
    assert ok == fixed and none is None
    p.write_bytes(data[: len(data) // 2])                             # badly damaged: refuse, don't guess
    with pytest.raises(OSError):
        nc.repair_truncated_h5(str(p), str(tmp_path))
