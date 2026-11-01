"""Tests for the shared N-CMAPSS pipeline. No model is fitted: the healthy baseline and the EKF use hand-set
parameters and the data is synthetic, so these tests only check the maths and data handling."""
import numpy as np
import pandas as pd
import pytest

from backend.pipeline import ncmapss as nc
from backend.tests.stub_bundle import manual_ekf, manual_reference


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


def _write_h5(path, df, y_rows=None):
    import h5py
    with h5py.File(path, "w") as f:
        groups = {"W": nc.W_VARS, "X_s": nc.XS_VARS[::-1], "T": nc.T_VARS, "A": ["unit", "cycle", "Fc", "hs"]}
        for split in ("dev", "test"):
            for g, names in groups.items():
                f[f"{g}_{split}"] = df[names].to_numpy()
            f[f"Y_{split}"] = df[["RUL"]].to_numpy()[: y_rows]
            f[f"X_v_{split}"] = np.column_stack([df["T30"] + 900, df["Ps30"] * 1.05, df["P50"] * 2])
        for g, names in groups.items():
            f[f"{g}_var"] = np.array(names, dtype="S")
        f["X_v_var"] = np.array(["T40", "P30", "P45"], dtype="S")


# ----------------------------------------------------------------------------------------------- Layer 1
def test_load_h5_maps_columns_by_name_and_drops_corrupt_rows(tmp_path):
    df = _flight_frame()
    df.loc[5, "T48"] = -1.0                                             # impossible reading
    p = tmp_path / "N-CMAPSS_DS99.h5"
    _write_h5(p, df)
    got = nc.load_h5(str(p), "dev")
    assert got.attrs["dropped_rows"] == 1 and len(got) == len(df) - 1
    keep = df.drop(index=5).reset_index(drop=True)
    for c in nc.W_VARS + nc.XS_VARS + nc.T_VARS:                       # X_s was stored in reversed order
        assert np.allclose(got[c].to_numpy(), keep[c].to_numpy(), rtol=1e-6), c
    assert got["unit"].dtype.kind == "i" and "v_T40" in got and "RUL" in got
    info = nc.inspect_h5(str(p))
    assert info["datasets"]["X_s_dev"]["shape"] == [len(df), 14]


def test_load_h5_refuses_misaligned_groups(tmp_path):
    p = tmp_path / "bad.h5"
    _write_h5(p, _flight_frame(), y_rows=-3)                           # Y shorter than the other groups
    with pytest.raises(ValueError, match="row-aligned"):
        nc.load_h5(str(p), "dev")


def test_validate_split_reports_and_rejects_bad_ordering():
    df = _flight_frame()
    rep = nc.validate_split(df, "dev")
    assert rep["engines"] == 2 and rep["rul_non_increasing_within_engine"] and rep["hs_values"] == [0, 1]
    shuffled = pd.concat([df[df.unit == 1].iloc[:10], df[df.unit == 2], df[df.unit == 1].iloc[10:]])
    with pytest.raises(ValueError, match="contiguous"):
        nc.validate_split(shuffled, "dev")


def test_aggregate_cycles_flight_summaries_match_pandas():
    df = _flight_frame()
    agg = nc.aggregate_cycles(df)
    g = df.groupby(["unit", "cycle"])
    assert len(agg) == 8 and list(agg[["unit", "cycle"]].iloc[0]) == [1, 1]
    assert np.allclose(agg["m_T30"], g["T30"].mean()) and np.allclose(agg["sd_Nc"], g["Nc"].std(ddof=0))
    assert np.allclose(agg["mn_T24"], g["T24"].min()) and np.allclose(agg["mx_T24"], g["T24"].max())
    assert np.allclose(agg["ls_Wf"], g["Wf"].last()) and agg["duration_s"].eq(50).all()
    phi = nc.poly_features(nc.scaled_w(df[nc.W_VARS].to_numpy()))
    phim = pd.DataFrame(phi).groupby([df.unit, df.cycle]).mean().to_numpy()
    assert np.allclose(agg[[f"phi_{j}" for j in range(phi.shape[1])]].to_numpy(), phim)
    assert agg["hs"].iloc[:4].tolist() == [1, 1, 0, 0] and np.allclose(agg["rul"].iloc[:4], [9, 8, 7, 6])


def test_physics_proxies_known_values():
    row = {"P2": 10.0, "P21": 20.0, "P24": 30.0, "Ps30": 300.0, "T2": 500.0, "T24": 600.0, "T30": 1200.0,
           "P40": 270.0, "P50": 30.0, "T48": 1800.0, "T50": 1200.0, "Nf": 2000.0, "Nc": 8000.0, "Wf": 2.0}
    p = nc.physics_1hz(pd.DataFrame([row])).iloc[0]
    assert p["fan_pr"] == pytest.approx(2.0) and p["hpc_pr"] == pytest.approx(10.0)
    assert p["burner_dp"] == pytest.approx(0.1) and p["lpt_tr"] == pytest.approx(1200 / 1800, rel=1e-6)
    assert p["eta_hpc"] == pytest.approx((10 ** nc.K_COLD - 1) / (2.0 - 1), rel=1e-5)


def test_reference_mean_identity_without_fitting():
    rng = np.random.default_rng(0)
    ref = nc.HealthyReference()
    ref.coef = rng.normal(size=(len(nc.poly_terms(4, 3)), 14))
    df = _flight_frame()
    agg = nc.aggregate_cycles(df)
    via_phi = ref.expected_from_phi(agg[[f"phi_{j}" for j in range(ref.coef.shape[0])]].to_numpy())
    direct = pd.DataFrame(ref.predict(df[nc.W_VARS].to_numpy())).groupby([df.unit, df.cycle]).mean().to_numpy()
    assert np.allclose(via_phi, direct)
    assert np.allclose(nc.HealthyReference.from_dict(ref.to_dict()).coef, ref.coef)


def test_health_mapping_per_component():
    t = pd.DataFrame([{k: 0.0 for k in nc.T_VARS}])
    t["HPT_eff_mod"], t["fan_flow_mod"] = -0.01, -0.04
    d_ref = {"fan": 0.08, "lpc": 0.05, "hpc": 0.05, "hpt": 0.02, "lpt": 0.03}
    hi = nc.component_health(t, d_ref).iloc[0]
    assert hi["hi_hpt"] == pytest.approx(0.5) and hi["hi_fan"] == pytest.approx(0.5) and hi["hi_lpc"] == 1.0
    end = pd.DataFrame([{**{k: 0.0 for k in nc.T_VARS}, "unit": u, "cycle": c, "HPT_eff_mod": -0.001 * u * c}
                        for u in (1, 2, 3) for c in (1, 2)])
    d = nc.fit_damage_reference(end, q=1.0)
    assert d["hpt"] == pytest.approx(0.006) and d["fan"] == pytest.approx(1e-3)          # floor for no damage


# ----------------------------------------------------------------------------------------------- Layer 3 EKF
def _ekf_cycles(e, n=60, seed=3):
    rng = np.random.default_rng(seed)
    theta = np.outer(np.linspace(0, 1, n), -0.02 * rng.uniform(0.2, 1, len(nc.T_VARS)))
    h = e.H_params[:, 0, :].T                                             # (S, P)
    res = theta @ h.T + 1e-3 * rng.normal(size=(n, len(nc.XS_VARS)))
    cyc = pd.DataFrame(res, columns=[f"res_{s}" for s in nc.XS_VARS])
    cyc["unit"], cyc["cycle"] = 1, np.arange(1, n + 1)
    cyc["alt_mean"], cyc["Mach_mean"], cyc["TRA_mean"] = 20000.0, 0.5, 60.0
    return cyc, theta


def test_ekf_tracks_known_parameters_gates_a_spike_and_stays_psd():
    e = manual_ekf(3)
    cyc, theta = _ekf_cycles(e)
    cyc.loc[40, "res_T48"] += 1.0                                         # faulty reading
    out = e.run(cyc)
    est = out[[f"kf_{t}" for t in nc.T_VARS]].to_numpy()
    assert np.abs(est[-1] - theta[-1]).max() < 3e-3
    assert out.loc[40, "gated_T48"] and not out["gated_T48"].drop(40).any()
    assert (out["ekf_min_eig"] > -1e-12).all() and np.isfinite(out["nis_total"]).all()
    assert (out[[f"ekf_{j}_sigma" for j in range(10)]].to_numpy() > 0).all()
    hi = nc.kf_component_health(out, 0.03)
    main = hi[[f"kf_hi_{c}" for c in nc.COMPONENTS]]
    assert ((main >= 0) & (main <= 1)).all().all()
    e2 = nc.GasPathEKF.from_dict(e.to_dict())
    assert np.allclose(e2.run(cyc)["kf_fan_eff_mod"], out["kf_fan_eff_mod"])


def test_ekf_jacobian_matches_finite_differences():
    e = manual_ekf(1)
    e.H_params[:, 1, :] = np.random.default_rng(2).normal(size=(10, 14))     # operating-point dependent part
    e.H = np.einsum("pg,pks->gks", e.A, e.H_params)
    z = np.array([1.0, 0.3, -0.2, 0.5])
    x = np.random.default_rng(4).normal(size=10) * 0.01
    jac, _ = e.h_at(z)
    eps = 1e-6
    fd = np.column_stack([(e.observe(z, x + eps * np.eye(10)[j]) - e.observe(z, x - eps * np.eye(10)[j])) / (2 * eps)
                          for j in range(10)])
    assert np.allclose(jac, fd, atol=1e-8)


def test_merge_states_merges_indistinguishable_parameters_only():
    rng = np.random.default_rng(0)
    hw = rng.normal(size=(20, 14, 4))
    hw[:, :, 1] = 2.0 * hw[:, :, 0]                                       # parameter 1 looks exactly like parameter 0
    theta = np.column_stack([np.linspace(0, -1, 50), np.linspace(0, -0.5, 50), rng.normal(size=50),
                             rng.normal(size=50)])
    groups, loads = nc.merge_states(hw, theta, merge_cos=0.95)
    assert sorted(map(sorted, groups)) == [[0, 1], [2], [3]]
    merged = loads[[sorted(g) for g in groups].index([0, 1])]
    assert np.isclose(np.linalg.norm(merged), 1.0) and abs(merged[0]) > abs(merged[1])


def test_ekf_diagnostics_merged_own_parameters_are_identifiable_but_cross_component_merge_is_not():
    e = manual_ekf(3)
    e.H_params[1] = 2.0 * e.H_params[0] + 1e-9                            # fan eff / flow indistinguishable
    # merge them into ONE state (the intended remedy): states = 9, fan state carries both parameters
    e.groups = [[0, 1]] + [[j] for j in range(2, 10)]
    e.A = np.zeros((10, 9))
    e.A[[0, 1], 0] = np.array([1.0, 2.0]) / np.sqrt(5.0)
    for g in range(1, 9):
        e.A[g + 1, g] = 1.0
    e.state_names = ["fan_eff_mod+fan_flow_mod"] + list(nc.T_VARS[2:])
    e.H = np.einsum("pg,pks->gks", e.A, e.H_params)
    cyc, _ = _ekf_cycles(e)
    rep = e.diagnostics(cyc, n_points=10)
    assert rep["unique_share_per_parameter"]["fan_eff_mod"] < 0.05           # raw parameters are collinear ...
    assert rep["components"]["fan"]["own_parameters_merged"] and rep["components"]["fan"]["identifiable"]   # ... the merged state is not
    e.groups = [[0, 1, 2, 3]] + [[j] for j in range(4, 10)]                  # fan merged with the LPC parameters
    e.A = np.zeros((10, 7))
    e.A[[0, 1, 2, 3], 0] = 0.5
    for g in range(1, 7):
        e.A[g + 3, g] = 1.0
    e.state_names = ["fan+lpc"] + list(nc.T_VARS[4:])
    e.H = np.einsum("pg,pks->gks", e.A, e.H_params)
    cross = e.diagnostics(cyc, n_points=10)["components"]
    assert cross["fan"]["merged_with"] == ["lpc"] and not cross["fan"]["identifiable"]


def test_ekf_diagnostics_flag_collinear_components():
    e = manual_ekf(3)
    e.H_params[1] = 2.0 * e.H_params[0]                                   # fan flow indistinguishable from fan eff
    e.H = e.H_params.copy()
    cyc, _ = _ekf_cycles(e)
    rep = e.diagnostics(cyc, n_points=10)
    assert not rep["components"]["fan"]["identifiable"] and rep["components"]["hpt"]["identifiable"]
    assert rep["unique_share_per_parameter"]["fan_eff_mod"] < 0.05


# ----------------------------------------------------------------------------------------------- Layer 4 features
def _cycles(n_units=2, n=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, n + 1):
            rows.append({"unit": u, "cycle": c, "res_T48": 0.001 * c + rng.normal() * 1e-4, "m_T48": 1800.0 + c,
                         "alt_mean": 2e4, "Mach_mean": 0.5, "TRA_mean": 60, "T2_mean": 500, "alt_max": 3e4,
                         "Mach_max": 0.7, "duration_s": 3600, "fc": 2})
    return pd.DataFrame(rows)


def test_featurizer_window_stats_history_and_causality():
    f = nc.CausalFeaturizer(["res_T48"], ["m_T48"], roll=3)
    cyc = _cycles()
    cyc.loc[0, "m_T48"] = np.nan
    f.fit(cyc)
    out = f.transform(cyc)
    u1 = out[out.unit == 1].reset_index(drop=True)
    x = cyc[cyc.unit == 1]["res_T48"].to_numpy()
    assert u1.loc[4, "res_T48__rm"] == pytest.approx(x[2:5].mean())
    assert u1.loc[4, "res_T48__rs"] == pytest.approx(x[2:5].std())
    assert u1.loc[4, "res_T48__rmin"] == pytest.approx(x[2:5].min())
    assert u1.loc[4, "res_T48__rmax"] == pytest.approx(x[2:5].max())
    assert u1.loc[4, "res_T48__slope"] == pytest.approx(np.polyfit(np.arange(3), x[2:5], 1)[0])
    assert u1.loc[0, "res_T48__slope"] == 0 and u1.loc[0, "hist_len"] == 1 and u1.loc[5, "hist_len"] == 3
    assert u1.loc[2, "exposure_h"] == pytest.approx(3.0)
    assert u1.loc[0, "m_T48"] == pytest.approx(f.fill["m_T48"])           # missing value -> training median
    cut = f.transform(cyc[(cyc.unit == 1) & (cyc.cycle <= 6)])
    assert np.allclose(cut[f.feature_names].to_numpy(), u1[f.feature_names].to_numpy()[:6])   # causal
    g = nc.CausalFeaturizer.from_dict(f.to_dict())
    assert np.allclose(g.transform(cyc)[f.feature_names].to_numpy(), out[f.feature_names].to_numpy())


def test_feature_blocks_and_leakage_audit():
    temporal, static = nc.default_feature_columns(n_states=3)
    f = nc.CausalFeaturizer(temporal, static)
    groups = {nc.feature_group(n) for n in f.feature_names}
    assert groups == {"res", "phys", "ekf", "innov", "raw", "cond"}
    assert nc.feature_group("ekf_2__slope") == "ekf" and nc.is_window_feature("res_T48__rm")
    for leak in (["rul"], ["HPT_eff_mod"], ["hi_hpt"], ["hs"]):
        with pytest.raises(ValueError, match="leaking"):
            nc.audit_features(leak)


# ----------------------------------------------------------------------------------------------- helpers
def test_inject_sensor_fault_only_touches_the_chosen_sensor():
    rows = [{"unit": u, "cycle": c, **{f"m_{s}": 100.0 * (j + 1) for j, s in enumerate(nc.XS_VARS)},
             **{f"phi_{j}": 1.0 if j == 0 else 0.0 for j in range(len(nc.poly_terms(4, 3)))}}
            for u in (1, 2) for c in range(1, 11)]
    ref = nc.HealthyReference()
    ref.coef = np.zeros((len(nc.poly_terms(4, 3)), 14))
    ref.coef[0] = 100.0 * np.arange(1, 15)
    cyc = nc.add_residuals(pd.DataFrame(rows), ref)
    bias = nc.inject_sensor_fault(cyc, ref, "T48", 0.02, start_frac=0.5)
    assert bias["faulted"].sum() == 10 and not bias["faulted"].iloc[:5].any()
    assert np.allclose(bias.loc[bias.faulted, "res_T48"], 0.02) and np.allclose(bias.loc[~bias.faulted, "res_T48"], 0)
    assert np.allclose(bias["res_T24"], cyc["res_T24"])
    spikes = nc.inject_sensor_fault(cyc, ref, "T48", 0.05, start_frac=0.5, every=3)
    assert 0 < spikes["faulted"].sum() < bias["faulted"].sum()


def test_truncated_h5_is_repaired(tmp_path):
    df = _flight_frame()
    p = tmp_path / "N-CMAPSS_DS98.h5"
    _write_h5(p, df)
    data = p.read_bytes()
    p.write_bytes(data[:-32])                                           # like Kaggle's DS08d-010
    fixed, repaired = nc.repair_truncated_h5(str(p), str(tmp_path))
    assert repaired is not None and 0 < len(nc.load_h5(fixed, "test")) <= len(df)
    p.write_bytes(data[: len(data) // 2])                               # badly damaged: refuse, don't guess
    with pytest.raises(OSError):
        nc.repair_truncated_h5(str(p), str(tmp_path))


def test_run_layers_with_hand_set_parameters():
    from backend.tests.stub_bundle import demo_frame
    out = nc.run_layers(demo_frame(n_units=1, n_cycles=8), manual_reference(), manual_ekf(), 0.03)
    assert {"res_T48", "kf_hi_hpt", "innov_Wf", "ekf_9"} <= set(out.columns) and len(out) == 8


def test_merge_states_joint_collinearity_and_inactive_parameters():
    rng = np.random.default_rng(5)
    hw = rng.normal(size=(15, 14, 5))
    hw[:, :, 2] = hw[:, :, 0] + hw[:, :, 1]                              # c = a + b: no pair is similar, jointly collinear
    hw[:, :, 4] = 0.0                                                     # never excited: not passed as active
    theta = rng.normal(size=(40, 5))
    pairwise, _ = nc.merge_states(hw, theta, merge_cos=0.95, active=[0, 1, 2, 3])
    assert sorted(map(sorted, pairwise)) == [[0], [1], [2], [3]]          # pairwise rule alone misses it
    joint, loads = nc.merge_states(hw, theta, merge_cos=0.95, min_unique=0.15, active=[0, 1, 2, 3])
    assert len(joint) == 3 and [3] in joint and all(4 not in g for g in joint)
    assert all(np.isclose(np.linalg.norm(v), 1.0) for v in loads)
    assert np.allclose(nc.unique_shares(np.column_stack([np.ones(3), np.zeros(3)])), [1.0, 0.0])


def test_ekf_fit_excludes_parameters_that_never_move():
    rng = np.random.default_rng(1)
    h_true = rng.normal(size=(10, 14))
    rows = []
    for u in range(1, 13):
        n = 50
        th = np.cumsum(-0.001 * rng.uniform(0, 1, size=(n, 10)), axis=0)
        th[:, :6] = 0.0                                                   # fan, LPC, HPC never degrade (like DS02)
        res = th @ h_true * 0.5 + 1e-4 * rng.normal(size=(n, 14))
        d = pd.DataFrame(res, columns=[f"res_{s}" for s in nc.XS_VARS])
        d[nc.T_VARS] = th
        d["unit"], d["cycle"], d["hs"] = u, np.arange(1, n + 1), (np.arange(n) < 8).astype(int)
        d["alt_mean"], d["Mach_mean"], d["TRA_mean"] = rng.uniform(5e3, 35e3, n), rng.uniform(.2, .8, n), rng.uniform(40, 90, n)
        rows.append(d)
    cyc = pd.concat(rows, ignore_index=True)
    e = nc.GasPathEKF().fit(cyc[cyc.unit <= 9], cyc_val=cyc[cyc.unit > 9])
    assert e.excluded == nc.T_VARS[:6] and all(not any(k < 6 for k in g) for g in e.groups)
    rep = e.diagnostics(cyc[cyc.unit <= 9], n_points=20)
    assert rep["state_condition_number_median"] < 1e6                    # no degenerate zero columns any more
    assert rep["components"]["fan"]["reason"].startswith("no degradation") and not rep["components"]["fan"]["identifiable"]
    assert rep["unique_share_per_parameter"]["fan_eff_mod"] is None
    out = e.run(cyc[cyc.unit > 9])
    assert (out["kf_fan_eff_mod"] == 0).all() and (nc.kf_component_health(out, 0.03)["kf_hi_fan"] == 1).all()
    e2 = nc.GasPathEKF.from_dict(e.to_dict())
    assert e2.excluded == e.excluded and np.allclose(e2.run(cyc[cyc.unit > 9])["kf_HPT_eff_mod"], out["kf_HPT_eff_mod"])
    with pytest.raises(ValueError, match="no health parameter"):
        flat = cyc.copy()
        flat[nc.T_VARS] = 0.0
        nc.GasPathEKF().fit(flat)
