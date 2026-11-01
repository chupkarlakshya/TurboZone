"""Serving side of the pipeline.md pipeline: load the (version 4) bundle exported by
training/ncmapss_colab_training.ipynb and run it on per-flight engine data.

    healthy baseline -> residuals -> EKF (states, uncertainty, innovations) -> causal features
    -> Random Forest / XGBoost / LightGBM / CatBoost -> scaler + Ridge -> RUL (+ empirical interval)
    health indices = EKF parameter estimates mapped with the per-component failure-level damage d_ref

Explanations: TreeSHAP of the base models (weighted by the Ridge effective weights), per-model Ridge contributions,
and occlusion per sensor through the whole pipeline (EKF included).

Bundle layout (backend/models/): manifest.json, schema.json, reference.json, ekf.json, featurizer.json, stack.json,
<model>.<ext> for the four base models, demo_cycles.csv, parity_outputs.csv (+ result tables).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import ncmapss as nc
from . import physics_monitors as pm
from . import stacking as st

BUNDLE_VERSION = 4
MODEL_EXT = {"randomforest": "joblib", "xgboost": "json", "lightgbm": "txt", "catboost": "cbm"}


def _finite(o):
    """NaN/inf -> None so manifest metrics stay JSON-serialisable."""
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def load_model(kind: str, path: str):
    if kind == "xgboost":
        import xgboost as xgb
        m = xgb.XGBRegressor()
        m.load_model(path)
        return m
    if kind == "lightgbm":
        import lightgbm as lgb
        return lgb.Booster(model_file=path)
    if kind == "catboost":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor()
        m.load_model(path)
        return m
    if kind == "randomforest":
        import joblib
        return joblib.load(path)
    raise ValueError(f"Unknown model type {kind}")


class ModelBundle:
    def __init__(self, manifest: dict, ref: nc.HealthyReference, ekf: nc.GasPathEKF, feat: nc.CausalFeaturizer,
                 models: dict, stack: st.StackModel, demo: pd.DataFrame | None = None,
                 forecast_models: dict | None = None, comp_rul_models: dict | None = None):
        self.manifest = manifest
        self.comp_rul = manifest.get("component_rul") or {}      # optional extension: component RUL (flights)
        self.comp_rul_models = comp_rul_models or {}               # component -> LightGBM model
        self.forecast = manifest.get("forecast") or {}           # optional extension: health h flights ahead
        self.forecast_models = forecast_models or {}               # (component, horizon) -> LightGBM model
        self.ref, self.ekf, self.feat = ref, ekf, feat
        self.models, self.stack = models, stack          # name -> fitted base model; Ridge meta-learner
        self.demo = demo
        self.d_ref = manifest["d_ref"]
        self.interval = manifest.get("interval") or {}
        self.targets = ["rul"] + [f"hi_{c}" for c in nc.COMPONENTS]
        self.features = feat.feature_names                      # every feature the featurizer computes
        self.model_features = manifest.get("model_features") or self.features   # the columns the RUL models use

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, model_dir: str) -> "ModelBundle":
        mpath = os.path.join(model_dir, "manifest.json")
        if not os.path.exists(mpath):
            raise FileNotFoundError(f"No manifest.json in {model_dir}: unzip the Colab bundle there.")
        with open(mpath, encoding="utf-8") as fh:
            man = _finite(json.load(fh))
        if man.get("version") != BUNDLE_VERSION:
            raise ValueError(f"Unsupported bundle version {man.get('version')} (expected {BUNDLE_VERSION}): retrain with "
                             "training/ncmapss_colab_training.ipynb")

        def js(name):
            with open(os.path.join(model_dir, man[name]), encoding="utf-8") as fh:
                return json.load(fh)

        ref = nc.HealthyReference.from_dict(js("reference_file"))
        ekf = nc.GasPathEKF.from_dict(js("ekf_file"))
        feat = nc.CausalFeaturizer.from_dict(js("featurizer_file"))
        if feat.feature_names != man["features"]:
            raise ValueError("featurizer.json and manifest feature lists differ")
        stack = st.StackModel.from_dict(js("stack_file"))
        models = {b["name"]: load_model(b["name"], os.path.join(model_dir, b["file"])) for b in man["base_models"]}
        if list(models) != stack.names:
            raise ValueError("base models and stack.json disagree on the model order")
        demo = None
        if man.get("demo_file") and os.path.exists(os.path.join(model_dir, man["demo_file"])):
            demo = pd.read_csv(os.path.join(model_dir, man["demo_file"]))
        fmodels = {}
        fc = man.get("forecast") or {}
        if fc:
            expect = feat.feature_names + [f"kf_hi_{c}" for c in nc.COMPONENTS]
            if fc.get("features") != expect:
                raise ValueError("forecast feature list differs from featurizer.json + EKF health columns")
            for c, by_h in (fc.get("files") or {}).items():
                for h, fn in by_h.items():
                    fmodels[(c, int(h))] = load_model("lightgbm", os.path.join(model_dir, fn))
        cmodels = {}
        cr = man.get("component_rul") or {}
        if cr:
            expect = feat.feature_names + [f"kf_hi_{c}" for c in nc.COMPONENTS]
            if cr.get("features") != expect:
                raise ValueError("component RUL feature list differs from featurizer.json + EKF health columns")
            for c, fn in (cr.get("files") or {}).items():
                cmodels[c] = load_model("lightgbm", os.path.join(model_dir, fn))
        return cls(man, ref, ekf, feat, models, stack, demo, fmodels, cmodels)

    # ------------------------------------------------------------------ prediction
    def base_predictions(self, frame: pd.DataFrame) -> np.ndarray:
        return st.predict_bases(self.models, self.stack.names, frame[self.model_features])

    def _extended_matrix(self, frame: pd.DataFrame, health: pd.DataFrame) -> np.ndarray:
        """Features of the extension models: all featurizer columns + today's EKF health of every component."""
        return np.column_stack([frame[self.features].to_numpy(dtype=np.float64)] +
                               [health[f"kf_hi_{c}"].to_numpy(dtype=np.float64) for c in nc.COMPONENTS])

    def _component_rul(self, frame: pd.DataFrame, health: pd.DataFrame) -> dict:
        """crul_<component> = flights until the component's health reaches its failure level (model output, clipped to
        [0, cap]) with empirical lo/hi; the dashboard shows it only while the EKF health is below `show_below`."""
        data = self._extended_matrix(frame, health)
        cap = float(self.comp_rul.get("cap", 100.0))
        out, intervals = {}, self.comp_rul.get("intervals", {})
        for c, m in self.comp_rul_models.items():
            pred = np.clip(np.asarray(m.predict(data), dtype=np.float64).reshape(-1), 0.0, cap)
            out[f"crul_{c}"] = pred
            iv = intervals.get(c) or {}
            if "low" in iv:
                out[f"crul_{c}_lo"] = np.clip(pred + iv["low"], 0.0, cap)
                out[f"crul_{c}_hi"] = np.clip(pred + iv["high"], 0.0, cap)
        return out

    def _forecast(self, frame: pd.DataFrame, health: pd.DataFrame) -> dict:
        """fc_<component>_h<h> = clip(current EKF health + learned change over h flights), with empirical lo/hi."""
        data = self._extended_matrix(frame, health)
        out, intervals = {}, self.forecast.get("intervals", {})
        for (c, h), m in self.forecast_models.items():
            cur = health[f"kf_hi_{c}"].to_numpy(dtype=np.float64)
            pred = np.clip(cur + np.asarray(m.predict(data), dtype=np.float64).reshape(-1), 0.0, 1.0)
            out[f"fc_{c}_h{h}"] = pred
            iv = (intervals.get(c) or {}).get(str(h)) or {}
            if "low" in iv:
                out[f"fc_{c}_h{h}_lo"] = np.clip(pred + iv["low"], 0.0, 1.0)
                out[f"fc_{c}_h{h}_hi"] = np.clip(pred + iv["high"], 0.0, 1.0)
        return out

    def predict_frame(self, frame: pd.DataFrame, health: pd.DataFrame, forecast: bool = True) -> pd.DataFrame:
        """frame: featurizer output; health: kf_hi_<component> (+ _sigma) columns aligned with frame rows."""
        base = self.base_predictions(frame)
        raw = self.stack.predict(base)
        contrib = self.stack.contributions(base)
        res = {"rul_raw": raw, "rul": np.clip(raw, 0, None), "rul_trees_only": np.clip(base.mean(axis=1), 0, None),
               "rul_spread": base.std(axis=1)}
        for i, n in enumerate(self.stack.names):
            res[f"base_{n}"] = base[:, i]
            res[f"contrib_{n}"] = contrib[:, i]
        if "low" in self.interval:                       # empirical residual quantiles (per predicted-RUL range)
            res["rul_q05"], res["rul_q95"] = st.apply_interval(res["rul"], self.interval)
        for c in nc.COMPONENTS:                          # health = EKF estimate mapped by d_ref (not an ML target)
            h = health[f"kf_hi_{c}"].to_numpy(dtype=np.float64)
            res[f"hi_{c}"] = res[f"hi_{c}_raw"] = res[f"hi_{c}_trees_only"] = h
            res[f"hi_{c}_spread"] = health[f"kf_hi_{c}_sigma"].to_numpy(dtype=np.float64)
        if forecast and self.forecast_models:
            res.update(self._forecast(frame, health))
        if forecast and self.comp_rul_models:
            res.update(self._component_rul(frame, health))
        return pd.DataFrame(res, index=frame.index)

    def run(self, cyc: pd.DataFrame) -> pd.DataFrame:
        """Per-flight table (aggregated columns) -> all layers, one row per flight (sorted by unit, cycle)."""
        df = nc.run_layers(cyc, self.ref, self.ekf, self.d_ref)
        frame = self.feat.transform(df)
        # combustor + nozzle physics monitors (first-principles, not learned; see physics_monitors.py)
        return pd.concat([df, self.predict_frame(frame, df), pm.monitor_columns(df)], axis=1)

    # ------------------------------------------------------------------ explanations
    def tree_shap(self, frame_row: pd.DataFrame) -> dict | None:
        try:
            import shap
        except ImportError:
            return None
        x = frame_row[self.model_features]
        w = self.stack.effective_weights()
        total = np.zeros(len(self.model_features))
        used = []
        for k, name in enumerate(self.stack.names):
            if w[k] == 0:
                continue
            try:
                sv = shap.TreeExplainer(self.models[name]).shap_values(x)
            except Exception:
                continue
            total += w[k] * np.asarray(sv).reshape(-1)[: len(self.model_features)]
            used.append(name)
        if not used:
            return None
        groups = pd.Series(total, index=[nc.source_of(f) for f in self.model_features]).groupby(level=0).sum()
        top_f = np.argsort(-np.abs(total))[:8]
        return {
            "method": "TreeSHAP of the base models weighted by the Ridge effective weights (" + ", ".join(used) + ")",
            "features": [{"name": self.model_features[i], "value": float(total[i])} for i in top_f],
            "groups": [{"name": g, "value": float(v)} for g, v in groups.reindex(groups.abs().sort_values(
                ascending=False).index).head(8).items()],
        }

    def contributions(self, frame_row: pd.DataFrame) -> dict:
        base = self.base_predictions(frame_row)
        c = self.stack.contributions(base)[0]
        return {"intercept": self.stack.intercept,
                "models": [{"name": n, "prediction": float(base[0, i]), "contribution": float(c[i])}
                           for i, n in enumerate(self.stack.names)],
                "note": "contribution = Ridge coefficient x scaled base prediction (not a reliability percentage)"}

    def _value(self, df_hist: pd.DataFrame, tgt: str) -> float:
        if tgt == "rul":
            frame = self.feat.transform(df_hist)
            return float(self.predict_frame(frame, df_hist, forecast=False)["rul"].iloc[-1])
        return float(df_hist[f"kf_{tgt}"].iloc[-1])

    def occlusion(self, df_engine: pd.DataFrame, pos: int, tgt: str, base_pred: float) -> dict:
        """For each sensor, pretend it read exactly the healthy-engine value for the whole history up to this flight
        (residual 0, flight summaries = healthy value, dependent physics proxies held at this engine's early-life
        median), re-run the EKF, features and models, and report how much the prediction moves."""
        hist = df_engine.iloc[: pos + 1].reset_index(drop=True)
        b = min(10, len(hist))
        rows = []
        for s in nc.XS_VARS:
            mod = hist.copy()
            mod[f"res_{s}"] = 0.0
            for pre in ("m_", "mn_", "mx_", "ls_"):
                if f"{pre}{s}" in mod:
                    mod[f"{pre}{s}"] = mod[f"ref_{s}"]
            for p, deps in nc.PROXY_DEPS.items():
                if s in deps and f"p_{p}" in mod:
                    mod[f"p_{p}"] = float(np.median(mod[f"p_{p}"].to_numpy()[:b]))
            k = self.ekf.run(mod)
            for c in k.columns:
                mod[c] = k[c].to_numpy()
            h = nc.kf_component_health(k, self.d_ref)
            for c in h.columns:
                mod[c] = h[c].to_numpy()
            rows.append({"sensor": s, "description": nc.SENSOR_INFO[s][0], "value": base_pred - self._value(mod, tgt)})
        rows.sort(key=lambda r: -abs(r["value"]))
        return {"method": "Occlusion to healthy-engine values through the whole pipeline (EKF, features, models); "
                          "value = prediction minus prediction-if-this-sensor-were-healthy", "sensors": rows}

    def explain(self, df_engine: pd.DataFrame, cycle: int, tgt: str) -> dict:
        if tgt not in self.targets:
            raise ValueError(f"Unknown target {tgt}; choose from {self.targets}")
        df_engine = df_engine.sort_values("cycle").reset_index(drop=True)
        hits = np.flatnonzero(df_engine["cycle"].to_numpy() == cycle)
        if len(hits) == 0:
            raise KeyError(cycle)
        pos = int(hits[0])
        hist = df_engine.iloc[: pos + 1]
        pred = self._value(hist, tgt)
        out = {"target": tgt, "cycle": int(cycle), "prediction": pred,
               "source": "four-model Ridge stack" if tgt == "rul" else "EKF health estimate (physics, not ML)",
               "tree_shap": None, "contributions": None,
               "occlusion": self.occlusion(df_engine, pos, tgt, pred)}
        if tgt == "rul":
            frame = self.feat.transform(hist).iloc[[-1]]
            out["tree_shap"] = self.tree_shap(frame)
            out["contributions"] = self.contributions(frame)
        return out

    # ------------------------------------------------------------------ info
    def info(self) -> dict:
        m = self.manifest
        return {
            "name": "N-CMAPSS physics-informed EKF + stacking ensemble (pipeline.md)", "ml": True,
            "dataset": m.get("dataset"), "run": m.get("run"), "files": m.get("files"), "targets": self.targets,
            "health_source": "EKF parameter estimates mapped by per-component failure-level damage (d_ref)",
            "states": self.ekf.state_names, "base_models": self.stack.names,
            "blend": {"alpha": self.stack.alpha, "intercept": self.stack.intercept,
                      "coef": dict(zip(self.stack.names, self.stack.coef.tolist())),
                      "effective_weight": dict(zip(self.stack.names, self.stack.effective_weights().tolist()))},
            "ekf_in_rul": (m.get("metrics") or {}).get("ekf_in_rul"),
            "interval": self.interval, "d_ref": self.d_ref, "metrics": m.get("metrics"),
            "reliability": m.get("reliability"), "observability": m.get("observability"),
            "splits": {k: len(v) for k, v in (m.get("splits") or {}).items()},
            "forecast": {k: self.forecast.get(k) for k in ("horizons", "target", "model", "useful", "metrics")}
            if self.forecast else None,
            "component_rul": {k: self.comp_rul.get(k) for k in ("level", "cap", "show_below", "useful", "metrics", "target")}
            if self.comp_rul else None,
            "fingerprints": m.get("dataset_fingerprints"), "build": m.get("build"),
        }
