"""Layer 4 at serving time: load the bundle exported by training/ncmapss_colab_training.ipynb and run
Layers 2-4 on per-cycle engine data (+ 0.1 Hz in-flight samples for the CNN), with TreeSHAP and occlusion
explanations.

Bundle layout (backend/models/): manifest.json, reference.json, kf.json, featurizer.json, <model>_<target>.<ext>,
cnn.onnx, lightgbm_rul_q05.txt / q95.txt, demo_cycles.csv, demo_samples.npz,
parity_outputs.csv.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import ncmapss as nc

TREE_TYPES = ("xgboost", "lightgbm", "catboost", "randomforest")


def _finite(o):
    """NaN/inf -> None so manifest metrics stay JSON-serialisable (e.g. corr of a never-degrading component)."""
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def load_tree(kind: str, path: str):
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


class OnnxCNN:
    """Multi-output 1D-CNN over 0.1 Hz in-flight windows, exported to ONNX: (batch, window, C) -> (batch, n_targets)."""

    def __init__(self, path: str, window: int, encoder: nc.SampleEncoder, outputs: list[str]):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.window, self.encoder, self.outputs = window, encoder, list(outputs)

    def predict_windows(self, w: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {"window": w.astype(np.float32)})[0]

    def cycle_predictions(self, samples, cyc, ref, occlude=None):
        return nc.cnn_cycle_predictions(self.predict_windows, self.encoder, samples, cyc, ref, self.window, occlude)


def load_samples(path: str) -> pd.DataFrame:
    z = np.load(path, allow_pickle=False)
    df = pd.DataFrame(z["values"], columns=[str(c) for c in z["columns"]])
    df.insert(0, "cycle", z["cycle"].astype(np.int64))
    df.insert(0, "unit", z["unit"].astype(np.int64))
    return df


class ModelBundle:
    def __init__(self, manifest: dict, ref: nc.HealthyReference, kf: nc.GasPathKF, feat: nc.CausalFeaturizer,
                 trees: dict, quantiles: dict | None = None, demo: pd.DataFrame | None = None,
                 cnn: OnnxCNN | None = None, demo_samples: pd.DataFrame | None = None):
        self.manifest = manifest
        self.ref, self.kf, self.feat = ref, kf, feat
        self.trees = trees                    # target -> list of (type, model) in manifest base order (trees only)
        self.cnn = cnn
        self.quantiles = quantiles or {}
        self.demo, self.demo_samples = demo, demo_samples
        self.targets = manifest["target_order"]
        self.d_ref = manifest["d_ref"]        # {component: failure-level damage} (one float in older bundles)
        self.features = manifest["features"]
        self.smoothing = manifest.get("smoothing", {})

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, model_dir: str) -> "ModelBundle":
        mpath = os.path.join(model_dir, "manifest.json")
        if not os.path.exists(mpath):
            raise FileNotFoundError(f"No manifest.json in {model_dir}: unzip the Colab bundle there.")
        with open(mpath, encoding="utf-8") as fh:
            man = _finite(json.load(fh))
        if man.get("version") != 3:
            raise ValueError(f"Unsupported bundle version {man.get('version')} (expected 3, N-CMAPSS)")

        def js(name):
            with open(os.path.join(model_dir, man[name]), encoding="utf-8") as fh:
                return json.load(fh)

        def path_of(rel):
            return os.path.join(model_dir, rel) if rel and os.path.exists(os.path.join(model_dir, rel)) else None

        ref = nc.HealthyReference.from_dict(js("reference_file"))
        kf = nc.GasPathKF.from_dict(js("kf_file"))
        feat = nc.CausalFeaturizer.from_dict(js("featurizer_file"))
        if feat.feature_names != man["features"]:
            raise ValueError("featurizer.json and manifest feature lists differ")
        trees = {}
        for tgt, spec in man["targets"].items():
            trees[tgt] = [(b["type"], load_tree(b["type"], os.path.join(model_dir, b["file"])))
                          for b in spec["base_models"] if b["type"] in TREE_TYPES]
        cnn = None
        c = man.get("cnn")
        if c and path_of(c.get("file")):
            cnn = OnnxCNN(path_of(c["file"]), c["window"], nc.SampleEncoder.from_dict(c["encoder"]), c["outputs"])
        quantiles = {}
        unc = man.get("uncertainty", {})
        for key, q in (("rul_q05_file", "q05"), ("rul_q95_file", "q95")):
            if path_of(unc.get(key)):
                quantiles[q] = load_tree("lightgbm", path_of(unc[key]))
        demo = pd.read_csv(path_of(man.get("demo_file"))) if path_of(man.get("demo_file")) else None
        demo_samples = load_samples(path_of(man.get("demo_samples_file"))) \
            if path_of(man.get("demo_samples_file")) else None
        return cls(man, ref, kf, feat, trees, quantiles, demo, cnn, demo_samples)

    # ------------------------------------------------------------------ prediction
    def _available(self, kind: str, have_cnn: bool) -> bool:
        if kind in TREE_TYPES:
            return True
        if kind == "cnn":
            return have_cnn
        return False

    def _meta(self, tgt, have_cnn: bool):
        """(blend weights, sequence bases used). Falls back to the tree-only blend when the CNN can't run (no
        cnn.onnx or no 0.1 Hz samples) - and for older bundles whose GRU is no longer supported."""
        spec = self.manifest["targets"][tgt]
        if "constant" in spec:                              # target did not vary in training
            return spec["meta"], []
        seq = [b for b in spec["base_models"] if b["type"] not in TREE_TYPES]
        if all(self._available(b["type"], have_cnn) for b in seq):
            return spec["meta"], seq
        return spec["trees_only_meta"], []

    def samples_for(self, unit) -> pd.DataFrame | None:
        if self.demo_samples is None:
            return None
        s = self.demo_samples[self.demo_samples["unit"] == int(unit)]
        return s if len(s) else None

    def cnn_cycle(self, samples: pd.DataFrame | None, df: pd.DataFrame, occlude: str | None = None):
        """Per-cycle CNN outputs aligned with df rows (sorted by unit, cycle), or None if the CNN can't run."""
        if self.cnn is None or samples is None or len(samples) == 0:
            return None
        return self.cnn.cycle_predictions(samples, df, self.ref, occlude)

    def base_predictions(self, frame: pd.DataFrame, cnn_cycle=None) -> dict:
        """target -> (n, n_bases) array, columns in manifest base order (trees, then sequence models if used)."""
        x = frame[self.features]
        have_cnn = cnn_cycle is not None
        out = {}
        for tgt in self.targets:
            cols = [np.asarray(m.predict(x), dtype=np.float64).reshape(-1) for _, m in self.trees[tgt]]
            for b in self._meta(tgt, have_cnn)[1]:
                cols.append(cnn_cycle[:, b["output_index"]])
            out[tgt] = np.column_stack(cols) if cols else np.zeros((len(frame), 0))
        return out

    @staticmethod
    def _clip(tgt, v):
        return nc.clip_target(tgt, v)

    def predict_frame(self, frame: pd.DataFrame, cnn_cycle=None) -> pd.DataFrame:
        units, cycles = frame["unit"].to_numpy(), frame["cycle_id"].to_numpy()
        if cnn_cycle is not None and not np.isfinite(cnn_cycle).all():
            cnn_cycle = None if not np.isfinite(cnn_cycle).any() else cnn_cycle
        base = self.base_predictions(frame, cnn_cycle)
        res = {}
        for tgt in self.targets:
            meta, _ = self._meta(tgt, cnn_cycle is not None)
            b = base[tgt]
            tm = self.manifest["targets"][tgt]["trees_only_meta"]
            n_tree = len(self.trees[tgt])
            trees_only = b[:, :n_tree] @ np.asarray(tm["coef"]) + tm["intercept"]
            raw = b @ np.asarray(meta["coef"]) + meta["intercept"]
            raw = np.where(np.isfinite(raw), raw, trees_only)          # cycles without 0.1 Hz samples
            a = float(self.smoothing.get(tgt, 1.0))
            res[f"{tgt}_raw"] = raw
            res[tgt] = nc.smooth_target(tgt, units, cycles, raw, a)
            res[f"{tgt}_trees_only"] = nc.smooth_target(tgt, units, cycles, trees_only, a)
            bb = self._clip(tgt, b)
            res[f"{tgt}_spread"] = np.nanstd(bb, axis=1) if b.shape[1] else np.zeros(len(b))
        x = frame[self.features]
        for q, m in self.quantiles.items():
            res[f"rul_{q}"] = np.clip(np.asarray(m.predict(x), dtype=np.float64), 0, None)
        return pd.DataFrame(res, index=frame.index)

    def run(self, cyc: pd.DataFrame, samples: pd.DataFrame | None = None) -> pd.DataFrame:
        """Per-cycle table (aggregated columns) [+ 0.1 Hz samples] -> Layers 2-4 outputs, one row per cycle."""
        df = nc.run_layers_1_to_3(cyc, self.ref, self.kf, self.d_ref)
        frame = self.feat.transform(df)
        return pd.concat([df, self.predict_frame(frame, self.cnn_cycle(samples, df))], axis=1)

    # ------------------------------------------------------------------ explanations
    def tree_shap(self, frame_row: pd.DataFrame, tgt: str, have_cnn: bool = False) -> dict | None:
        try:
            import shap
        except ImportError:
            return None
        meta, _ = self._meta(tgt, have_cnn)
        x = frame_row[self.features]
        total = np.zeros(len(self.features))
        used = []
        for k, (kind, m) in enumerate(self.trees[tgt]):
            w = float(meta["coef"][k])
            if w == 0:
                continue
            try:
                sv = shap.TreeExplainer(m).shap_values(x)
            except Exception:
                continue
            total += w * np.asarray(sv).reshape(-1)[: len(self.features)]
            used.append(kind)
        if not used:
            return None
        groups = pd.Series(total, index=[nc.source_of(f) for f in self.features]).groupby(level=0).sum()
        top_f = np.argsort(-np.abs(total))[:8]
        return {
            "method": "TreeSHAP of the tree models, weighted by the blend weights (" + ", ".join(used) + ")",
            "features": [{"name": self.features[i], "value": float(total[i])} for i in top_f],
            "groups": [{"name": g, "value": float(v)} for g, v in groups.reindex(groups.abs().sort_values(
                ascending=False).index).head(8).items()],
        }

    def occlusion(self, df_engine: pd.DataFrame, pos: int, tgt: str, base_pred: float, samples=None) -> dict:
        """Full-stack (trees + sequence model), model-agnostic: for each sensor, pretend it read exactly the
        healthy-engine value for the whole history up to this cycle (residual 0; dependent physics proxies held at
        this engine's healthy baseline; 0.1 Hz CNN inputs replaced the same way), re-run the Kalman filter, features
        and models, and report how much the prediction moves."""
        hist = df_engine.iloc[: pos + 1].reset_index(drop=True)
        b = self.feat.baseline_cycles
        rows = []
        for s in nc.XS_VARS:
            mod = hist.copy()
            mod[f"res_{s}"] = 0.0
            for p, deps in nc.PROXY_DEPS.items():
                if s in deps and f"p_{p}" in mod:
                    mod[f"p_{p}"] = float(np.median(mod[f"p_{p}"].to_numpy()[:b]))
            k = self.kf.run(mod)
            for c in k.columns:
                mod[c] = k[c].to_numpy()
            frame = self.feat.transform(mod)
            pred = float(self.predict_frame(frame, self.cnn_cycle(samples, mod, occlude=s))[tgt].iloc[-1])
            rows.append({"sensor": s, "description": nc.SENSOR_INFO[s][0], "value": base_pred - pred})
        rows.sort(key=lambda r: -abs(r["value"]))
        return {"method": "Occlusion to healthy-engine values, full stack incl. sequence model and Kalman filter "
                          "(value = prediction minus prediction-if-this-sensor-were-healthy)", "sensors": rows}

    def explain(self, df_engine: pd.DataFrame, cycle: int, tgt: str, samples: pd.DataFrame | None = None) -> dict:
        if tgt not in self.targets:
            raise ValueError(f"Unknown target {tgt}; choose from {self.targets}")
        df_engine = df_engine.sort_values("cycle").reset_index(drop=True)
        hits = np.flatnonzero(df_engine["cycle"].to_numpy() == cycle)
        if len(hits) == 0:
            raise KeyError(cycle)
        pos = int(hits[0])
        hist = df_engine.iloc[: pos + 1]
        frame = self.feat.transform(hist)
        cc = self.cnn_cycle(samples, hist.reset_index(drop=True))
        pred = float(self.predict_frame(frame, cc)[tgt].iloc[-1])
        return {"target": tgt, "cycle": int(cycle), "prediction": pred,
                "tree_shap": self.tree_shap(frame.iloc[[-1]], tgt, cc is not None),
                "occlusion": self.occlusion(df_engine, pos, tgt, pred, samples)}

    # ------------------------------------------------------------------ info
    def info(self) -> dict:
        m = self.manifest
        have_cnn = self.cnn is not None
        weights = {}
        for t in self.targets:
            meta, seq = self._meta(t, have_cnn)
            names = [k for k, _ in self.trees[t]] + [b["type"] for b in seq]
            weights[t] = dict(zip(names, meta["coef"]))
        return {
            "name": "N-CMAPSS physics-informed stack", "ml": True, "dataset": m.get("dataset"), "run": m.get("run"),
            "files": m.get("files"), "targets": self.targets, "cnn_used": have_cnn,
            "blend_weights": weights, "model_choice": m.get("model_choice"), "smoothing": self.smoothing,
            "d_ref": self.d_ref, "metrics": m.get("metrics"), "kf_quality_test": m.get("kf_quality_test"),
            "label_summary": m.get("label_summary"),
        }
