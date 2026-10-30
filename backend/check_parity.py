"""Verify the backend reproduces the Colab notebook's predictions for the bundled test engines.

    python -m backend.check_parity            # uses backend/models (or MODEL_DIR)

Run this once after unzipping a new bundle. Differences should be ~0 (float rounding only).
"""
import os
import sys

import numpy as np
import pandas as pd

from .app import MODEL_DIR
from .pipeline.bundle import ModelBundle
from .pipeline.service import truth_columns


def main() -> int:
    b = ModelBundle.load(MODEL_DIR)
    ref = pd.read_csv(os.path.join(MODEL_DIR, b.manifest["parity_file"]))
    out = b.run(truth_columns(b.demo), b.demo_samples)
    m = ref.merge(out, on=["unit", "cycle"], suffixes=("_nb", ""))
    if len(m) != len(ref):
        print(f"FAIL: matched {len(m)} of {len(ref)} rows")
        return 1
    # full-stack columns are only reproducible when every base model of the notebook is available here
    full_ok = not b.manifest.get("gru") and not (b.manifest.get("cnn") and (b.cnn is None or b.demo_samples is None))
    worst = 0.0
    for t in b.targets:
        for col_nb, col in ((f"pred_{t}", t), (f"pred_{t}_trees_only", f"{t}_trees_only")):
            d = float(np.max(np.abs(m[col_nb] - m[col])))
            if full_ok or col_nb.endswith("_trees_only"):
                worst = max(worst, d / (100.0 if t == "rul" else 1.0))
            print(f"{col_nb:28s} max |notebook - backend| = {d:.2e}")
    for c in b.manifest["components"]:
        print(f"kf_hi_{c:24s} max diff = {np.max(np.abs(m[f'kf_hi_{c}_nb'] - m[f'kf_hi_{c}'])):.2e}")
    if b.manifest.get("gru"):
        print("NOTE: older bundle with a GRU (no longer supported) - only the trees_only columns are checked.")
    if b.manifest.get("cnn") and (b.cnn is None or b.demo_samples is None):
        print("NOTE: CNN not loaded (no cnn.onnx / demo_samples.npz) - only the trees_only columns are checked.")
    ok = worst < 1e-3
    print("PARITY", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
