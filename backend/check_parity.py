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
    out = b.run(truth_columns(b.demo))
    m = ref.merge(out, on=["unit", "cycle"], suffixes=("_nb", ""))
    if len(m) != len(ref):
        print(f"FAIL: matched {len(m)} of {len(ref)} rows")
        return 1
    checks = [("pred_rul", "rul", 100.0)] + [(f"pred_base_{n}", f"base_{n}", 100.0) for n in b.stack.names] + \
        [(f"kf_hi_{c}", f"kf_hi_{c}", 1.0) for c in b.manifest["components"]]
    for key in sorted(c for c in ref.columns if c.startswith("pred_fc_")):        # health forecasts, if bundled
        checks.append((key, key.replace("pred_", "", 1), 1.0))
    for key in sorted(c for c in ref.columns if c.startswith("pred_crul_")):      # component RUL, if bundled
        checks.append((key, key.replace("pred_", "", 1), 100.0))
    worst = 0.0
    for col_nb, col, scale in checks:
        a = m[col_nb if col_nb != col else f"{col}_nb"]
        d = float(np.max(np.abs(a - m[col])))
        worst = max(worst, d / scale)
        print(f"{col_nb:28s} max |notebook - backend| = {d:.2e}")
    ok = worst < 1e-3
    print("PARITY", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
