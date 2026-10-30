"""Serve the dashboard with the STUB bundle (synthetic data, fixed formulas) - for UI checks only.

    python -m backend.tests.serve_stub [port]
"""
import sys

import uvicorn

from backend import app as appmod
from backend.tests.stub_bundle import stub_bundle

if __name__ == "__main__":
    appmod.set_bundle(stub_bundle())
    uvicorn.run(appmod.app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8001)
