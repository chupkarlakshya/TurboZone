"""Build training/ncmapss_colab_training.ipynb from ncmapss_colab_training.py.

Each `# %% [pipeline:<name>]` cell is replaced by the full source of backend/pipeline/<name>.py, so the notebook is
self-contained on Colab (no repo upload) while sharing exactly the same code as the backend. The build is stamped
into BUILD_INFO (git commit + a hash of the script and the shared modules), which ends up in the bundle manifest.

    python training/build_notebook.py
"""
import hashlib
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ncmapss_colab_training.py")
PIPELINE = os.path.join(HERE, "..", "backend", "pipeline")
OUT = os.path.join(HERE, "ncmapss_colab_training.ipynb")


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


text = read(SRC)
modules = {name: read(os.path.join(PIPELINE, f"{name}.py"))
           for name in re.findall(r"^# %% \[pipeline:(\w+)\]", text, flags=re.M)}
code_sha = hashlib.sha256((text + "".join(modules[k] for k in sorted(modules))).encode("utf-8")).hexdigest()
try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True,
                            timeout=10).stdout.strip() or None
except Exception:
    commit = None
text, n = re.subn(r"^BUILD_INFO = \{.*?\}", f"BUILD_INFO = {{'commit': {commit!r}, 'code_sha256': {code_sha!r}}}",
                  text, count=1, flags=re.M)
assert n == 1, "BUILD_INFO line not found in the training script"

cells = []
for chunk in re.split(r"^# %%", text, flags=re.M)[1:]:
    header, _, body = chunk.partition("\n")
    body = body.strip("\n")
    kind = header.strip()
    m = re.fullmatch(r"\[pipeline:(\w+)\]", kind)
    if m:
        body = modules[m.group(1)].strip("\n")
    if kind == "[markdown]":
        md = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
        cells.append({"cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": body.splitlines(keepends=True)})
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", OUT, f"({len(cells)} cells, modules: {', '.join(modules)})")
