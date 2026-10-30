"""Build training/ncmapss_colab_training.ipynb from ncmapss_colab_training.py.

The `# %% [pipeline]` cell is replaced by the full source of backend/pipeline/ncmapss.py, so the notebook is
self-contained on Colab (no repo upload) while sharing exactly the same code as the backend.

    python training/build_notebook.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "ncmapss_colab_training.py")
MODULE = os.path.join(HERE, "..", "backend", "pipeline", "ncmapss.py")
OUT = os.path.join(HERE, "ncmapss_colab_training.ipynb")

text = open(SRC, encoding="utf-8").read()
module_src = open(MODULE, encoding="utf-8").read()
cells = []
for chunk in re.split(r"^# %%", text, flags=re.M)[1:]:
    header, _, body = chunk.partition("\n")
    body = body.strip("\n")
    kind = header.strip()
    if kind == "[pipeline]":
        body = module_src.strip("\n")
    if kind == "[markdown]":
        md = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
        cells.append({"cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": body.splitlines(keepends=True)})
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1)
print("wrote", OUT, f"({len(cells)} cells)")
