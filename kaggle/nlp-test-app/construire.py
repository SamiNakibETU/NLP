#!/usr/bin/env python3
"""source.py (cellules '# %%') -> nlp-test-app.ipynb + kernel-metadata.json. Usage : python kaggle/nlp-test-app/construire.py"""
import json
import sys
from pathlib import Path

import nbformat

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent))
sys.path.insert(0, str(ICI))
from construire_notebooks import cellules  # noqa: E402

import base64
import io
import tempfile
import zipfile

import preparer_code

# La surcouche (corrections non poussées) est embarquée dans le notebook : pas de jeu de données à téléverser.
with tempfile.TemporaryDirectory() as tmp:
    dossier = preparer_code.preparer(Path(tmp), "codingod")
    tampon = io.BytesIO()
    with zipfile.ZipFile(tampon, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(dossier.rglob("*")):
            if f.is_file() and f.name != "dataset-metadata.json" and "tests" not in f.parts:
                z.write(f, f.relative_to(dossier).as_posix())
cellules_nb = cellules((ICI / "source.py").read_text(encoding="utf-8"))
cellules_nb.insert(1, nbformat.v4.new_code_cell("PATCH_B64 = %r  # surcouche : speech_app, code, configs (zip, base64)"
                                                % base64.b64encode(tampon.getvalue()).decode()))
print("surcouche :", len(tampon.getvalue()) // 1024, "Ko")
nb = nbformat.v4.new_notebook(cells=cellules_nb)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbformat.write(nb, ICI / "nlp-test-app.ipynb")
meta = {"id": "codingod/nlp-test-app", "title": "nlp-test-app", "code_file": "nlp-test-app.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4"}
(ICI / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("ok", len(nb.cells), "cellules")
