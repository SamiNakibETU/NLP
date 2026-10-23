#!/usr/bin/env python3
"""Convertit kaggle/<nom>/source.py (cellules '# %%') en <nom>.ipynb et écrit kernel-metadata.json.

    python kaggle/construire_notebooks.py [--utilisateur codingod]
"""
import argparse
import json
from pathlib import Path

import nbformat

DOSSIER = Path(__file__).resolve().parent
KERNELS = {
    "nlp-lora-traduction": {"titre": "nlp-lora-traduction", "datasets": ["{u}/nlp-code", "{u}/nlp-opus"], "kernels": []},
    "nlp-eval-systemes": {"titre": "nlp-eval-systemes", "datasets": ["{u}/nlp-code"], "kernels": []},
    "nlp-eval-s3-lora": {"titre": "nlp-eval-s3-lora", "datasets": ["{u}/nlp-code"],
                         "kernels": ["{u}/nlp-lora-traduction", "{u}/nlp-eval-systemes"]},
    "nlp-eval-traducteurs": {"titre": "nlp-eval-traducteurs", "datasets": ["{u}/nlp-code"],
                             "kernels": ["{u}/nlp-lora-traduction", "{u}/nlp-eval-systemes"]},
    "nlp-comet": {"titre": "nlp-comet", "datasets": [],
                  "kernels": ["{u}/nlp-lora-traduction", "{u}/nlp-eval-systemes", "{u}/nlp-eval-s3-lora", "{u}/nlp-eval-traducteurs"]},
    "nlp-eval-s3": {"titre": "nlp-eval-s3", "datasets": ["{u}/nlp-code"], "kernels": ["{u}/nlp-lora-traduction"]},
    "nlp-analyse-llm": {"titre": "nlp-analyse-llm", "datasets": ["{u}/nlp-code", "{u}/nlp-opus"], "kernels": ["{u}/nlp-lora-traduction"]},
}


def cellules(texte):
    blocs, courant, type_ = [], [], "code"
    for ligne in texte.splitlines():
        if ligne.startswith("# %%"):
            if courant:
                blocs.append((type_, "\n".join(courant).strip("\n")))
            courant, type_ = [], "markdown" if "[markdown]" in ligne else "code"
        else:
            courant.append(ligne)
    if courant:
        blocs.append((type_, "\n".join(courant).strip("\n")))
    sortie = []
    for t, c in blocs:
        if t == "markdown":
            c = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in c.splitlines())
            sortie.append(nbformat.v4.new_markdown_cell(c))
        elif c.strip():
            sortie.append(nbformat.v4.new_code_cell(c))
    return sortie


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--utilisateur", default="codingod")
    p.add_argument("noms", nargs="*")
    a = p.parse_args()
    for nom, conf in KERNELS.items():
        src = DOSSIER / nom / "source.py"
        if not src.exists() or (a.noms and nom not in a.noms):
            continue
        nb = nbformat.v4.new_notebook(cells=cellules(src.read_text(encoding="utf-8")))
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        nbformat.write(nb, DOSSIER / nom / f"{nom}.ipynb")
        meta = {"id": f"{a.utilisateur}/{nom}", "title": conf["titre"], "code_file": f"{nom}.ipynb", "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                "dataset_sources": [d.format(u=a.utilisateur) for d in conf["datasets"]], "competition_sources": [],
                "kernel_sources": [k.format(u=a.utilisateur) for k in conf["kernels"]], "model_sources": [],
                "machine_shape": "NvidiaTeslaT4"}
        (DOSSIER / nom / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("ok", nom, len(nb.cells), "cellules")
