"""Assemble le dossier à déposer sur Hugging Face Spaces (édition figée : CPU gratuit, aucun modèle).

    python make_space.py --snapshot snapshot --out space

Crée space/ avec le code, style.css, l'archive et un README à en-tête Spaces. L'espace se crée ensuite sur
huggingface.co/new-space (SDK Gradio), de préférence en PRIVÉ (articles de presse), puis on y dépose le contenu de space/.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).parent
CODE = ["engines.py", "feeds.py", "textmt.py", "reader.py", "render.py", "plots.py", "clip.py", "snapshot.py", "style.css"]

README = """---
title: Le Porte-Parole
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
---

# Le Porte-Parole (édition figée)

Presse arabe de cinq journaux et vidéos sous-titrées en français par un modèle de parole, figées à la date indiquée
sur la page. Projet de recherche, usage non commercial. Les textes appartiennent aux journaux ; chaque article renvoie à
sa source. Aucun extrait du corpus MGB2 n'est inclus.
"""

WRAPPER = '''"""Point d'entrée Hugging Face Spaces : lance l'application sur l'archive figée."""
import os

os.environ.setdefault("PORTEPAROLE_SNAPSHOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshot"))

from porteparole import build_app, launch_kwargs  # noqa: E402

build_app().launch(**launch_kwargs())
'''

REQUIREMENTS = "gradio>=6.0\nmatplotlib\nnumpy\n"


def make_space(snapshot_dir: str | Path, out: str | Path) -> Path:
    snapshot_dir, out = Path(snapshot_dir), Path(out)
    if not (snapshot_dir / "edition.json").exists():
        raise FileNotFoundError(f"{snapshot_dir}/edition.json introuvable : prépare l'archive avec snapshot.py build")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in CODE:
        shutil.copy(HERE / name, out / name)
    shutil.copy(HERE / "app.py", out / "porteparole.py")     # le vrai code ; app.py est le point d'entrée Spaces
    shutil.copytree(snapshot_dir, out / "snapshot")
    (out / "app.py").write_text(WRAPPER, encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")
    (out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", default="snapshot")
    p.add_argument("--out", default="space")
    a = p.parse_args()
    out = make_space(a.snapshot, a.out)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"{out} prêt ({size:.1f} Mo, {sum(1 for f in out.rglob('*') if f.is_file())} fichiers)")


if __name__ == "__main__":
    main()
