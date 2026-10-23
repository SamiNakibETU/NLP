#!/usr/bin/env python3
"""Prépare le jeu de données privé nlp-app-patch (surcouche légère posée sur le clone de la branche integration : speech_app, code, configs, LoRA).

    python kaggle/nlp-test-app/preparer_code.py --sortie <dossier> [--utilisateur codingod]
Puis : kaggle datasets create -p <dossier>/nlp-app-patch -r zip      (première fois)
       kaggle datasets version -p <dossier>/nlp-app-patch -r zip -m "message"   (ensuite)
"""
import argparse
import json
import os
import shutil
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
IGNORER = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "sessions", "cache", "latest_logs",
                                 "*.env", ".env", "*.pem", "*.key", "verification_reelle.json")


def preparer(sortie: Path, utilisateur: str) -> Path:
    dest = sortie / "nlp-app-patch"
    if dest.exists():
        shutil.rmtree(dest)
    for dossier in ("speech_app", "code", "configs"):
        shutil.copytree(RACINE / dossier, dest / dossier, ignore=IGNORER)
    lora = RACINE / "modeles" / "lora_traduction"          # non suivi par git : seul cas où le dépôt cloné ne suffit pas
    if (lora / "adapter_config.json").is_file() and os.environ.get("KAGGLE_AVEC_LORA") == "1":
        shutil.copytree(lora, dest / "modeles" / "lora_traduction", ignore=IGNORER)
    for motif in ("*.env", "*.pem", "*.key"):
        assert not list(dest.rglob(motif)), "secret détecté"
    (dest / "dataset-metadata.json").write_text(json.dumps(
        {"title": "nlp-app-patch", "id": f"{utilisateur}/nlp-app-patch", "licenses": [{"name": "other"}]}), encoding="utf-8")
    return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sortie", required=True)
    p.add_argument("--utilisateur", default="codingod")
    a = p.parse_args()
    print(preparer(Path(a.sortie), a.utilisateur))
