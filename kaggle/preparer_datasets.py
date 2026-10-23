#!/usr/bin/env python3
"""Prépare les deux jeux de données PRIVÉS à téléverser sur Kaggle (aucun secret, aucune donnée lourde).

- nlp-code : code/, configs/, kaggle/commun/kaggle_commun.py et le projecteur S3 du temps 1 (34,6 Mo).
- nlp-opus : sous-ensemble de donnees/presse/p_v1.jsonl (corpus OPUS filtré), phrases arabes de 4 à 60 mots,
  avec les deux phrases arabes précédentes du même document (champ ``ctx``).

    python kaggle/preparer_datasets.py --sortie <dossier_de_travail> [--utilisateur codingod]
"""
import argparse
import json
import random
import re
import shutil
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
MOTIF_AR = re.compile(r"[؀-ۿ]")
QUOTAS_TRAIN = {"News-Commentary": 9500, "GlobalVoices": 7500, "TED2020": 3500}  # 20 500 paires
QUOTAS_VAL = {"News-Commentary": 200, "GlobalVoices": 200, "TED2020": 100}      # 500 paires
ECART_MAX_CONTEXTE = 3   # écart max d'id_origine pour considérer deux lignes voisines dans le même document


def preparer_code(sortie: Path, utilisateur: str):
    dest = sortie / "nlp-code"
    if dest.exists():
        shutil.rmtree(dest)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "tests", ".pytest_cache")
    shutil.copytree(RACINE / "code", dest / "code", ignore=ignore)
    shutil.copytree(RACINE / "configs", dest / "configs", ignore=ignore)
    (dest / "modeles" / "s3_temps1_50k").mkdir(parents=True)
    shutil.copy2(RACINE / "modeles" / "s3_temps1_50k" / "projecteur_meilleur.pt",
                 dest / "modeles" / "s3_temps1_50k" / "projecteur_meilleur.pt")
    shutil.copy2(RACINE / "kaggle" / "commun" / "kaggle_commun.py", dest / "kaggle_commun.py")
    for motif in ("*.env", "*.pem", "*.key"):
        assert not list(dest.rglob(motif)), "secret détecté dans le jeu de données"
    (dest / "dataset-metadata.json").write_text(json.dumps(
        {"title": "nlp-code", "id": f"{utilisateur}/nlp-code", "licenses": [{"name": "other"}]}), encoding="utf-8")
    return dest


def preparer_opus(sortie: Path, utilisateur: str):
    dest = sortie / "nlp-opus"
    dest.mkdir(parents=True, exist_ok=True)
    lignes = [json.loads(l) for l in open(RACINE / "donnees" / "presse" / "p_v1.jsonl", encoding="utf-8")]
    nb_mots = lambda t: len(t.split())
    eligibles = {}
    for i, r in enumerate(lignes):
        if r["source"] not in QUOTAS_TRAIN or not (4 <= nb_mots(r["ar"]) <= 60) or MOTIF_AR.search(r["fr"]):
            continue
        eligibles.setdefault(r["source"], []).append(i)
    hasard = random.Random(42)
    choix = {}
    for source in QUOTAS_TRAIN:
        indices = eligibles[source][:]
        hasard.shuffle(indices)
        nv, nt = QUOTAS_VAL[source], QUOTAS_TRAIN[source]
        for i in indices[:nv]:
            choix[i] = "val"
        for i in indices[nv:nv + nt]:
            choix[i] = "train"

    def contexte(i):
        ctx = []
        j = i
        for _ in range(2):
            if j == 0 or lignes[j - 1]["source"] != lignes[i]["source"] \
                    or not (0 < lignes[j]["id_origine"] - lignes[j - 1]["id_origine"] <= ECART_MAX_CONTEXTE):
                break
            j -= 1
            ctx.insert(0, lignes[j]["ar"])
        return ctx

    with open(dest / "opus_ar_fr.jsonl", "w", encoding="utf-8") as f:
        for i in sorted(choix):  # ordre du fichier d'origine = ordre des documents
            r = lignes[i]
            f.write(json.dumps({"id": r["id"], "source": r["source"], "partie": choix[i], "ar": r["ar"],
                                "fr": r["fr"], "ctx": contexte(i)}, ensure_ascii=False) + "\n")
    (dest / "dataset-metadata.json").write_text(json.dumps(
        {"title": "nlp-opus", "id": f"{utilisateur}/nlp-opus", "licenses": [{"name": "other"}]}), encoding="utf-8")
    return dest, len(choix)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sortie", required=True)
    p.add_argument("--utilisateur", default="codingod")
    a = p.parse_args()
    sortie = Path(a.sortie)
    print(preparer_code(sortie, a.utilisateur))
    print(preparer_opus(sortie, a.utilisateur))
