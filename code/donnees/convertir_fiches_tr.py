#!/usr/bin/env python3
"""Convertisseur bidirectionnel de fiches d'annotation T/R (CSV <-> JSONL).

Permet aux étudiants et annotateurs de travailler dans un tableur (CSV)
et de convertir facilement leurs fiches au format d'entraînement/test (JSONL),
ou inversement.

Champs pris en charge (conformes à SCHEMA_TR) :
id, jeu, chaine, emission, date, url, debut, fin, type, dialecte, locuteurs,
ar, fr, fr_2, annotateurs
"""

import argparse
import csv
import json
from pathlib import Path
import sys

COLONNES_TR = [
    "id",
    "jeu",
    "chaine",
    "emission",
    "date",
    "url",
    "debut",
    "fin",
    "type",
    "dialecte",
    "locuteurs",
    "ar",
    "fr",
    "fr_2",
    "annotateurs",
]


def csv_vers_jsonl(chemin_csv, chemin_jsonl):
    """Convertit un fichier CSV d'annotation en JSONL typé."""
    chemin_csv = Path(chemin_csv)
    chemin_jsonl = Path(chemin_jsonl)
    chemin_jsonl.parent.mkdir(parents=True, exist_ok=True)

    nb_convertis = 0
    with open(chemin_csv, "r", encoding="utf-8-sig") as f_in, open(
        chemin_jsonl, "w", encoding="utf-8"
    ) as f_out:
        lecteur = csv.DictReader(f_in)
        for ligne in lecteur:
            fiche = {}
            for col in COLONNES_TR:
                val = ligne.get(col, "").strip() if ligne.get(col) is not None else ""

                if col in ("debut", "fin"):
                    try:
                        fiche[col] = float(val) if val else 0.0
                    except ValueError:
                        fiche[col] = val
                elif col == "locuteurs":
                    try:
                        fiche[col] = int(val) if val else 1
                    except ValueError:
                        fiche[col] = val
                elif col == "fr_2":
                    fiche[col] = val if val else None
                else:
                    fiche[col] = val

            f_out.write(json.dumps(fiche, ensure_ascii=False) + "\n")
            nb_convertis += 1

    return nb_convertis


def jsonl_vers_csv(chemin_jsonl, chemin_csv):
    """Convertit un fichier JSONL de fiches T/R en CSV éditable."""
    chemin_jsonl = Path(chemin_jsonl)
    chemin_csv = Path(chemin_csv)
    chemin_csv.parent.mkdir(parents=True, exist_ok=True)

    nb_convertis = 0
    with open(chemin_jsonl, "r", encoding="utf-8") as f_in, open(
        chemin_csv, "w", encoding="utf-8", newline=""
    ) as f_out:
        writer = csv.DictWriter(f_out, fieldnames=COLONNES_TR, lineterminator="\n")
        writer.writeheader()
        for ligne in f_in:
            ligne = ligne.strip()
            if not ligne:
                continue
            data = json.loads(ligne)
            ligne_csv = {}
            for col in COLONNES_TR:
                val = data.get(col)
                if val is None:
                    ligne_csv[col] = ""
                else:
                    ligne_csv[col] = str(val)
            writer.writerow(ligne_csv)
            nb_convertis += 1

    return nb_convertis


def main():
    parser = argparse.ArgumentParser(
        description="Convertir les fiches d'annotation T/R entre CSV et JSONL."
    )
    parser.add_argument("--source", required=True, help="Fichier d'entrée.")
    parser.add_argument("--cible", required=True, help="Fichier de sortie.")
    parser.add_argument(
        "--sens",
        choices=["csv-vers-jsonl", "jsonl-vers-csv", "auto"],
        default="auto",
        help="Sens de conversion (défaut : auto selon les extensions).",
    )
    args = parser.parse_args()

    source = Path(args.source)
    cible = Path(args.cible)
    sens = args.sens

    if sens == "auto":
        if source.suffix.lower() == ".csv" and cible.suffix.lower() == ".jsonl":
            sens = "csv-vers-jsonl"
        elif source.suffix.lower() == ".jsonl" and cible.suffix.lower() == ".csv":
            sens = "jsonl-vers-csv"
        else:
            parser.error("Impossible de déterminer le sens. Spécifiez --sens.")

    if sens == "csv-vers-jsonl":
        nb = csv_vers_jsonl(source, cible)
        print(f"[OK] {nb} fiches converties de CSV ({source}) vers JSONL ({cible}).")
    else:
        nb = jsonl_vers_csv(source, cible)
        print(f"[OK] {nb} fiches converties de JSONL ({source}) vers CSV ({cible}).")


if __name__ == "__main__":
    main()
