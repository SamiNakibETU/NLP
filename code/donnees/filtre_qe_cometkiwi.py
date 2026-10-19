#!/usr/bin/env python3
"""Filtre 7 de l'étape D4 : filtrage qualité par COMETKiwi.

Ce script est conçu pour s'exécuter sur Kaggle avec accélération GPU
sur le Jeu D (ou sur le Jeu P).

Modèle de Quality Estimation (QE sans référence) :
  Unbabel/wmt22-cometkiwi-da (accès Hugging Face requis)

Règle du plan (D4 filtre 7) :
  "On retire les 20 % de segments les moins bien notés (proposé)."
  Le pourcentage est lu depuis seuils.yaml (qe_rejet_pire_pct: 20).

Le script dispose :
- d'une interface en ligne de commande avec --help
- d'un mode --essai (échantillon réduit)
- d'un mode --mode-factice (simulation sans GPU/dépendance lourde pour tests locaux)
- d'une sortie des rejets vers rejets.csv avec identifiant, filtre, motif, valeur_mesuree, seuil_applique
- d'un contrôle d'intégrité : entrée == gardées + rejetées.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

from seuils import charger_seuils


def scoreur_factice(paires):
    """Génère des scores déterministes factices pour les tests sans GPU/comet."""
    scores = []
    for ar, fr in paires:
        # Score déterministe basé sur les longueurs et caractères
        base = 0.75 + 0.1 * math.sin(len(ar) + len(fr))
        scores.append(round(base, 4))
    return scores


def charger_modele_comet():
    """Charge le modèle COMETKiwi si la bibliothèque comet est installée."""
    try:
        from comet import download_model, load_from_checkpoint

        chemin_modele = download_model("Unbabel/wmt22-cometkiwi-da")
        modele = load_from_checkpoint(chemin_modele)
        return modele
    except Exception as e:
        raise RuntimeError(
            f"Impossible de charger COMETKiwi ({e}). "
            "Utilisez --mode-factice pour exécuter les tests sans GPU/comet."
        )


def calculer_scores_comet(modele, paires, taille_lot=16):
    """Calcule les scores QE avec le modèle COMET officiel."""
    donnees = [{"src": ar, "mt": fr} for ar, fr in paires]
    sorties = modele.predict(donnees, batch_size=taille_lot, gpus=1)
    return sorties.scores


def appliquer_filtre_qe(
    fichier_entree,
    fichier_sortie,
    fichier_rejets,
    pourcentage_rejet_pire,
    mode_factice=False,
    essai=False,
    max_essai=50,
):
    """Applique le filtre COMETKiwi en retirant les X % pires segments."""
    fichier_entree = Path(fichier_entree)
    fichier_sortie = Path(fichier_sortie)
    fichier_rejets = Path(fichier_rejets)

    fichier_sortie.parent.mkdir(parents=True, exist_ok=True)
    fichier_rejets.parent.mkdir(parents=True, exist_ok=True)

    lignes = []
    with open(fichier_entree, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if essai and idx >= max_essai:
                break
            lignes.append(json.loads(line))

    total_entree = len(lignes)
    if total_entree == 0:
        raise ValueError(f"Fichier d'entrée vide : {fichier_entree}")

    paires = [(item.get("ar", item.get("ar_manuel", "")), item["fr"]) for item in lignes]

    # Calcul des scores QE
    if mode_factice:
        scores = scoreur_factice(paires)
    else:
        modele = charger_modele_comet()
        scores = calculer_scores_comet(modele, paires)

    # Attribution du score à chaque élément
    for item, s in zip(lignes, scores):
        item["score_qe"] = s

    # Calcul du seuil percentile (ex. 20% pires)
    scores_tries = sorted(scores)
    # L'index du seuil : les k éléments les plus bas sont rejetés
    nb_rejets = math.floor(total_entree * (pourcentage_rejet_pire / 100.0))
    if nb_rejets > 0:
        seuil_coupure = scores_tries[nb_rejets - 1]
    else:
        seuil_coupure = float("-inf")

    # Séparation gardés vs rejetés
    # En cas d'égalités au seuil exact, on garantit exactement le rejet des pires
    elements_avec_rang = sorted(enumerate(lignes), key=lambda x: (x[1]["score_qe"], x[0]))
    indices_rejetes = set(idx for idx, _ in elements_avec_rang[:nb_rejets])

    gardees = []
    rejetees = []

    for idx, item in enumerate(lignes):
        identifiant = item.get("id", f"ligne-{idx+1}")
        if idx in indices_rejetes:
            rejetees.append(
                {
                    "identifiant": identifiant,
                    "filtre": "D4_7_cometkiwi",
                    "motif": "qe_inferieur_au_seuil",
                    "valeur_mesuree": f"{item['score_qe']:.4f}",
                    "seuil_applique": f"pire_{pourcentage_rejet_pire}pct (>= {seuil_coupure:.4f})",
                }
            )
        else:
            gardees.append(item)

    # Écriture des fiches gardées
    with open(fichier_sortie, "w", encoding="utf-8") as f_out:
        for item in gardees:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Écriture des rejets
    with open(fichier_rejets, "w", encoding="utf-8", newline="") as f_rej:
        writer = csv.writer(f_rej, lineterminator="\n")
        writer.writerow(["identifiant", "filtre", "motif", "valeur_mesuree", "seuil_applique"])
        for r in rejetees:
            writer.writerow(
                [
                    r["identifiant"],
                    r["filtre"],
                    r["motif"],
                    r["valeur_mesuree"],
                    r["seuil_applique"],
                ]
            )

    assert total_entree == len(gardees) + len(rejetees), "Incohérence conservation des lignes"

    return {
        "total_entree": total_entree,
        "total_gardees": len(gardees),
        "total_rejetees": len(rejetees),
        "seuil_coupure": seuil_coupure,
        "pourcentage_rejet": pourcentage_rejet_pire,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filtre 7 D4 : Qualité COMETKiwi (retrait des 20%% pires segments)."
    )
    parser.add_argument("--entree", required=True, help="Fichier JSONL d'entrée.")
    parser.add_argument("--sortie", required=True, help="Fichier JSONL de sortie (gardés).")
    parser.add_argument("--rejets", required=True, help="Fichier CSV de sortie (rejets).")
    parser.add_argument(
        "--mode-factice",
        action="store_true",
        help="Utilise un scoreur factice déterministe (sans GPU ni dépendance comet).",
    )
    parser.add_argument(
        "--essai",
        action="store_true",
        help="Mode essai : traite un échantillon de 50 lignes.",
    )

    args = parser.parse_args()

    seuils = charger_seuils()
    pct_rejet = seuils["proposes"]["qe_rejet_pire_pct"]

    print("=== Filtre D4-7 COMETKiwi ===")
    print(f"Entrée : {args.entree}")
    print(f"Règle seuils.yaml : retrait des {pct_rejet}% pires segments")
    print(f"Mode factice : {args.mode_factice}")

    t0 = time.time()
    res = appliquer_filtre_qe(
        fichier_entree=args.entree,
        fichier_sortie=args.sortie,
        fichier_rejets=args.rejets,
        pourcentage_rejet_pire=pct_rejet,
        mode_factice=args.mode_factice,
        essai=args.essai,
    )
    duree = time.time() - t0

    print(f"\nTraitement terminé en {duree:.2f} s.")
    print(f"Total entrée   : {res['total_entree']}")
    print(f"Total gardées  : {res['total_gardees']}")
    print(f"Total rejetées : {res['total_rejetees']}")
    print(f"Seuil de coupure QE : {res['seuil_coupure']}")


if __name__ == "__main__":
    main()
