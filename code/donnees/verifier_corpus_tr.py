#!/usr/bin/env python3
"""Vérificateur d'intégrité et de conformité du corpus de test T et R (80 fiches).

Règles vérifiées sur le corpus :
1. Champs complets et conformes au schéma SCHEMA_TR (aucune erreur de validation).
2. Nombre d'extraits : 60 pour T et 20 pour R (80 au total).
3. Durée de chaque extrait comprise entre 20 et 30 s inclus (seuils.yaml).
4. Date de diffusion postérieure au 1er octobre 2026 (date_min_extrait: 2026-10-02).
5. Trois types d'extraits à parts égales (plateau, reportage, debat).
6. Au moins quatre chaînes de télévision différentes.
7. Deux extraits au maximum par émission.
8. Aucune émission commune entre T et R.
9. Exactement 15 fiches avec deuxième traduction indépendante (fr_2 non vide).
10. Calcul et affichage de l'empreinte SHA256 des fichiers figés.
"""

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schemas import SCHEMA_TR
from seuils import charger_seuils
from validation import valider_fiche


def calculer_sha256(chemin):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def charger_fiches(chemin):
    fiches = []
    with open(chemin, "r", encoding="utf-8") as f:
        for num, ligne in enumerate(f, start=1):
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                fiche = json.loads(ligne)
                fiches.append((num, fiche))
            except json.JSONDecodeError as exc:
                fiches.append((num, {"_json_erreur": str(exc)}))
    return fiches


def verifier_corpus_tr(chemin_t, chemin_r, seuils=None):
    """Vérifie l'ensemble des 80 fiches T et R contre les 9 règles du plan."""
    if seuils is None:
        seuils = charger_seuils()

    c = seuils["constantes"]
    nb_cible_t = c["extraits_t"]
    nb_cible_r = c["extraits_r"]
    duree_min = c["duree_extrait_min_s"]
    duree_max = c["duree_extrait_max_s"]
    date_min = date.fromisoformat(c["date_min_extrait"])
    nb_fr2_cible = c["extraits_avec_fr2"]

    fiches_t_raw = charger_fiches(chemin_t)
    fiches_r_raw = charger_fiches(chemin_r)

    fiches_t = [f for _, f in fiches_t_raw]
    fiches_r = [f for _, f in fiches_r_raw]
    toutes_fiches = fiches_t + fiches_r

    violations = []

    # 1. Nombre d'extraits
    if len(fiches_t) != nb_cible_t:
        violations.append(
            (
                "regle_nombre_extraits",
                f"Jeu T : {len(fiches_t)} extraits trouvés (attendu : {nb_cible_t})",
            )
        )
    if len(fiches_r) != nb_cible_r:
        violations.append(
            (
                "regle_nombre_extraits",
                f"Jeu R : {len(fiches_r)} extraits trouvés (attendu : {nb_cible_r})",
            )
        )

    # 2. Validation unitaire de schéma
    for source_nom, raw_list in [("T", fiches_t_raw), ("R", fiches_r_raw)]:
        for num, fiche in raw_list:
            if "_json_erreur" in fiche:
                violations.append(
                    ("regle_format_json", f"{source_nom} ligne {num} : JSON invalide ({fiche['_json_erreur']})")
                )
                continue
            errs, _ = valider_fiche(fiche, "tr", seuils)
            for code, msg in errs:
                violations.append(
                    ("regle_validation_schema", f"{source_nom} ligne {num} ({fiche.get('id')}) : {code} - {msg}")
                )

    # 3. Durée entre 20 et 30 s
    for source_nom, raw_list in [("T", fiches_t_raw), ("R", fiches_r_raw)]:
        for num, fiche in raw_list:
            debut = fiche.get("debut")
            fin = fiche.get("fin")
            if isinstance(debut, (int, float)) and isinstance(fin, (int, float)):
                duree = fin - debut
                if not (duree_min <= duree <= duree_max):
                    violations.append(
                        (
                            "regle_duree_hors_fenetre",
                            f"{source_nom} {fiche.get('id')} : durée {duree:.1f} s hors fenêtre [{duree_min}, {duree_max}]",
                        )
                    )

    # 4. Dates de diffusion
    for source_nom, raw_list in [("T", fiches_t_raw), ("R", fiches_r_raw)]:
        for num, fiche in raw_list:
            date_str = fiche.get("date")
            if isinstance(date_str, str):
                try:
                    d_val = date.fromisoformat(date_str)
                    if d_val < date_min:
                        violations.append(
                            (
                                "regle_date_avant_minimum",
                                f"{source_nom} {fiche.get('id')} : date {date_str} antérieure au minimum ({date_min})",
                            )
                        )
                except ValueError:
                    pass

    # 5. Trois types à parts égales
    # Dans T (60 fiches -> 20 de chaque) et globalement si 80
    compteur_types_t = Counter(f.get("type") for f in fiches_t if f.get("type"))
    for t_nom in ["plateau", "reportage", "debat"]:
        if len(fiches_t) == nb_cible_t and compteur_types_t.get(t_nom, 0) != nb_cible_t // 3:
            violations.append(
                (
                    "regle_types_non_equilibres",
                    f"Jeu T : type '{t_nom}' compte {compteur_types_t.get(t_nom, 0)} extraits (attendu : {nb_cible_t // 3})",
                )
            )

    compteur_types_total = Counter(f.get("type") for f in toutes_fiches if f.get("type"))
    # Si jeux non complets, vérification de parité globale
    if len(toutes_fiches) == (nb_cible_t + nb_cible_r):
        c_min = min(compteur_types_total.get(t, 0) for t in ["plateau", "reportage", "debat"])
        c_max = max(compteur_types_total.get(t, 0) for t in ["plateau", "reportage", "debat"])
        if c_max - c_min > 2:
            violations.append(
                (
                    "regle_types_non_equilibres",
                    f"Déséquilibre global des types : {dict(compteur_types_total)}",
                )
            )

    # 6. Au moins quatre chaînes différentes
    chaines_distinctes = set(f.get("chaine") for f in toutes_fiches if f.get("chaine"))
    if len(chaines_distinctes) < 4:
        violations.append(
            (
                "regle_chaines_insuffisantes",
                f"{len(chaines_distinctes)} chaînes distinctes trouvées ({chaines_distinctes}), attendu : au moins 4",
            )
        )

    # 7. Deux extraits au maximum par émission
    emissions_compteur = Counter(f.get("emission") for f in toutes_fiches if f.get("emission"))
    for em, count in emissions_compteur.items():
        if count > 2:
            violations.append(
                (
                    "regle_max_extraits_par_emission",
                    f"L'émission '{em}' compte {count} extraits (maximum autorisé : 2)",
                )
            )

    # 8. Aucune émission commune à T et R
    emissions_t = set(f.get("emission") for f in fiches_t if f.get("emission"))
    emissions_r = set(f.get("emission") for f in fiches_r if f.get("emission"))
    communes = emissions_t & emissions_r
    if communes:
        violations.append(
            (
                "regle_emission_commune_tr",
                f"Émissions présentes à la fois dans T et R : {', '.join(communes)}",
            )
        )

    # 9. Exactement 15 fiches avec deuxième traduction fr_2
    nb_fr2 = sum(1 for f in toutes_fiches if f.get("fr_2") and str(f.get("fr_2")).strip())
    if nb_fr2 != nb_fr2_cible:
        violations.append(
            (
                "regle_nb_fr2_incorrect",
                f"{nb_fr2} fiches avec deuxième traduction fr_2 (attendu : exactement {nb_fr2_cible})",
            )
        )

    # 10. Calcul des empreintes SHA256
    sha_t = calculer_sha256(chemin_t) if Path(chemin_t).exists() else None
    sha_r = calculer_sha256(chemin_r) if Path(chemin_r).exists() else None

    return {
        "valide": len(violations) == 0,
        "violations": violations,
        "nb_t": len(fiches_t),
        "nb_r": len(fiches_r),
        "nb_fr2": nb_fr2,
        "chaines": list(chaines_distinctes),
        "sha256_t": sha_t,
        "sha256_r": sha_r,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Vérifier la conformité des 80 fiches d'annotation des jeux T et R."
    )
    parser.add_argument("--t", required=True, help="Chemin du fichier t.jsonl.")
    parser.add_argument("--r", required=True, help="Chemin du fichier r.jsonl.")
    args = parser.parse_args()

    seuils = charger_seuils()
    print("=== Vérification du corpus d'extraits de test T et R ===")
    print(f"Fichier T : {args.t}")
    print(f"Fichier R : {args.r}")

    res = verifier_corpus_tr(args.t, args.r, seuils)

    print(f"\nExtraits T : {res['nb_t']} / {seuils['constantes']['extraits_t']}")
    print(f"Extraits R : {res['nb_r']} / {seuils['constantes']['extraits_r']}")
    print(f"Chaînes    : {len(res['chaines'])} distinctes ({', '.join(res['chaines'])})")
    print(f"Fiches fr2 : {res['nb_fr2']} / {seuils['constantes']['extraits_avec_fr2']}")
    print(f"SHA256 T   : {res['sha256_t']}")
    print(f"SHA256 R   : {res['sha256_r']}")

    if res["valide"]:
        print("\n[SUCCÈS] Toutes les règles du plan sont parfaitement respectées.")
        sys.exit(0)
    else:
        print(f"\n[ÉCHEC] {len(res['violations'])} violation(s) détectée(s) :")
        for code, msg in res["violations"]:
            print(f"  - [{code}] {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
