#!/usr/bin/env python3
"""Générateur des grilles et fichiers de support pour les étapes humaines obligatoires :
- D2 : Comparaison à l'aveugle des traductions candidates (100 segments, colonnes A/B/C/D anonymisées) + clé secrète
- D5 : Grille de double relecture humaine qualité (200 segments, statut correcte/défaut mineur/erreur de sens)
- D6 : Lot de 100 débuts de phrase pour relecture humaine de traduisibilité
- Noms : Vérification du modèle de recueil manuel des 100 noms JT 2026

Reproductible avec graine 42.
"""

import csv
import json
import random
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
DOSSIER_DISTILLE = RACINE / "donnees" / "distille"
DOSSIER_NOMS = RACINE / "donnees" / "noms"
DOSSIER_DISTILLE.mkdir(parents=True, exist_ok=True)

GRAINE = 42
CANDIDATS = [
    "Qwen/Qwen2.5-32B-Instruct",
    "google/gemma-2-27b-it",
    "CohereLabs/aya-expanse-32b",
    "Qwen/Qwen2.5-7B-Instruct",
]


def generer_grille_d2():
    rng = random.Random(GRAINE)
    fichier_csv = DOSSIER_DISTILLE / "d2_grille_aveugle_factice.csv"
    fichier_cle = DOSSIER_DISTILLE / "d2_cle_secrete_factice.json"

    lignes = []
    cle_secrete = {}
    labels = ["A", "B", "C", "D"]

    for i in range(1, 101):
        sid = f"d-comp-{i:04d}"
        ar = f"أكد المتحدث الرسمي أن المشاورات الدبلوماسية ستتواصل في الأيام المقبلة بشأن التطورات الأخيرة ({i})"
        
        candidats_melanges = list(CANDIDATS)
        rng.shuffle(candidats_melanges)
        attribution = dict(zip(labels, candidats_melanges))
        cle_secrete[sid] = attribution

        ligne = {
            "id_segment": sid,
            "ar": ar,
            "traduction_A": f"[Simulation factice] Le porte-parole officiel a confirmé que les consultations diplomatiques se poursuivraient dans les prochains jours ({i}).",
            "traduction_B": f"[Simulation factice] Le porte-parole a affirmé la poursuite des discussions diplomatiques sur les récents développements ({i}).",
            "traduction_C": f"[Simulation factice] Selon le porte-parole officiel, les concertations diplomatiques continueront concernant la situation ({i}).",
            "traduction_D": f"[Simulation factice] Le porte-parole a déclaré que les échanges diplomatiques allaient continuer dans les jours à venir ({i}).",
            "note_juge1_A": "",
            "note_juge1_B": "",
            "note_juge1_C": "",
            "note_juge1_D": "",
            "note_juge2_A": "",
            "note_juge2_B": "",
            "note_juge2_C": "",
            "note_juge2_D": "",
            "remarques": "GRILLE FACTICE - NE PAS FOURNIR AUX ANNOTATEURS",
        }
        lignes.append(ligne)

    with open(fichier_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(lignes[0].keys()))
        writer.writeheader()
        writer.writerows(lignes)

    with open(fichier_cle, "w", encoding="utf-8") as f:
        json.dump(cle_secrete, f, indent=2, ensure_ascii=False)

    print(f"[OK] Grille D2 factice générée : {fichier_csv} (100 segments)")
    print(f"[OK] Clé secrète D2 factice enregistrée : {fichier_cle}")


def generer_grille_d5():
    fichier_csv = DOSSIER_DISTILLE / "d5_grille_controle_humain_factice.csv"
    lignes = []

    for i in range(1, 201):
        sid = f"d-{i:06d}"
        ar = f"أعلنت وزارة الخارجية عن توقيع اتفاقية تعاون اقتصادي واستثماري مع نظيرتها الفرنسية ({i})"
        fr = f"[Simulation factice] Le ministère des Affaires étrangères a annoncé la signature d'un accord ({i})."
        lignes.append({
            "id_segment": sid,
            "ar": ar,
            "fr": fr,
            "juge_1": "",  # correcte | defaut_mineur | erreur_de_sens
            "juge_2": "",  # correcte | defaut_mineur | erreur_de_sens
            "accord": "",
            "commentaire": "GRILLE FACTICE - NE PAS FOURNIR AUX ANNOTATEURS",
        })

    with open(fichier_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(lignes[0].keys()))
        writer.writeheader()
        writer.writerows(lignes)

    print(f"[OK] Grille D5 factice générée : {fichier_csv} (200 segments)")


def generer_grille_d6():
    fichier_csv = DOSSIER_DISTILLE / "d6_echantillon_relecture_factice.csv"
    lignes = []

    for i in range(1, 101):
        sid = f"d6-{i:04d}"
        ar = f"قال رئيس الوفد المفاوض إن المرحلة القادمة ستشهد خطوات عملية ملموسة"
        fr = f"[Simulation factice] Le chef de la délégation a déclaré que la prochaine étape connaîtrait des mesures."
        coupe = [30, 50, 70][i % 3]
        if coupe == 30:
            deb_ar = "قال رئيس الوفد"
            deb_fr = "Le chef de la délégation"
            trad = "oui"
        elif coupe == 50:
            deb_ar = "قال رئيس الوفد المفاوض إن المرحلة"
            deb_fr = "Le chef de la délégation"
            trad = "non"
        else:
            deb_ar = "قال رئيس الوفد المفاوض إن المرحلة القادمة ستشهد خطوات"
            deb_fr = "Le chef de la délégation de négociation a déclaré"
            trad = "oui"

        lignes.append({
            "id": sid,
            "coupe_pct": coupe,
            "ar_complet": ar,
            "fr_complet": fr,
            "debut_ar": deb_ar,
            "debut_fr": deb_fr,
            "traduisible_machine": trad,
            "validation_humaine": "",  # oui | non
            "correction_debut_fr": "",
            "commentaire": "GRILLE FACTICE - NE PAS FOURNIR AUX ANNOTATEURS",
        })

    with open(fichier_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(lignes[0].keys()))
        writer.writeheader()
        writer.writerows(lignes)

    print(f"[OK] Grille D6 factice générée : {fichier_csv} (100 exemples)")



if __name__ == "__main__":
    generer_grille_d2()
    generer_grille_d5()
    generer_grille_d6()
