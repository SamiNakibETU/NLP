"""Tests unitaires pour l'extraction et la validation des noms propres (Jeu Noms).

Vérifie :
- La structure et les colonnes de noms_v1.csv
- L'absence de doublons d'identifiants Wikidata
- Les types autorisés : personne, organisation, lieu
- La conformité au seuil cible [300, 500] entrées
- La présence et validité du modèle d'accueil manuel
"""

import csv
from pathlib import Path
import sys
import unittest

DOSSIER_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOSSIER_CODE))

from seuils import charger_seuils

DOSSIER_NOMS = Path(__file__).resolve().parents[3] / "donnees" / "noms"


class TestNomsPropres(unittest.TestCase):
    def setUp(self):
        self.seuils = charger_seuils()
        self.noms_min = self.seuils["constantes"]["noms_min"]
        self.noms_max = self.seuils["constantes"]["noms_max"]

    def test_modele_manuel_structure(self):
        chemin_modele = DOSSIER_NOMS / "noms_manuel_modele.csv"
        if not chemin_modele.exists():
            self.skipTest("noms_manuel_modele.csv pas encore généré")
        with open(chemin_modele, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(
                reader.fieldnames,
                ["forme_arabe", "variantes", "forme_francaise", "type", "id_wikidata"],
            )

    def test_noms_v1_validite_si_present(self):
        chemin_csv = DOSSIER_NOMS / "noms_v1.csv"
        if not chemin_csv.exists():
            self.skipTest("noms_v1.csv pas encore généré")

        with open(chemin_csv, "r", encoding="utf-8") as f:
            lignes = list(csv.DictReader(f))

        # 1. Nombre de lignes dans [300, 500]
        nb = len(lignes)
        self.assertGreaterEqual(nb, self.noms_min, f"Trop peu d'entrées : {nb} < {self.noms_min}")
        self.assertLessEqual(nb, self.noms_max, f"Trop d'entrées : {nb} > {self.noms_max}")

        # 2. Aucun doublon d'identifiant Wikidata
        qids = [r["id_wikidata"] for r in lignes]
        self.assertEqual(len(qids), len(set(qids)), "Présence de doublons id_wikidata dans noms_v1.csv")

        # 3. Champs obligatoires non vides
        for idx, r in enumerate(lignes, start=1):
            self.assertTrue(r["forme_arabe"].strip(), f"Ligne {idx} : forme_arabe vide")
            self.assertTrue(r["forme_francaise"].strip(), f"Ligne {idx} : forme_francaise vide")
            self.assertIn(
                r["type"],
                ["personne", "organisation", "lieu"],
                f"Ligne {idx} : type invalide '{r['type']}'",
            )
            self.assertTrue(r["id_wikidata"].startswith("Q"), f"Ligne {idx} : id_wikidata invalide '{r['id_wikidata']}'")


if __name__ == "__main__":
    unittest.main()
