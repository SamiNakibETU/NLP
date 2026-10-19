"""Tests unitaires des filtres de l'étape D4 (presse Jeu P et COMETKiwi).

Vérifie :
- Filtre 2 (langue) : acceptation du français, rejet d'autres langues, arabe résiduel, texte vide
- Filtre 3 (ratio de longueur) : acceptation dans [0.6, 2.5], rejet hors bornes
- Filtre 6 (doublons) : rejet des doublons exacts
- Conservation stricte : entrée == gardées + rejetées
- Filtre 7 (COMETKiwi) : retrait des 20 % pires segments (mode factice)
"""

import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

DOSSIER_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOSSIER_CODE))

from filtrer_presse import (
    verifier_filtre_langue,
    verifier_filtre_ratio,
    filtrer_corpus,
)
from filtre_qe_cometkiwi import appliquer_filtre_qe
from seuils import charger_seuils


class TestFiltresD4(unittest.TestCase):
    def setUp(self):
        self.seuils = charger_seuils()
        self.r_min = self.seuils["proposes"]["rapport_longueur_min"]
        self.r_max = self.seuils["proposes"]["rapport_longueur_max"]
        self.qe_pct = self.seuils["proposes"]["qe_rejet_pire_pct"]

    def test_filtre_langue_francais_valide(self):
        ok, motif, val, seuil = verifier_filtre_langue("Le président a prononcé un discours hier soir.")
        self.assertTrue(ok)
        self.assertIsNone(motif)
        self.assertEqual(val, "fr")

    def test_filtre_langue_non_francais(self):
        ok, motif, val, seuil = verifier_filtre_langue("This is clearly an English sentence with no French words.")
        self.assertFalse(ok)
        self.assertEqual(motif, "langue_non_francaise")
        self.assertNotEqual(val, "fr")

    def test_filtre_langue_arabe_residuel(self):
        ok, motif, val, seuil = verifier_filtre_langue("Voici la déclaration : قال الوزير")
        self.assertFalse(ok)
        self.assertEqual(motif, "arabe_dans_sortie")

    def test_filtre_langue_vide(self):
        ok, motif, val, seuil = verifier_filtre_langue("   ")
        self.assertFalse(ok)
        self.assertEqual(motif, "texte_vide")

    def test_filtre_ratio_valide(self):
        ar = "قال الوزير إن المفاوضات مستمرة حتى نهاية الشهر"
        fr = "Le ministre a indiqué que les négociations se poursuivraient jusqu'à la fin du mois."
        ok, motif, val, seuil = verifier_filtre_ratio(ar, fr, self.r_min, self.r_max)
        self.assertTrue(ok)
        self.assertIsNone(motif)

    def test_filtre_ratio_trop_court(self):
        ar = "قال الوزير إن المفاوضات مستمرة حتى نهاية الشهر في العاصمة باريس"
        fr = "Oui."
        ok, motif, val, seuil = verifier_filtre_ratio(ar, fr, self.r_min, self.r_max)
        self.assertFalse(ok)
        self.assertEqual(motif, "rapport_longueur_hors_bornes")

    def test_filtre_ratio_trop_long(self):
        ar = "نعم"
        fr = "C'est absolument certain et indiscutable selon les observateurs et spécialistes internationaux."
        ok, motif, val, seuil = verifier_filtre_ratio(ar, fr, self.r_min, self.r_max)
        self.assertFalse(ok)
        self.assertEqual(motif, "rapport_longueur_hors_bornes")

    def test_filtre_ratio_arabe_vide(self):
        ok, motif, val, seuil = verifier_filtre_ratio("", "Texte en français", self.r_min, self.r_max)
        self.assertFalse(ok)
        self.assertEqual(motif, "arabe_vide")

    def test_filtrer_corpus_conservation_et_rejets(self):
        # Création d'un dossier temporaire avec données factices
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dossier_brut = tmp_path / "brut"
            dossier_brut.mkdir()

            f_brut = dossier_brut / "factice.jsonl"
            exemples = [
                # 1. OK
                {"id_origine": 1, "source": "Factice", "ar": "قال الرئيس كلمة مهمة اليوم", "fr": "Le président a prononcé un discours important aujourd'hui."},
                # 2. Rejet langue (anglais)
                {"id_origine": 2, "source": "Factice", "ar": "قال الرئيس كلمة مهمة اليوم", "fr": "The president gave an important speech today in the capital."},
                # 3. Rejet ratio (texte francais valide mais ratio trop court)
                {"id_origine": 3, "source": "Factice", "ar": "قال الرئيس كلمة مهمة جدا اليوم في افتتاح المؤتمر السنوي العام بحضور جميع الوفود المشاركة من كافة الدول", "fr": "Oui, c'est très bien."},
                # 4. Rejet doublon exact de la ligne 1
                {"id_origine": 4, "source": "Factice", "ar": "قال الرئيس كلمة مهمة اليوم", "fr": "Le président a prononcé un discours important aujourd'hui."},
                # 5. OK
                {"id_origine": 5, "source": "Factice", "ar": "المؤتمر ينعقد غدا في تونس", "fr": "La conférence se tiendra demain à Tunis."},
            ]
            with open(f_brut, "w", encoding="utf-8") as f:
                for ex in exemples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

            f_sortie = tmp_path / "p_v1.jsonl"
            f_rejets = tmp_path / "rejets.csv"

            res = filtrer_corpus(
                dossier_brut=dossier_brut,
                fichier_sortie=f_sortie,
                fichier_rejets=f_rejets,
                ratio_min=self.r_min,
                ratio_max=self.r_max,
            )

            self.assertEqual(res["total_entree"], 5)
            self.assertEqual(res["total_gardees"], 2)
            self.assertEqual(res["total_rejetees"], 3)
            self.assertEqual(res["total_entree"], res["total_gardees"] + res["total_rejetees"])

            # Vérification du fichier rejets.csv
            with open(f_rejets, "r", encoding="utf-8") as f:
                lignes_csv = list(csv.DictReader(f))
            self.assertEqual(len(lignes_csv), 3)
            motifs = [r["motif"] for r in lignes_csv]
            self.assertIn("langue_non_francaise", motifs)
            self.assertIn("rapport_longueur_hors_bornes", motifs)
            self.assertIn("doublon_exact", motifs)

    def test_filtre_cometkiwi_factice_retrait_20pct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            f_entree = tmp_path / "entree.jsonl"
            f_sortie = tmp_path / "sortie.jsonl"
            f_rejets = tmp_path / "rejets.csv"

            # 10 fiches factices -> 20 % de 10 = 2 rejets
            lignes = [
                {"id": f"p-{i:06d}", "ar": f"نص رقم {i}", "fr": f"Texte numéro {i} en français pour test."}
                for i in range(1, 11)
            ]
            with open(f_entree, "w", encoding="utf-8") as f:
                for l in lignes:
                    f.write(json.dumps(l, ensure_ascii=False) + "\n")

            res = appliquer_filtre_qe(
                fichier_entree=f_entree,
                fichier_sortie=f_sortie,
                fichier_rejets=f_rejets,
                pourcentage_rejet_pire=self.qe_pct,
                mode_factice=True,
            )

            self.assertEqual(res["total_entree"], 10)
            self.assertEqual(res["total_gardees"], 8)
            self.assertEqual(res["total_rejetees"], 2)
            self.assertEqual(res["total_entree"], res["total_gardees"] + res["total_rejetees"])


if __name__ == "__main__":
    unittest.main()
