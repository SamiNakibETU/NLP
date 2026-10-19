"""Tests unitaires du vérificateur de corpus de test T et R.

Vérifie que verifier_corpus_tr.py :
- Accepte un jeu factice valide de 80 fiches (60 T + 20 R) respectant toutes les contraintes.
- Détecte et signale spécifiquement chaque règle violée :
  1. Durée hors fenêtre [20, 30] s
  2. Types non équilibrés (plateau, reportage, debat)
  3. Moins de 4 chaînes distinctes
  4. Plus de 2 extraits par émission
  5. Émission commune entre T et R
  6. Nombre incorrect de secondes traductions (!= 15)
  7. Date de diffusion antérieure au 2 octobre 2026
  8. Nombre d'extraits incorrect (!= 60 pour T ou != 20 pour R)
"""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

DOSSIER_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOSSIER_CODE))

from seuils import charger_seuils
from verifier_corpus_tr import verifier_corpus_tr


def generer_corpus_factice_valide():
    """Génère 60 fiches T et 20 fiches R respectant rigoureusement toutes les règles."""
    fiches_t = []
    fiches_r = []

    chaines = ["Al Jazeera", "Al Arabiya", "France 24", "Sky News Arabia"]
    types = ["plateau", "reportage", "debat"]

    # 60 fiches pour T : 20 plateau, 20 reportage, 20 debat
    # 30 émissions distinctes (2 extraits max par émission)
    for i in range(60):
        t_idx = i // 20
        chaine = chaines[i % len(chaines)]
        num_emission = (i // 2) + 1
        emission = f"JT Soir T-{num_emission:02d}"
        fr2 = "Deuxième traduction indépendante" if i < 15 else None

        fiches_t.append(
            {
                "id": f"t-{i+1:03d}",
                "jeu": "T",
                "chaine": chaine,
                "emission": emission,
                "date": f"2026-10-{(i%20)+2:02d}",
                "url": f"https://exemple.test/t/{i+1}",
                "debut": 100.0,
                "fin": 125.0,  # 25s
                "type": types[t_idx],
                "dialecte": "aucun",
                "locuteurs": 1,
                "ar": f"نص تجريبي للتقرير رقم {i+1}",
                "fr": f"Texte de test pour le reportage numéro {i+1}.",
                "fr_2": fr2,
                "annotateurs": "AA, BB",
            }
        )

    # 20 fiches pour R : émissions R-01 à R-10 (aucun chevauchement avec T)
    for j in range(20):
        t_idx = j % 3
        chaine = chaines[j % len(chaines)]
        num_emission = (j // 2) + 1
        emission = f"Emission R-{num_emission:02d}"

        fiches_r.append(
            {
                "id": f"r-{j+1:03d}",
                "jeu": "R",
                "chaine": chaine,
                "emission": emission,
                "date": f"2026-10-{(j%20)+2:02d}",
                "url": f"https://exemple.test/r/{j+1}",
                "debut": 50.0,
                "fin": 74.0,  # 24s
                "type": types[t_idx],
                "dialecte": "aucun",
                "locuteurs": 1,
                "ar": f"نص تجريبي لضبط النموذج {j+1}",
                "fr": f"Texte de test pour le réglage numéro {j+1}.",
                "fr_2": None,
                "annotateurs": "CC, DD",
            }
        )

    return fiches_t, fiches_r


class TestVerifierCorpusTR(unittest.TestCase):
    def setUp(self):
        self.seuils = charger_seuils()

    def _ecrire_fichiers(self, tmpdir, fiches_t, fiches_r):
        tmp = Path(tmpdir)
        p_t = tmp / "t.jsonl"
        p_r = tmp / "r.jsonl"
        with open(p_t, "w", encoding="utf-8") as ft:
            for item in fiches_t:
                ft.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(p_r, "w", encoding="utf-8") as fr:
            for item in fiches_r:
                fr.write(json.dumps(item, ensure_ascii=False) + "\n")
        return p_t, p_r

    def test_corpus_valide_80_fiches(self):
        fiches_t, fiches_r = generer_corpus_factice_valide()
        with tempfile.TemporaryDirectory() as tmpdir:
            p_t, p_r = self._ecrire_fichiers(tmpdir, fiches_t, fiches_r)
            res = verifier_corpus_tr(p_t, p_r, self.seuils)
            self.assertTrue(res["valide"], f"Erreurs inattendues : {res['violations']}")
            self.assertEqual(len(res["violations"]), 0)
            self.assertEqual(res["nb_t"], 60)
            self.assertEqual(res["nb_r"], 20)
            self.assertEqual(res["nb_fr2"], 15)
            self.assertIsNotNone(res["sha256_t"])
            self.assertIsNotNone(res["sha256_r"])

    def test_regle_duree_hors_fenetre(self):
        fiches_t, fiches_r = generer_corpus_factice_valide()
        fiches_t[0]["debut"] = 0.0
        fiches_t[0]["fin"] = 15.0  # 15s < 20s
        with tempfile.TemporaryDirectory() as tmpdir:
            p_t, p_r = self._ecrire_fichiers(tmpdir, fiches_t, fiches_r)
            res = verifier_corpus_tr(p_t, p_r, self.seuils)
            self.assertFalse(res["valide"])
            codes = [v[0] for v in res["violations"]]
            self.assertIn("regle_duree_hors_fenetre", codes)

    def test_regle_types_non_equilibres(self):
        fiches_t, fiches_r = generer_corpus_factice_valide()
        # On remplace tous les 'debat' par 'plateau'
        for f in fiches_t:
            if f["type"] == "debat":
                f["type"] = "plateau"
        with tempfile.TemporaryDirectory() as tmpdir:
            p_t, p_r = self._ecrire_fichiers(tmpdir, fiches_t, fiches_r)
            res = verifier_corpus_tr(p_t, p_r, self.seuils)
            self.assertFalse(res["valide"])
            codes = [v[0] for v in res["violations"]]
            self.assertIn("regle_types_non_equilibres", codes)

    def test_regle_chaines_insuffisantes(self):
        fiches_t, fiches_r = generer_corpus_factice_valide()
        # Seulement 2 chaînes
        for f in fiches_t + fiches_r:
            f["chaine"] = "Al Jazeera" if "1" in f["id"] else "France 24"
        with tempfile.TemporaryDirectory() as tmpdir:
            p_t, p_r = self._ecrire_fichiers(tmpdir, fiches_t, fiches_r)
            res = verifier_corpus_tr(p_t, p_r, self.seuils)
            self.assertFalse(res["valide"])
            codes = [v[0] for v in res["violations"]]
            self.assertIn("regle_chaines_insuffisantes", codes)

    def test_regle_max_extraits_par_emission(self):
        fiches_t, fiches_r = generer_corpus_factice_valide()
        # 3 extraits avec la même émission
        fiches_t[0]["emission"] = "Emission Surutilisee"
        fiches_t[1]["emission"] = "Emission Surutilisee"
        fiches_t[2]["emission"] = "Emission Surutilisee"
        with tempfile.TemporaryDirectory() as tmpdir:
            p_t, p_r = self._ecrire_fichiers(tmpdir, fiches_t, fiches_r)
            res = verifier_corpus_tr(p_t, p_r, self.seuils)
            self.assertFalse(res["valide"])
            codes = [v[0] for v in res["violations"]]
            self.assertIn("regle_max_extraits_par_emission", codes)

    def test_regle_emission_commune_tr(self):
        fiches_t, fiches_r = generer_corpus_factice_valide()
        # Emission T-01 présente dans R
        fiches_r[0]["emission"] = fiches_t[0]["emission"]
        with tempfile.TemporaryDirectory() as tmpdir:
            p_t, p_r = self._ecrire_fichiers(tmpdir, fiches_t, fiches_r)
            res = verifier_corpus_tr(p_t, p_r, self.seuils)
            self.assertFalse(res["valide"])
            codes = [v[0] for v in res["violations"]]
            self.assertIn("regle_emission_commune_tr", codes)

    def test_regle_nb_fr2_incorrect(self):
        fiches_t, fiches_r = generer_corpus_factice_valide()
        # 16 fiches avec fr_2 au lieu de 15
        fiches_t[15]["fr_2"] = "Traduction en trop"
        with tempfile.TemporaryDirectory() as tmpdir:
            p_t, p_r = self._ecrire_fichiers(tmpdir, fiches_t, fiches_r)
            res = verifier_corpus_tr(p_t, p_r, self.seuils)
            self.assertFalse(res["valide"])
            codes = [v[0] for v in res["violations"]]
            self.assertIn("regle_nb_fr2_incorrect", codes)

    def test_regle_date_avant_minimum(self):
        fiches_t, fiches_r = generer_corpus_factice_valide()
        fiches_t[0]["date"] = "2026-09-25"  # Avant 2026-10-02
        with tempfile.TemporaryDirectory() as tmpdir:
            p_t, p_r = self._ecrire_fichiers(tmpdir, fiches_t, fiches_r)
            res = verifier_corpus_tr(p_t, p_r, self.seuils)
            self.assertFalse(res["valide"])
            codes = [v[0] for v in res["violations"]]
            self.assertIn("regle_date_avant_minimum", codes)

    def test_regle_nombre_extraits(self):
        fiches_t, fiches_r = generer_corpus_factice_valide()
        # Seulement 59 dans T
        fiches_t = fiches_t[:59]
        with tempfile.TemporaryDirectory() as tmpdir:
            p_t, p_r = self._ecrire_fichiers(tmpdir, fiches_t, fiches_r)
            res = verifier_corpus_tr(p_t, p_r, self.seuils)
            self.assertFalse(res["valide"])
            codes = [v[0] for v in res["violations"]]
            self.assertIn("regle_nombre_extraits", codes)


if __name__ == "__main__":
    unittest.main()
