"""Tests de entrainer_temps1.py avec de petits modèles et des vecteurs factices (hors ligne)."""

import contextlib
import io
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

RACINE_PROJET = Path(__file__).resolve().parents[3]
DOSSIER_PROJECTEUR = RACINE_PROJET / "code" / "projecteur"
if str(DOSSIER_PROJECTEUR) not in sys.path:
    sys.path.insert(0, str(DOSSIER_PROJECTEUR))

from entrainer_temps1 import (  # noqa: E402
    balayer_taux,
    charger_config,
    entrainer,
    evaluer,
    lire_metriques,
    lots_par_longueur,
    main,
    preparer_essai,
    separer_validation,
    taux_apprentissage,
)


class TestOutils(unittest.TestCase):
    def test_lots_par_longueur(self):
        longueurs = [random.Random(k).randint(10, 300) for k in range(203)]
        lots = lots_par_longueur(range(203), longueurs, 16, random.Random(0), facteur_seau=4)
        self.assertEqual(sorted(i for lot in lots for i in lot), list(range(203)))
        self.assertTrue(all(len(lot) <= 16 for lot in lots))
        # Dans un lot, les longueurs sont voisines : bien moins de rembourrage qu'un lot au hasard.
        perte = sum(max(longueurs[i] for i in lot) * len(lot) - sum(longueurs[i] for i in lot) for lot in lots)
        self.assertLess(perte, 0.5 * sum(longueurs))

    def test_separer_validation(self):
        train, validation = separer_validation(100, 10, 42)
        self.assertEqual((len(train), len(validation)), (90, 10))
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(separer_validation(100, 10, 42), (train, validation))

    def test_taux_apprentissage(self):
        self.assertAlmostEqual(taux_apprentissage(0, 100, 10), 0.1)
        self.assertAlmostEqual(taux_apprentissage(9, 100, 10), 1.0)
        self.assertAlmostEqual(taux_apprentissage(10, 100, 10), 1.0)
        self.assertAlmostEqual(taux_apprentissage(100, 100, 10), 0.0)
        self.assertLess(taux_apprentissage(60, 100, 10), taux_apprentissage(30, 100, 10))


class TestEntrainement(unittest.TestCase):
    def setUp(self):
        self.dossier = Path(tempfile.mkdtemp(prefix="test_entrainer_temps1_"))
        self.config = charger_config()
        self.modele = preparer_essai(self.config, "temps_1_pilote", self.dossier)
        self.silence = []

    def tearDown(self):
        shutil.rmtree(self.dossier, ignore_errors=True)

    def test_entrainement_puis_evaluation(self):
        resultat = entrainer(self.config, "temps_1_pilote", self.modele, journal=self.silence.append)
        sortie = Path(resultat["dossier"])
        for nom in ("metriques.jsonl", "projecteur_meilleur.pt", "reprise.pt"):
            self.assertTrue((sortie / nom).is_file(), nom)
        metriques = lire_metriques(sortie)
        validations = [l for l in metriques if l["type"] == "validation"]
        self.assertEqual([l["etape"] for l in validations], [0, 4, 8])
        self.assertIn("wer_pct", validations[0])
        test = evaluer(self.config, "temps_1_pilote", self.modele, journal=self.silence.append)
        self.assertEqual(test["nb_segments"], 6)
        self.assertTrue((sortie / "hypotheses_test.jsonl").is_file())

    def test_reprise(self):
        self.config["temps_1_pilote"]["etapes_max"] = 4
        entrainer(self.config, "temps_1_pilote", self.modele, journal=self.silence.append)
        self.config["temps_1_pilote"]["etapes_max"] = None
        self.silence.clear()
        resultat = entrainer(self.config, "temps_1_pilote", self.modele, journal=self.silence.append)
        self.assertIn("reprise à l'étape 4", self.silence)
        self.assertEqual(resultat["etapes"], 8)
        validations = [l["etape"] for l in lire_metriques(resultat["dossier"]) if l["type"] == "validation"]
        self.assertEqual(validations, [0, 4, 8])

    def test_balayage(self):
        resultats = balayer_taux(self.config, [1e-4, 1e-2], 3, "temps_1_pilote", self.modele,
                                 journal=self.silence.append)
        self.assertEqual([r["learning_rate"] for r in resultats], [1e-4, 1e-2])
        self.assertTrue(all(r["perte_validation"] > 0 for r in resultats))

    def test_mode_essai(self):
        with contextlib.redirect_stdout(io.StringIO()):
            resultat = main(["--essai"])
        self.assertEqual(resultat["etapes"], 8)


class TestNotebook(unittest.TestCase):
    def test_notebook_en_mode_essai(self):
        """notebooks/s3_temps1_pilote.ipynb s'exécute de bout en bout avec les petits modèles factices."""
        try:
            import nbformat
            from nbclient import NotebookClient
        except ImportError:
            self.skipTest("nbformat / nbclient absents")
        import os

        chemin = RACINE_PROJET / "notebooks" / "s3_temps1_pilote.ipynb"
        notebook = nbformat.read(chemin, as_version=4)
        os.environ["S3_NOTEBOOK_ESSAI"] = "1"
        try:
            NotebookClient(notebook, timeout=300, kernel_name="python3",
                           resources={"metadata": {"path": str(RACINE_PROJET / "notebooks")}}).execute()
        finally:
            os.environ.pop("S3_NOTEBOOK_ESSAI", None)


if __name__ == "__main__":
    unittest.main()
