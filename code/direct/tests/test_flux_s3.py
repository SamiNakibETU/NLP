"""Tests de flux_s3.py (détection des silences, clôture des segments, deux modes, notebook), hors ligne."""

import sys
import unittest
from pathlib import Path

import numpy as np

RACINE_PROJET = Path(__file__).resolve().parents[3]
DOSSIER_DIRECT = RACINE_PROJET / "code" / "direct"
if str(DOSSIER_DIRECT) not in sys.path:
    sys.path.insert(0, str(DOSSIER_DIRECT))

from flux_s3 import FluxS3, debut_silence, energies_db, lecteur_html, reglages_direct_s3  # noqa: E402
from transcrire_s3 import transcripteur_factice  # noqa: E402

SR = 16000


def bruit_avec_pauses(duree_s: float, pauses: list) -> np.ndarray:
    signal = (0.1 * np.random.default_rng(0).standard_normal(int(duree_s * SR))).astype(np.float32)
    for a, b in pauses:
        signal[int(a * SR): int(b * SR)] *= 0.001
    return signal


class TestSilences(unittest.TestCase):
    def test_energies_db(self):
        niveaux = energies_db(np.concatenate([np.ones(1600), np.zeros(1600), np.ones(800)]), 1600)
        self.assertEqual(len(niveaux), 2)  # la trame incomplète est ignorée
        self.assertAlmostEqual(niveaux[0], 0.0, places=3)
        self.assertLess(niveaux[1], -100)

    def test_debut_silence_apres_parole_seulement(self):
        self.assertIsNone(debut_silence(np.array([True] * 8), 5))  # silence sans parole avant
        self.assertEqual(debut_silence(np.array([True, True, False, False, True, True, True, True, True]), 5), 4)
        self.assertIsNone(debut_silence(np.array([False, True, True, True, True, False]), 5))  # trop court


class TestFluxS3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.transcripteur = transcripteur_factice()

    def diffuser(self, signal, mode="deux_passes"):
        flux = FluxS3(self.transcripteur, reglages_direct_s3(mode))
        return flux, list(flux.diffuser(signal))

    def test_pause_clot_le_segment(self):
        flux, messages = self.diffuser(bruit_avec_pauses(12, [(6.0, 7.0)]))
        self.assertEqual(len(flux.evenements), 2)
        premier = flux.evenements[0]
        self.assertAlmostEqual(premier["fin_parole_s"], 6.0, places=1)
        self.assertLessEqual(premier["fin_s"], 7.0)
        self.assertGreaterEqual(flux.evenements[1]["debut_s"], 6.5)
        self.assertTrue(messages[-1]["est_clos"])
        self.assertTrue(all(b["horodatage"] >= a["horodatage"] for a, b in zip(messages, messages[1:])))

    def test_coupure_d_office_sans_pause(self):
        flux, _ = self.diffuser(bruit_avec_pauses(20, []))
        self.assertEqual(len(flux.evenements), 2)
        self.assertLessEqual(flux.evenements[0]["fin_s"], 15.0)
        self.assertGreaterEqual(flux.evenements[0]["fin_s"], 10.0)  # dans les 5 dernières secondes
        self.assertEqual(flux.evenements[1]["debut_s"], flux.evenements[0]["fin_s"])

    def test_pas_de_segment_trop_court(self):
        # 0,6 s de parole entre deux pauses : rattachée au segment suivant, pas close seule.
        flux, _ = self.diffuser(bruit_avec_pauses(10, [(0.0, 1.0), (1.6, 2.4)]))
        self.assertTrue(all(e["fin_s"] - e["debut_s"] >= 2.0 for e in flux.evenements))

    def test_mode_secours_un_message_par_segment(self):
        flux, messages = self.diffuser(bruit_avec_pauses(12, [(6.0, 7.0)]), "secours_morceaux")
        self.assertEqual(len(messages), len(flux.evenements))
        self.assertTrue(all(m["est_clos"] for m in messages))
        self.assertEqual(flux.bilan()["taux_reecriture_pct"], 0.0)

    def test_silence_seul_ne_produit_rien(self):
        flux, messages = self.diffuser(np.zeros(5 * SR, dtype=np.float32))
        self.assertEqual(messages, [])
        self.assertEqual(flux.bilan(), {"nb_segments": 0})

    def test_lecteur_html(self):
        signal = bruit_avec_pauses(12, [(6.0, 7.0)])
        _, messages = self.diffuser(signal)
        messages[0]["texte_provisoire"] = "</script><b>"  # le texte ne doit pas casser la page
        page = lecteur_html(signal, messages)
        self.assertIn("data:audio/wav;base64,", page)
        self.assertEqual(page.count("</script>"), 1)
        self.assertIn('"horodatage"', page)


class TestNotebook(unittest.TestCase):
    def test_notebook_en_mode_essai(self):
        """notebooks/s3_direct.ipynb s'exécute de bout en bout avec les petits modèles factices."""
        try:
            import nbformat
            from nbclient import NotebookClient
        except ImportError:
            self.skipTest("nbformat / nbclient absents")
        import os

        notebook = nbformat.read(RACINE_PROJET / "notebooks" / "s3_direct.ipynb", as_version=4)
        os.environ["S3_NOTEBOOK_ESSAI"] = "1"
        try:
            NotebookClient(notebook, timeout=300, kernel_name="python3",
                           resources={"metadata": {"path": str(RACINE_PROJET / "notebooks")}}).execute()
        finally:
            os.environ.pop("S3_NOTEBOOK_ESSAI", None)


if __name__ == "__main__":
    unittest.main()
