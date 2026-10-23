"""Tests de transcrire_s3.py (découpage, chaîne complète avec de petits modèles, SRT), hors ligne."""

import contextlib
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

RACINE_PROJET = Path(__file__).resolve().parents[3]
DOSSIER_PROJECTEUR = RACINE_PROJET / "code" / "projecteur"
if str(DOSSIER_PROJECTEUR) not in sys.path:
    sys.path.insert(0, str(DOSSIER_PROJECTEUR))

from transcrire_s3 import charger_audio, decouper, main, nettoyer_traduction, transcripteur_factice  # noqa: E402

SR = 16000
REGLAGES = {"duree_max_morceau_s": 20, "recherche_coupure_s": 5, "fenetre_energie_s": 0.1, "duree_min_morceau_s": 2}


class TestDecoupage(unittest.TestCase):
    def test_couvre_tout_le_signal_sans_trou(self):
        signal = np.random.default_rng(0).standard_normal(int(67.3 * SR)).astype(np.float32)
        bornes = decouper(signal, REGLAGES)
        self.assertEqual(bornes[0][0], 0)
        self.assertEqual(bornes[-1][1], len(signal))
        for (a, b), (c, _) in zip(bornes, bornes[1:]):
            self.assertEqual(b, c)
        self.assertTrue(all(b - a <= (20 + 2) * SR for a, b in bornes))

    def test_coupe_dans_le_silence(self):
        signal = np.random.default_rng(0).standard_normal(40 * SR).astype(np.float32)
        signal[int(17.0 * SR): int(17.4 * SR)] = 0.0  # silence entre 17,0 et 17,4 s
        premiere_coupure = decouper(signal, REGLAGES)[0][1] / SR
        self.assertTrue(17.0 <= premiere_coupure <= 17.4, premiere_coupure)

    def test_court_reste_rattache(self):
        signal = np.ones(int(21 * SR), dtype=np.float32)
        signal[int(19.5 * SR): int(19.6 * SR)] = 0.0  # coupure vers 19,55 s : reste de 1,45 s < 2 s
        bornes = decouper(signal, REGLAGES)
        self.assertEqual(bornes, [(0, len(signal))])
        self.assertEqual(len(decouper(np.ones(5 * SR, dtype=np.float32), REGLAGES)), 1)


class TestTranscription(unittest.TestCase):
    def setUp(self):
        self.dossier = Path(tempfile.mkdtemp(prefix="test_transcrire_s3_"))

    def tearDown(self):
        shutil.rmtree(self.dossier, ignore_errors=True)

    def test_chaine_complete_factice(self):
        transcripteur = transcripteur_factice()
        signal = (0.1 * np.random.default_rng(1).standard_normal(45 * SR)).astype(np.float32)
        segments = transcripteur.transcrire_signal(signal)
        self.assertEqual(len(segments), len(decouper(signal, transcripteur.decoupage)))
        self.assertEqual(segments[0]["debut_s"], 0.0)
        self.assertAlmostEqual(segments[-1]["fin_s"], 45.0, places=2)
        self.assertTrue(all(isinstance(s["texte"], str) for s in segments))

    def test_traduction_factice(self):
        transcripteur = transcripteur_factice()
        traductions = transcripteur.traduire(["نص قصير", "", "نص آخر أطول قليلا"])
        self.assertEqual(len(traductions), 3)
        self.assertEqual(traductions[1], "")  # texte vide : pas de génération
        self.assertTrue(all(isinstance(t, str) for t in traductions))

    def test_traduction_directe_factice(self):
        transcripteur = transcripteur_factice()
        signal = (0.1 * np.random.default_rng(2).standard_normal(30 * SR)).astype(np.float32)
        segments = transcripteur.traduire_signal(signal)
        self.assertEqual(len(segments), len(decouper(signal, transcripteur.decoupage)))
        self.assertTrue(all(isinstance(s["texte"], str) for s in segments))

    def test_nettoyer_traduction(self):
        self.assertEqual(nettoyer_traduction("Voici la traduction en français :\n\nBonjour."), "Bonjour.")
        self.assertEqual(nettoyer_traduction("Pour traduire le texte arabe en français, voici la traduction : Oui."), "Oui.")
        self.assertEqual(nettoyer_traduction('"Vous avez créé une ONG ?"'), "Vous avez créé une ONG ?")
        self.assertEqual(nettoyer_traduction("Il a dit : voici le plan."), "Il a dit : voici le plan.")

    def test_charger_audio_reechantillonne_en_mono(self):
        chemin = self.dossier / "stereo_44k.wav"
        sf.write(str(chemin), np.zeros((44100, 2), dtype=np.float32), 44100)
        signal = charger_audio(chemin)
        self.assertEqual(signal.ndim, 1)
        self.assertLessEqual(abs(len(signal) - SR), 1)
        with self.assertRaises(FileNotFoundError):
            charger_audio(self.dossier / "absent.wav")

    def test_mode_essai_et_srt(self):
        chemin_srt = self.dossier / "sortie.srt"
        with contextlib.redirect_stdout(io.StringIO()):
            segments = main(["--essai", "--srt", str(chemin_srt)])
        self.assertEqual(len(segments), 2)  # 30 s -> deux morceaux
        self.assertTrue(chemin_srt.is_file())


class TestNotebook(unittest.TestCase):
    def test_notebook_en_mode_essai(self):
        """notebooks/s3_tester_audio.ipynb s'exécute de bout en bout avec les petits modèles factices."""
        try:
            import nbformat
            from nbclient import NotebookClient
        except ImportError:
            self.skipTest("nbformat / nbclient absents")
        import os

        notebook = nbformat.read(RACINE_PROJET / "notebooks" / "s3_tester_audio.ipynb", as_version=4)
        os.environ["S3_NOTEBOOK_ESSAI"] = "1"
        try:
            NotebookClient(notebook, timeout=300, kernel_name="python3",
                           resources={"metadata": {"path": str(RACINE_PROJET / "notebooks")}}).execute()
        finally:
            os.environ.pop("S3_NOTEBOOK_ESSAI", None)


if __name__ == "__main__":
    unittest.main()
