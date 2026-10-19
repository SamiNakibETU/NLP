"""Tests unitaires pour l'Étape 3 : système S0, découpage VAD, export SRT et interface démo."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

DOSSIER_DEMO = Path(__file__).resolve().parents[1]
if str(DOSSIER_DEMO) not in sys.path:
    sys.path.insert(0, str(DOSSIER_DEMO))

from app import traiter_depot_fichier, main as app_main  # noqa: E402
from srt import formater_horodatage_srt, generer_contenu_srt, sauvegarder_srt  # noqa: E402
from systeme_s0 import (  # noqa: E402
    SystemeS0,
    charger_config_s0,
    decouper_audio_vad,
    executer_mode_essai_s0,
    generer_signal_parole_test,
    main as s0_main,
)


class TestExportSRT(unittest.TestCase):
    """Vérifie le formatage exact des sous-titres SubRip (.srt)."""

    def test_formater_horodatage(self):
        self.assertEqual(formater_horodatage_srt(0.0), "00:00:00,000")
        self.assertEqual(formater_horodatage_srt(1.25), "00:00:01,250")
        self.assertEqual(formater_horodatage_srt(3661.005), "01:01:01,005")
        with self.assertRaises(ValueError):
            formater_horodatage_srt(-0.1)

    def test_generer_et_sauvegarder_srt(self):
        segments = [
            {"debut_s": 0.0, "fin_s": 2.5, "texte_fr": "Bonjour et bienvenue au journal."},
            {"debut_s": 2.5, "fin_s": 5.125, "texte_fr": "Voici les titres de l'actualité."},
        ]
        attendu = (
            "1\n"
            "00:00:00,000 --> 00:00:02,500\n"
            "Bonjour et bienvenue au journal.\n\n"
            "2\n"
            "00:00:02,500 --> 00:00:05,125\n"
            "Voici les titres de l'actualité.\n"
        )
        self.assertEqual(generer_contenu_srt(segments), attendu)

        with tempfile.TemporaryDirectory() as tmpdir:
            chemin = sauvegarder_srt(segments, Path(tmpdir) / "test.srt")
            self.assertTrue(chemin.exists())
            self.assertEqual(chemin.read_text(encoding="utf-8"), attendu)


class TestDecoupageVAD(unittest.TestCase):
    """Vérifie le découpage par détection de voix et la limite de 30 secondes."""

    def test_decoupage_pause_silence(self):
        signal = generer_signal_parole_test(duree_s=4.0, sr=16000)
        tranches = decouper_audio_vad(signal, sr=16000, duree_max_s=30.0, silence_min_s=0.5)
        self.assertGreaterEqual(len(tranches), 2)
        for deb, fin, arr in tranches:
            self.assertGreater(fin, deb)
            self.assertLessEqual(fin - deb, 30.0)
            self.assertGreater(len(arr), 0)

    def test_respect_plafond_30_secondes(self):
        """Un signal continu de 65 secondes est obligatoirement découpé en segments <= 30 s."""
        sr = 16000
        t = np.linspace(0.0, 65.0, 65 * sr, endpoint=False, dtype=np.float32)
        signal_continu = 0.2 * np.sin(2.0 * np.pi * 200.0 * t).astype(np.float32)
        tranches = decouper_audio_vad(signal_continu, sr=sr, duree_max_s=30.0)
        self.assertGreaterEqual(len(tranches), 3)
        for deb, fin, _ in tranches:
            self.assertLessEqual(round(fin - deb, 3), 30.0)


class TestSystemeS0EtDemo(unittest.TestCase):
    """Vérifie l'exécution de bout en bout de S0 et de l'interface de dépôt."""

    def test_config_s0_conforme(self):
        cfg = charger_config_s0()
        self.assertEqual(cfg["systeme"]["code"], "S0")
        self.assertEqual(cfg["ecoute"]["modele_id"], "openai/whisper-small")
        self.assertEqual(cfg["traduction"]["modele_id"], "Helsinki-NLP/opus-mt-ar-fr")

    def test_chaine_s0_et_interface_depot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            res = executer_mode_essai_s0(dossier_sortie=tmpdir)
            self.assertEqual(res["systeme"], "S0")
            self.assertGreaterEqual(len(res["segments"]), 1)
            self.assertIn("-->", res["srt"])
            self.assertTrue(Path(res["chemin_srt"]).exists())

            chemin_wav = Path(tmpdir) / "extrait_essai_ar.wav"
            ar, fr, srt_txt, srt_path, resume = traiter_depot_fichier(
                str(chemin_wav),
                mode_substitution=True,
            )
            self.assertTrue(len(ar) > 0)
            self.assertTrue(len(fr) > 0)
            self.assertIn("00:00:", srt_txt)
            self.assertTrue(Path(srt_path).exists())
            self.assertIn("RTF =", resume)

            self.assertEqual(s0_main(["--essai"]), 0)
            self.assertEqual(app_main(["--essai"]), 0)


if __name__ == "__main__":
    unittest.main()
