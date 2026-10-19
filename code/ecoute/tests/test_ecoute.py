"""Tests unitaires pour le module code/ecoute (sections 3.1 et 3.2 de specifications.md).

Vérifie :
1. Lecture dynamique de `d_enc` (encodeur audio) et `d_llm` (`hidden_size` du LLM) sans dimension en dur.
2. Calcul du nombre de vecteurs utiles et retrait du silence ajouté pour atteindre 30 secondes.
3. Extraction de vecteurs par `ExtracteurEncodeurAudio` en mode factice.
4. Règle de choix du modèle d'écoute : si deux modèles sont à moins de 3 points de WER,
   on retient celui dont l'encodeur est le plus simple à réutiliser dans S3.
5. Règle de choix du LLM au score chrF++ sans entraînement.
6. Exécution du mode `--essai` de `comparer_briques.py`.
"""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

RACINE_PROJET = Path(__file__).resolve().parents[3]
DOSSIER_ECOUTE = RACINE_PROJET / "code" / "ecoute"
if str(DOSSIER_ECOUTE) not in sys.path:
    sys.path.insert(0, str(DOSSIER_ECOUTE))

from comparer_briques import (  # noqa: E402
    choisir_meilleur_llm,
    choisir_meilleur_modele_ecoute,
    evaluer_briques_mode_essai,
)
from encodeur import (  # noqa: E402
    ExtracteurEncodeurAudio,
    calculer_nb_vecteurs_utiles,
    extraire_vecteurs_sans_silence,
    lire_dimension_encodeur,
    lire_dimension_llm,
)


class TestModuleEcoute(unittest.TestCase):
    """Tests unitaires des fonctions d'écoute et de choix des briques."""

    def test_lire_dimensions_dynamiques(self):
        """Vérifie que d_enc et d_llm sont lus depuis la configuration sans constante en dur."""
        self.assertEqual(lire_dimension_encodeur({"d_model": 1280}), 1280)
        self.assertEqual(lire_dimension_encodeur({"audio_config": {"hidden_size": 1024}}), 1024)
        self.assertEqual(lire_dimension_llm({"hidden_size": 2560}), 2560)
        self.assertEqual(lire_dimension_llm({"text_config": {"hidden_size": 2048}}), 2048)

        with self.assertRaises(ValueError):
            lire_dimension_encodeur({})
        with self.assertRaises(ValueError):
            lire_dimension_llm({})

    def test_retrait_silence_padding_30s(self):
        """Pour Whisper (1 500 vecteurs sur 30 s, soit 50 Hz), 6 s gardent exactement 300 vecteurs."""
        self.assertEqual(calculer_nb_vecteurs_utiles(6.0, nb_total_vecteurs=1500, duree_fenetre_s=30.0), 300)
        self.assertEqual(calculer_nb_vecteurs_utiles(30.0, nb_total_vecteurs=1500, duree_fenetre_s=30.0), 1500)

        tenseur = torch.randn(1, 1500, 1280)
        tronque = extraire_vecteurs_sans_silence(tenseur, duree_reelle_s=6.0, duree_fenetre_s=30.0)
        self.assertEqual(tuple(tronque.shape), (1, 300, 1280))
        self.assertTrue(torch.equal(tronque, tenseur[:, :300, :]))

    def test_extracteur_encodeur_factice(self):
        """Vérifie que ExtracteurEncodeurAudio retire bien le silence de complétion."""
        extracteur = ExtracteurEncodeurAudio(mode_factice=True, d_enc_factice=1280, duree_fenetre_s=30.0)
        sr = 16000
        audio_4s = np.zeros(4 * sr, dtype=np.float32)
        sortie = extracteur.extraire(audio_4s, sr=sr, retirer_silence_padding=True)
        self.assertEqual(sortie["nb_vecteurs_complets"], 1500)
        self.assertEqual(sortie["nb_vecteurs_utiles"], 200)
        self.assertEqual(tuple(sortie["vecteurs"].shape), (200, 1280))

    def test_regle_choix_modele_ecoute_ecart_3_points(self):
        """Si deux modèles sont à moins de 3 points de WER, on garde l'encodeur le plus simple pour S3."""
        candidats_proches = [
            {"nom": "whisper-large-v3", "wer_pct": 11.8, "simplicite_s3": 1, "rtf": 0.30},
            {"nom": "cohere-transcribe-arabic", "wer_pct": 9.5, "simplicite_s3": 3, "rtf": 0.35},
            {"nom": "qwen3-asr-1.7b", "wer_pct": 14.0, "simplicite_s3": 2, "rtf": 0.25},
        ]
        # Écart entre cohere (9.5) et whisper (11.8) = 2.3 <= 3.0 -> whisper retenu (simplicite_s3=1)
        choix = choisir_meilleur_modele_ecoute(candidats_proches, ecart_max_pts=3.0)
        self.assertEqual(choix["retenu"]["nom"], "whisper-large-v3")

        # Si l'écart dépasse 3.0 points (ex. 8.0 vs 12.5 = 4.5 > 3.0), le plus précis l'emporte
        candidats_ecartes = [
            {"nom": "whisper-large-v3", "wer_pct": 12.5, "simplicite_s3": 1, "rtf": 0.30},
            {"nom": "cohere-transcribe-arabic", "wer_pct": 8.0, "simplicite_s3": 3, "rtf": 0.35},
        ]
        choix_ecarte = choisir_meilleur_modele_ecoute(candidats_ecartes, ecart_max_pts=3.0)
        self.assertEqual(choix_ecarte["retenu"]["nom"], "cohere-transcribe-arabic")

    def test_choix_meilleur_llm_chrf(self):
        """Vérifie que le LLM au meilleur chrF++ est retenu."""
        candidats_llm = [
            {"nom": "qwen3-4b", "chrf_pp": 54.2, "rtf": 0.4},
            {"nom": "gemma-4-e4b", "chrf_pp": 56.8, "rtf": 0.5},
            {"nom": "qwen3-1.7b", "chrf_pp": 49.1, "rtf": 0.2},
        ]
        choix = choisir_meilleur_llm(candidats_llm)
        self.assertEqual(choix["retenu"]["nom"], "gemma-4-e4b")

    def test_mode_essai_comparer_briques(self):
        """Vérifie que evaluer_briques_mode_essai tourne de bout en bout et écrit le CSV."""
        with tempfile.TemporaryDirectory() as tmp:
            res = evaluer_briques_mode_essai(dossier_sortie=tmp)
            self.assertEqual(res["choix_ecoute"]["retenu"]["nom"], "whisper-large-v3")
            self.assertEqual(res["choix_llm"]["retenu"]["nom"], "qwen3-4b")
            self.assertTrue(Path(res["chemin_csv"]).exists())


if __name__ == "__main__":
    unittest.main()
