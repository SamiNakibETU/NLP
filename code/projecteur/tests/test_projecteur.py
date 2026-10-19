"""Tests unitaires pour le système S3 (regroupement, projecteur, assemblage inputs_embeds, temps 1 et 2)."""

from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.nn as nn

RACINE_PROJET = Path(__file__).resolve().parents[3]
DOSSIER_PROJECTEUR = RACINE_PROJET / "code" / "projecteur"
if str(DOSSIER_PROJECTEUR) not in sys.path:
    sys.path.insert(0, str(DOSSIER_PROJECTEUR))

from architecture_s3 import (  # noqa: E402
    ProjecteurAudioLLM,
    assembler_entree_s3,
    regrouper_vecteurs_audio,
)
from entrainement_s3 import (  # noqa: E402
    diagnostiquer_stagnation_s3,
    entrainer_s3_temps1,
    entrainer_s3_temps2_ou_un_temps,
)


class TestModuleProjecteurS3(unittest.TestCase):
    """Vérifie le regroupement, le retrait du silence, le masquage -100 et la convergence du projecteur."""

    def test_regroupement_et_dimensions_projecteur(self):
        """Pour Whisper (1 280) et Qwen3-4B (2 560) ou 2 048, k=4 produit la bonne forme et taille."""
        proj = ProjecteurAudioLLM.depuis_configs(
            config_encodeur={"d_model": 1280},
            config_llm={"hidden_size": 2048},
            k=4,
        )
        # (4 * 1280) * 2048 + 2048 * 2048 + biais = 10 485 760 + 4 194 304 + 4 096 = ~14,68 M paramètres
        self.assertGreater(proj.nb_parametres(), 14_000_000)
        self.assertLess(proj.nb_parametres(), 16_000_000)

        # 6 secondes sur 30 secondes (1 500 vecteurs) -> 300 vecteurs utiles -> 75 vecteurs regroupés (k=4)
        vecteurs_30s = torch.randn(1500, 1280)
        sortie = proj(vecteurs_30s, duree_reelle_s=6.0, duree_fenetre_s=30.0)
        self.assertEqual(tuple(sortie.shape), (75, 2048))

    def test_retrait_silence_et_masquage_uniquement_sur_reponse(self):
        """Vérifie que le silence de complétion est retiré et que la perte ne porte que sur la réponse."""
        proj = ProjecteurAudioLLM(d_enc=64, d_llm=128, k=4)
        table_emb = nn.Embedding(256, 128)

        # Deux tenseurs de 30 s identiques sur les 6 premières secondes (300 vecteurs) mais différents dans le silence (300..1500)
        v_a = torch.randn(1500, 64)
        v_b = v_a.clone()
        v_b[300:, :] = torch.randn(1200, 64) * 100.0

        out_a = proj(v_a, duree_reelle_s=6.0, duree_fenetre_s=30.0)
        out_b = proj(v_b, duree_reelle_s=6.0, duree_fenetre_s=30.0)
        self.assertTrue(torch.allclose(out_a, out_b))

        ass = assembler_entree_s3(
            vecteurs_audio_projetes=out_a,
            ids_debut_user=torch.tensor([1, 2]),
            ids_consigne=torch.tensor([10, 11, 12]),
            ids_fin_user_debut_assistant=torch.tensor([3, 4]),
            table_embeddings_llm=table_emb,
            ids_reponse=torch.tensor([50, 51, 52, 53]),
        )
        # Longueur avant réponse : 2 + 75 + 3 + 2 = 82 positions masquées à -100 ; 4 positions supervisées
        self.assertEqual(ass["nb_masques"], 82)
        self.assertEqual(ass["nb_supervises"], 4)
        self.assertTrue(all(x == -100 for x in ass["labels"][:82].tolist()))
        self.assertEqual(ass["labels"][82:].tolist(), [50, 51, 52, 53])

        diag = diagnostiquer_stagnation_s3(
            assemblage=ass,
            nb_vecteurs_complets=1500,
            nb_vecteurs_utiles=300,
            duree_reelle_s=6.0,
            duree_fenetre_s=30.0,
            learning_rate=1e-4,
        )
        self.assertTrue(diag["diagnostic_global_ok"])

    def test_temps1_temps2_et_un_seul_temps_baissent_erreur(self):
        """Vérifie que l'erreur de validation baisse au Temps 1 (projecteur seul), Temps 2 et Un seul temps."""
        with tempfile.TemporaryDirectory() as tmp:
            res_t1 = entrainer_s3_temps1(tmp, nb_etapes=60)
            self.assertTrue(res_t1["baisse_val_loss"])
            self.assertEqual(res_t1["nb_params_llm_entrainables"], 0)

            res_t2 = entrainer_s3_temps2_ou_un_temps(
                tmp,
                chemin_projecteur_temps1=res_t1["chemin_projecteur"],
                nb_etapes=60,
                part_rappel_transcription=0.15,
            )
            self.assertTrue(res_t2["baisse_val_loss"])
            self.assertEqual(res_t2["nb_exemples_transcription_rappel"], 3)  # 15 % de 20 = 3

            res_1t = entrainer_s3_temps2_ou_un_temps(
                tmp,
                chemin_projecteur_temps1=None,
                nb_etapes=60,
            )
            self.assertTrue(res_1t["baisse_val_loss"])
            self.assertEqual(res_1t["nb_exemples_transcription_rappel"], 0)


if __name__ == "__main__":
    unittest.main()
