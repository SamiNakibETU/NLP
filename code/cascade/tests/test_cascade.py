r"""Tests unitaires pour le module code/cascade (Étape 6 — section 4.1 et section 5).

Vérifie :
1. La composition exacte des 200 exemples dans les proportions 40 % / 30 % / 20 % / 10 %.
2. La détection des noms propres et la présence du contexte (2 phrases précédentes) dans le chat template.
3. Le masquage `-100` de tous les tokens du prompt (`system` + `user`), l'erreur ne portant que sur la réponse.
4. L'entraînement de bout en bout sur 200 exemples factices, l'interruption simulée à l'étape 100 et la reprise
   jusqu'à l'étape 200.
5. La génération de l'exemple documenté de logits et de softmax ($\sum p_i = 1$) et du chat template.
6. L'inférence en mode différé S1/S2 avec fenêtre glissante de 2 phrases et export SRT.
7. L'exécution sans erreur du notebook Kaggle `notebooks/s2_cascade_lora.ipynb` en mode essai.
"""

import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

import nbformat
from nbclient import NotebookClient

RACINE_PROJET = Path(__file__).resolve().parents[3]
DOSSIER_CASCADE = RACINE_PROJET / "code" / "cascade"
if str(DOSSIER_CASCADE) not in sys.path:
    sys.path.insert(0, str(DOSSIER_CASCADE))

from entrainement_lora import entrainer_lora_s2_essai, estimer_budget_calcul_s2  # noqa: E402
from inference_s1_s2 import SystemeCascadeS1S2, executer_mode_essai_inference  # noqa: E402
from preparation_exemples import (  # noqa: E402
    charger_lexique_noms,
    detecter_noms_propres_segment,
    encoder_exemple_avec_masquage_s2,
    generer_exemples_factices_s2,
)


class TestModuleCascade(unittest.TestCase):
    """Tests unitaires de la préparation, de l'entraînement LoRA et de l'inférence S1/S2."""

    def test_proportions_200_exemples(self):
        """Vérifie les proportions 40 % (80), 30 % (60), 20 % (40), 10 % (20) sur 200 exemples."""
        jeu, comptes = generer_exemples_factices_s2(nb_total=200, graine=42)
        self.assertEqual(len(jeu), 200)
        self.assertEqual(comptes["d_transcription_manuelle"], 80)
        self.assertEqual(comptes["d_sortie_ecoute"], 60)
        self.assertEqual(comptes["p_presse"], 40)
        self.assertEqual(comptes["d6_debut_phrase"], 20)
        self.assertEqual(comptes["total"], 200)

    def test_chat_template_et_masquage_etiquettes(self):
        """Vérifie que le chat template contient les 2 phrases précédentes, les noms propres et masque le prompt."""
        lexique = charger_lexique_noms()
        texte_ar = "التقى الرئيس الأسبق بيل كلينتون بالوفد في المؤتمر."
        noms = detecter_noms_propres_segment(texte_ar, lexique)
        self.assertTrue(any(n["forme_francaise"] == "Bill Clinton" for n in noms))

        enc = encoder_exemple_avec_masquage_s2(
            texte_arabe=texte_ar,
            traduction_fr="L'ancien président Bill Clinton a rencontré la délégation lors de la conférence.",
            phrases_precedentes_fr=["Première phrase traduite.", "Deuxième phrase traduite."],
            noms_propres=noms,
        )
        tpl = enc["chat_template_complet"]
        self.assertIn("Première phrase traduite.", tpl)
        self.assertIn("Deuxième phrase traduite.", tpl)
        self.assertIn("Bill Clinton", tpl)
        self.assertIn(texte_ar, tpl)

        labels = enc["labels"].tolist()
        nb_prompt = enc["nb_tokens_prompt_masques"]
        self.assertGreater(nb_prompt, 10)
        self.assertTrue(all(x == -100 for x in labels[:nb_prompt]))
        self.assertTrue(all(x != -100 for x in labels[nb_prompt:]))

    def test_entrainement_200_exemples_interruption_et_reprise(self):
        """Vérifie l'entraînement sur 200 exemples, l'interruption simulée à 100 étapes et la reprise à 200."""
        with tempfile.TemporaryDirectory() as tmp:
            res_coupure = entrainer_lora_s2_essai(
                dossier_sortie=tmp,
                nb_exemples=200,
                nb_etapes_max=200,
                sauvegarder_tous_les=50,
                interrompre_a_etape=100,
                reprendre=False,
            )
            self.assertTrue(res_coupure["interrompu"])
            self.assertEqual(res_coupure["etape_finale"], 100)

            res_reprise = entrainer_lora_s2_essai(
                dossier_sortie=tmp,
                nb_exemples=200,
                nb_etapes_max=200,
                sauvegarder_tous_les=50,
                interrompre_a_etape=None,
                reprendre=True,
            )
            self.assertTrue(res_reprise["reprise_effectuee"])
            self.assertEqual(res_reprise["etape_depart"], 100)
            self.assertEqual(res_reprise["etape_finale"], 200)
            self.assertFalse(res_reprise["interrompu"])
            self.assertEqual(len(res_reprise["historique"]), 4)  # étapes 50, 100, 150, 200

            # Vérification de la baisse de la perte d'entraînement sur les 200 étapes
            self.assertLess(
                res_reprise["historique"][-1]["train_loss"],
                res_reprise["historique"][0]["train_loss"],
            )

            # Vérification du livrable de logits et softmax (somme = 1.0)
            with open(res_reprise["chemin_logits_template"], "r", encoding="utf-8") as f:
                doc_logits = json.load(f)
            self.assertAlmostEqual(doc_logits["somme_probabilites_softmax"], 1.0, places=4)
            self.assertEqual(len(doc_logits["top_k_predictions"]), 5)
            self.assertIn("<|im_start|>system", doc_logits["chat_template_recu_par_le_modele"])

    def test_estimation_budget_calcul_4_5(self):
        """Vérifie le calcul du budget GPU à partir de 200 étapes chronométrées."""
        budget = estimer_budget_calcul_s2(
            temps_200_etapes_s=760.0,
            nb_exemples_entrainement=50000,
            lot_effectif=16,
            nb_passes=2,
            quota_hebdo_kaggle_h=30.0,
        )
        self.assertEqual(budget["nb_etapes_total"], 6250)
        self.assertAlmostEqual(budget["duree_totale_estimee_h"], 6.6, places=1)
        self.assertLess(budget["part_quota_hebdo_pct"], 25.0)

    def test_inference_mode_differe_contexte_glissant(self):
        """Vérifie que le mode différé injecte bien la phrase du segment 1 en contexte du segment 2."""
        with tempfile.TemporaryDirectory() as tmp:
            res = executer_mode_essai_inference(dossier_sortie=tmp)
            self.assertEqual(len(res["segments"]), 2)
            self.assertEqual(len(res["segments"][0]["contexte_utilise"]), 0)
            self.assertEqual(len(res["segments"][1]["contexte_utilise"]), 1)
            self.assertTrue(Path(res["chemin_srt"]).exists())

    def test_notebook_kaggle_s2_mode_essai(self):
        """Vérifie que notebooks/s2_cascade_lora.ipynb est valide et s'exécute de bout en bout en mode essai."""
        nb_path = RACINE_PROJET / "notebooks" / "s2_cascade_lora.ipynb"
        self.assertTrue(nb_path.exists())
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        nbformat.validate(nb)
        ns = {"__name__": "__main__"}
        for cell in nb.cells:
            if cell.cell_type == "code":
                exec(compile(cell.source, str(nb_path), "exec"), ns)
        self.assertIn("r2", ns)
        self.assertEqual(ns["r2"]["etape_finale"], 200)
        self.assertTrue(ns["r2"]["reprise_effectuee"])


if __name__ == "__main__":
    unittest.main()
