"""Tests du système S3 (projecteur, SpeechLLMS3, temps 1 et 2, DatasetS3) avec de petits modèles aléatoires.

Repris de ``projecteur_v1/tests/test_miniature.py`` (dépôt MLP), converti en unittest.
Correspondance des 7 contrôles d'origine :
- test_projector_lengths                     -> test_longueurs_projecteur
- test_audio_embeddings_drop_padding         -> test_embeddings_audio_sans_rembourrage
- test_sequence_assembly_and_label_masking   -> test_assemblage_et_masquage_des_labels
- test_stage1_only_projector_gets_gradients  -> test_temps1_gradients_projecteur_seul
- test_stage2_projector_and_lora_get_gradients -> test_temps2_gradients_projecteur_et_lora
- test_encoder_stays_frozen_and_eval         -> code/ecoute/tests/test_encoder_s3.py (l'encodeur vit
  désormais dans encoder_s3.py) ; ici test_aucun_encodeur_dans_le_modele
- test_generation_runs_with_left_padding     -> test_generation_avec_rembourrage_a_gauche
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

RACINE_PROJET = Path(__file__).resolve().parents[3]
DOSSIER_PROJECTEUR = RACINE_PROJET / "code" / "projecteur"
if str(DOSSIER_PROJECTEUR) not in sys.path:
    sys.path.insert(0, str(DOSSIER_PROJECTEUR))

from donnees_s3 import DatasetS3, regrouper_lot  # noqa: E402
from etapes_s3 import preparer_temps  # noqa: E402
from factices_s3 import D_ENC, TokenizerFactice, petit_llm, vecteurs_factices  # noqa: E402
from modele_s3 import (  # noqa: E402
    LimiteParSegment, SpeechLLMS3, encoder_texte, limite_tokens, prefixe_chatml, retirer_boucles, suffixe_chatml,
)
from projecteur_s3 import ProjecteurS3, longueur_apres_convolution, nb_vecteurs_llm  # noqa: E402

CONFIG_S3 = yaml.safe_load((RACINE_PROJET / "configs" / "s3.yaml").read_text(encoding="utf-8"))
CIBLES = ["مرحبا", "bonjour tout le monde"]


def construire_modele(nb_reductions=None):
    torch.manual_seed(0)
    config = json.loads(json.dumps(CONFIG_S3))
    if nb_reductions is not None:
        config["briques"]["nb_reductions"] = nb_reductions
    return SpeechLLMS3.depuis_config(config, petit_llm(), TokenizerFactice(), D_ENC)


class TestProjecteur(unittest.TestCase):
    def test_longueurs_projecteur(self):
        self.assertEqual(longueur_apres_convolution(500), 250)
        self.assertEqual(nb_vecteurs_llm(500, 2), 125)
        self.assertEqual(nb_vecteurs_llm(1500, 2), 375)
        self.assertEqual(nb_vecteurs_llm(123, 0), 123)
        projecteur = ProjecteurS3(32, 16, nb_reductions=2)
        y, sortie = projecteur(torch.randn(2, 50, 32), torch.tensor([30, 50]))
        self.assertEqual(tuple(y.shape), (2, 13, 16))  # 50 -> 25 -> 13
        self.assertEqual(sortie.tolist(), [nb_vecteurs_llm(30, 2), nb_vecteurs_llm(50, 2)])
        self.assertEqual(sortie.tolist(), [8, 13])

    def test_sans_reduction_layernorm_et_mlp_seulement(self):
        projecteur = ProjecteurS3(32, 16, nb_reductions=0)
        self.assertEqual(len(projecteur.convolutions), 0)
        y, sortie = projecteur(torch.randn(2, 50, 32), torch.tensor([30, 50]))
        self.assertEqual(tuple(y.shape), (2, 50, 16))
        self.assertEqual(sortie.tolist(), [30, 50])
        # Taille réelle prévue : d_enc 1280 (mesuré), d_llm 1536 (Qwen2.5-1.5B).
        nb = ProjecteurS3(1280, 1536, 0).nb_parametres()
        self.assertEqual(nb, 2 * 1280 + 1280 * 3072 + 3072 + 3072 * 1536 + 1536)

    def test_nb_reductions_negatif_refuse(self):
        with self.assertRaises(ValueError):
            ProjecteurS3(32, 16, nb_reductions=-1)


class TestSpeechLLMS3(unittest.TestCase):
    def test_config_s3(self):
        briques = CONFIG_S3["briques"]
        self.assertEqual(briques["modele_llm_id"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(briques["modele_ecoute_id"], "CohereLabs/cohere-transcribe-arabic-07-2026")
        self.assertEqual(briques["nb_reductions"], 0)
        for cle in ("duree_fenetre_encodeur_s", "debit_encodeur_hz"):
            self.assertNotIn(cle, briques)

    def test_embeddings_audio_sans_rembourrage(self):
        vecteurs = vecteurs_factices((30, 50))
        m = construire_modele(nb_reductions=2)
        audio = m.embeddings_audio(vecteurs)
        self.assertEqual([a.shape[0] for a in audio], [8, 13])  # 30 -> 15 -> 8 ; 50 -> 25 -> 13
        self.assertEqual(audio[0].shape[1], m.llm.config.hidden_size)
        m0 = construire_modele()
        self.assertEqual([a.shape[0] for a in m0.embeddings_audio(vecteurs)], [30, 50])

    def test_rembourrage_sans_effet_sur_un_segment(self):
        """Un segment seul ou dans un lot donne les mêmes embeddings audio utiles."""
        m = construire_modele(nb_reductions=2).eval()
        vecteurs = vecteurs_factices((30, 50))
        seul = m.embeddings_audio(vecteurs[:1])[0]
        en_lot = m.embeddings_audio(vecteurs)[0]
        self.assertTrue(torch.allclose(seul, en_lot, atol=1e-5))

    def test_assemblage_et_masquage_des_labels(self):
        m = construire_modele()
        audio = m.embeddings_audio(vecteurs_factices())
        sequences, labels = m.construire_sequences(audio, "traduction", CIBLES)
        n_prefixe = len(encoder_texte(m.tokenizer, prefixe_chatml(m.systeme)))
        n_suffixe = len(encoder_texte(m.tokenizer, suffixe_chatml(m.consignes["traduction"])))
        for i, (s, l) in enumerate(zip(sequences, labels)):
            cible = encoder_texte(m.tokenizer, CIBLES[i]) + [m.tokenizer.eos_token_id]
            self.assertEqual(len(s), n_prefixe + audio[i].shape[0] + n_suffixe + len(cible))
            self.assertTrue((l[: len(s) - len(cible)] == -100).all())  # rien hors cible
            self.assertEqual(l[len(s) - len(cible):].tolist(), cible)  # cible + EOS
        # rembourrage à droite pour l'entraînement
        _, masque, etiquettes = m.rembourrer(sequences, labels, "droite")
        self.assertTrue(masque[0, : len(sequences[0])].all())
        self.assertFalse(masque[0, len(sequences[0]):].any())
        self.assertTrue((etiquettes[masque == 0] == -100).all())

    def test_perte_identique_a_celle_du_llm(self):
        """La perte calculée sur la cible seule est celle de ``llm(labels=...)`` sur toute la séquence."""
        m = construire_modele()
        vecteurs = vecteurs_factices()
        sequences, labels = m.construire_sequences(m.embeddings_audio(vecteurs), "transcription", CIBLES)
        embeddings, masque, etiquettes = m.rembourrer(sequences, labels, "droite")
        reference = m.llm(inputs_embeds=embeddings, attention_mask=masque, labels=etiquettes).loss
        sortie = m(vecteurs, "transcription", CIBLES)
        self.assertTrue(torch.allclose(sortie.loss, reference, atol=1e-5))
        nb_cible = sum(len(encoder_texte(m.tokenizer, c)) + 1 for c in CIBLES)
        self.assertEqual(sortie.logits.shape[0], nb_cible)

    def test_tache_inconnue(self):
        m = construire_modele()
        with self.assertRaises(KeyError):
            m.construire_sequences(m.embeddings_audio(vecteurs_factices()), "resume", CIBLES)

    def test_temps1_gradients_projecteur_seul(self):
        m = construire_modele()
        parametres = preparer_temps(m, 1, CONFIG_S3["temps_2"]["lora"])
        sortie = m(vecteurs_factices(), "transcription", CIBLES)
        sortie.loss.backward()
        self.assertTrue(torch.isfinite(sortie.loss))
        self.assertTrue(all(p.grad is not None for p in m.projecteur.parameters()))
        self.assertTrue(all(p.grad is None for p in m.llm.parameters()))
        self.assertEqual({id(p) for p in parametres}, {id(p) for p in m.projecteur.parameters()})

    def test_temps2_gradients_projecteur_et_lora(self):
        m = construire_modele()
        config_lora = CONFIG_S3["temps_2"]["lora"]
        parametres = preparer_temps(m, 2, config_lora)
        sortie = m(vecteurs_factices(), "traduction", CIBLES)
        sortie.loss.backward()
        lora = [(n, p) for n, p in m.llm.named_parameters() if "lora_" in n]
        base = [(n, p) for n, p in m.llm.named_parameters() if "lora_" not in n]
        self.assertTrue(lora and all(p.requires_grad for _, p in lora))
        self.assertTrue(all(not p.requires_grad for _, p in base))
        self.assertTrue(all(p.grad is not None for p in m.projecteur.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for _, p in lora))
        self.assertTrue(all(p.grad is None for _, p in base))
        self.assertEqual(len(parametres), len(list(m.projecteur.parameters())) + len(lora))
        # LoRA lu dans s3.yaml
        config_peft = m.llm.peft_config["default"]
        self.assertEqual((config_peft.r, config_peft.lora_alpha), (config_lora["r"], config_lora["lora_alpha"]))
        self.assertEqual(set(config_peft.target_modules), set(config_lora["target_modules"]))

    def test_aucun_encodeur_dans_le_modele(self):
        """Les vecteurs sont précalculés : le modèle ne contient que le LLM et le projecteur."""
        m = construire_modele()
        self.assertEqual({nom for nom, _ in m.named_children()}, {"llm", "projecteur"})

    def test_generation_avec_rembourrage_a_gauche(self):
        m = construire_modele().eval()
        vecteurs = vecteurs_factices()
        textes = m.generer(vecteurs, "transcription", max_nouveaux_tokens=5)
        self.assertEqual(len(textes), 2)
        self.assertTrue(all(isinstance(t, str) for t in textes))
        sequences, labels = m.construire_sequences(m.embeddings_audio(vecteurs), "transcription", None)
        _, masque, _ = m.rembourrer(sequences, labels, "gauche")
        self.assertEqual(int(masque[0, -1]), 1)  # le plus court est rembourré à gauche
        self.assertEqual(int(masque[0, 0]), 0)
        self.assertTrue(masque[1].all())


class TestAntiBoucles(unittest.TestCase):
    def test_retirer_boucles(self):
        self.assertEqual(retirer_boucles("a b c d"), "a b c d")
        self.assertEqual(retirer_boucles("x بسم الله بسم الله بسم الله بسم الله بسم"), "x بسم الله بسم")
        self.assertEqual(retirer_boucles("جدا جدا جدا"), "جدا جدا جدا")  # 3 répétitions : gardées
        self.assertEqual(retirer_boucles("w w w w w fin"), "w fin")
        self.assertEqual(retirer_boucles("w w w w", repetitions_min=0), "w w w w")

    def test_limite_tokens(self):
        self.assertEqual(limite_tokens(125, 12.5, 7, 10), 80)  # 10 s -> 70 + 10
        self.assertEqual(limite_tokens(1, 12.5, 7, 10), 11)

    def test_limite_par_segment_en_generation(self):
        m = construire_modele().eval()
        vecteurs = vecteurs_factices((30, 50, 40))
        limites = [2, 5, 3]
        sequences, labels = m.construire_sequences(m.embeddings_audio(vecteurs), "transcription", None)
        embeddings, masque, _ = m.rembourrer(sequences, labels, "gauche")
        from transformers import LogitsProcessorList

        sortie = m.llm.generate(inputs_embeds=embeddings, attention_mask=masque, max_new_tokens=20, do_sample=False,
                                logits_processor=LogitsProcessorList([LimiteParSegment(limites, m.tokenizer.eos_token_id)]))
        for ids, limite in zip(sortie.tolist(), limites):
            utiles = ids[: ids.index(m.tokenizer.eos_token_id)] if m.tokenizer.eos_token_id in ids else ids
            self.assertLessEqual(len(utiles), limite)
        self.assertEqual(len(m.generer(vecteurs, "transcription", max_nouveaux_tokens=20, limites_tokens=limites)), 3)


def tokenizer_qwen_en_cache():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(CONFIG_S3["briques"]["modele_llm_id"], local_files_only=True)
    except Exception:
        return None


class TestGabaritChatML(unittest.TestCase):
    @unittest.skipUnless(tokenizer_qwen_en_cache(), "tokenizer Qwen2.5 absent du cache local")
    def test_gabarit_identique_a_apply_chat_template(self):
        tokenizer = tokenizer_qwen_en_cache()
        consignes = CONFIG_S3["consignes"]
        audio = "@@AUDIO@@"
        for consigne in (consignes["temps_1_transcription"], consignes["temps_2_traduction"]):
            reference = tokenizer.apply_chat_template(
                [{"role": "system", "content": consignes["systeme"]}, {"role": "user", "content": audio + consigne}],
                tokenize=False, add_generation_prompt=True)
            self.assertEqual(prefixe_chatml(consignes["systeme"]) + audio + suffixe_chatml(consigne), reference)
        self.assertEqual(tokenizer.eos_token, "<|im_end|>")


class TestDatasetS3(unittest.TestCase):
    def setUp(self):
        self.dossier = Path(tempfile.mkdtemp(prefix="test_dataset_s3_"))
        vecteurs = vecteurs_factices((30, 50, 12))
        np.save(self.dossier / "vecteurs_00000.npy", np.concatenate([v.numpy() for v in vecteurs[:2]]).astype(np.float16))
        np.save(self.dossier / "vecteurs_00001.npy", vecteurs[2].numpy().astype(np.float16))
        lignes = [
            {"id": "a", "texte_ar": "مرحبا", "fichier_npy": "vecteurs_00000.npy", "ligne_debut": 0, "nb_vecteurs": 30},
            {"id": "b", "texte_ar": "bonjour", "fichier_npy": "vecteurs_00000.npy", "ligne_debut": 30, "nb_vecteurs": 50},
            {"id": "c", "texte_ar": "salut", "fichier_npy": "vecteurs_00001.npy", "ligne_debut": 0, "nb_vecteurs": 12},
        ]
        with open(self.dossier / "index.jsonl", "w", encoding="utf-8") as fichier:
            for ligne in lignes:
                fichier.write(json.dumps(dict(ligne, d_enc=D_ENC, encodeur="essai"), ensure_ascii=False) + "\n")
        self.vecteurs = vecteurs

    def tearDown(self):
        shutil.rmtree(self.dossier, ignore_errors=True)

    def test_lecture(self):
        jeu = DatasetS3(self.dossier)
        self.assertEqual((len(jeu), jeu.d_enc), (3, D_ENC))
        v, texte = jeu[1]
        self.assertEqual((tuple(v.shape), v.dtype, texte), ((50, D_ENC), torch.float32, "bonjour"))
        self.assertTrue(torch.allclose(v, self.vecteurs[1].half().float()))

    def test_champ_texte_absent(self):
        with self.assertRaises(KeyError):
            DatasetS3(self.dossier, champ_texte="texte_fr")

    def test_index_absent(self):
        with self.assertRaises(FileNotFoundError):
            DatasetS3(self.dossier / "absent")

    def test_bout_en_bout_perte_et_gradients(self):
        """Index factice -> DatasetS3 -> SpeechLLMS3 -> perte finie, gradients sur le projecteur seul."""
        chargeur = torch.utils.data.DataLoader(DatasetS3(self.dossier), batch_size=3, collate_fn=regrouper_lot)
        vecteurs, textes = next(iter(chargeur))
        self.assertEqual([v.shape[0] for v in vecteurs], [30, 50, 12])
        m = construire_modele()
        preparer_temps(m, 1, CONFIG_S3["temps_2"]["lora"])
        sortie = m(vecteurs, "transcription", textes)
        sortie.loss.backward()
        self.assertTrue(torch.isfinite(sortie.loss))
        self.assertTrue(all(p.grad is not None for p in m.projecteur.parameters()))
        self.assertTrue(all(p.grad is None for p in m.llm.parameters()))


if __name__ == "__main__":
    unittest.main()
