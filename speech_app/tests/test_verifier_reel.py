"""Garde-fous du script de vérification réelle (verifier_reel.py) et des correctifs trouvés en l'écrivant.

Aucun modèle, aucun GPU : moteur factice, faux LLM. Voir la vraie vérification : `python verifier_reel.py`.
"""
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lazy_engines
import live_moteurs
import session_api
import session_report as sr
import verifier_reel as vr
from sessions import Session


class TestOutils(unittest.TestCase):
    def test_wer(self):
        self.assertEqual(vr.wer("مرحبا بكم في النشرة", "مرحبا بكم في النشرة"), 0.0)
        self.assertAlmostEqual(vr.wer("a b c d", "a x c"), 0.5)
        self.assertEqual(vr.wer("", ""), 0.0)

    def test_srt_valide_accepte_crlf_et_refuse_le_casse(self):
        bon = "1\r\n00:00:00,000 --> 00:00:02,000\r\nBonjour\r\n\r\n2\r\n00:00:02,500 --> 00:00:04,000\r\nSalut\r\n"
        self.assertEqual(vr.srt_valide(bon), 2)
        self.assertEqual(vr.srt_valide("1\n00:00 --> 00:02\nBonjour\n"), 0)
        self.assertEqual(vr.srt_valide(""), 0)

    def test_assembler_met_un_silence_entre_les_phrases(self):
        a = np.ones(1600, dtype=np.float32)
        s = vr.assembler([a, a], silence_s=0.6)
        self.assertEqual(len(s), 3 * 9600 + 2 * 1600)

    def test_meilleur_groupe_ne_garde_que_la_meme_voix(self):
        emb = {"a1": np.array([1.0, 0.0]), "a2": np.array([0.99, 0.14]), "b1": np.array([0.0, 1.0])}
        emb = {k: v / np.linalg.norm(v) for k, v in emb.items()}
        self.assertEqual(sorted(vr.meilleur_groupe(["a1", "a2", "b1"], emb, 0.8)), ["a1", "a2"])

    def test_analyse_detecte_arabe_et_doublons(self):
        msgs = [
            {"type": "provisoire", "id_segment": "s1", "fr": "Bonjour", "fin_s": 1.0, "recu_a_s": 1.2},
            {"type": "definitif", "id_segment": "s1", "fr": "Bonjour à tous", "fin_s": 2.0, "recu_a_s": 2.5},
            {"type": "definitif", "id_segment": "s2", "fr": "Bonjour à tous", "fin_s": 4.0, "recu_a_s": 4.6},
            {"type": "definitif", "id_segment": "s3", "fr": "Le président محمد", "fin_s": 6.0, "recu_a_s": 6.5},
        ]
        verifs, chiffres = vr.analyser_messages(msgs, 3, None)
        self.assertFalse(verifs["pas_de_doublon"])
        self.assertFalse(verifs["aucun_caractere_arabe_dans_fr"])
        self.assertEqual(chiffres["doublons_texte"], 1)
        self.assertEqual(chiffres["definitifs_precedes_d_un_gris"], 1)
        self.assertAlmostEqual(chiffres["retard_reel_vs_fin_segment_s"]["max"], 0.6)


class TestRejeuFactice(unittest.TestCase):
    def test_rejeu_rapide_donne_du_gris_puis_du_blanc_sans_arabe(self):
        flux_mod = vr.importer_flux()
        moteur = vr.MoteurChronometre(live_moteurs.MoteurFactice())
        flux = flux_mod.FluxIncremental(moteur, flux_mod.reglages_direct_live())
        signal = live_moteurs.signal_demo([0, 1, 2, 3], duree_phrase_s=6.0, pause_s=1.2)
        messages, _ = vr.rejouer(flux, signal, vitesse=200.0)
        verifs, chiffres = vr.analyser_messages(messages, 4, None)
        self.assertEqual(chiffres["n_definitifs"], 4, chiffres)
        self.assertTrue(all(verifs.values()), verifs)
        self.assertGreater(len(moteur.temps["transcrire"]), 4)
        self.assertEqual(len(moteur.temps["definitif"]), 4)


class TestCorrectifs(unittest.TestCase):
    def test_le_rapport_decharge_toujours_le_pool_partage(self):
        """Sans moteur d'interface (PORTEPAROLE_ENGINES vide), le rapport doit quand même décharger le groupe « direct »."""
        self.assertIs(session_api._pool(), lazy_engines.POOL)

    def test_resume_extractif_controle_aussi_ses_horodatages(self):
        anciens = os.environ.get("PORTEPAROLE_LLM_RAPPORT"), os.environ.get("PORTEPAROLE_SESSIONS")
        os.environ["PORTEPAROLE_LLM_RAPPORT"] = "off"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["PORTEPAROLE_SESSIONS"] = tmp
                s = Session.creer("test", "faux", "extractif")
                s.ecrire_audio(np.zeros(16000 * 12, dtype=np.float32))
                for i in range(3):
                    s.ajouter_segment({"id": i, "debut_s": 4.0 * i, "fin_s": 4.0 * i + 3.5, "ar": "",
                                       "fr": f"Les marchés resteront prudents dans les jours à venir, phrase {i}."})
                r = sr.generer_rapport(s, {"diarisation": "off", "lexique": [], "docx": False})
        finally:
            for cle, v in zip(("PORTEPAROLE_LLM_RAPPORT", "PORTEPAROLE_SESSIONS"), anciens):
                if v is None:
                    os.environ.pop(cle, None)
                else:
                    os.environ[cle] = v
        self.assertTrue(r["methode_resume"].startswith("extractif"))
        self.assertGreater(r["controle_horodatages"]["cites"], 0)
        self.assertEqual(r["controle_horodatages"]["invalides"], [])

    def test_rattacher_horodatages_au_segment_le_plus_proche(self):
        segs = [{"debut_s": 0.0}, {"debut_s": 65.0}, {"debut_s": 130.0}]
        textes = ["Bonsoir et bienvenue.", "Les pourparlers entre les deux délégations se poursuivent.",
                  "Les marchés financiers restent prudents."]
        md = chr(10).join(['## Points clés', '- Les pourparlers des délégations continuent', '- Marchés prudents [09:09]', '- Déjà daté [00:00]', '', '## Chiffres', '- aucun chiffre cité'])
        sortie, n = sr.rattacher_horodatages(md, segs, textes)
        self.assertEqual(n, 2)
        self.assertIn("continuent [01:05]", sortie)
        self.assertIn("Marchés prudents [02:10]", sortie)
        self.assertNotIn("09:09", sortie)
        self.assertIn("- aucun chiffre cité", sortie)
        self.assertNotIn("aucun chiffre cité [", sortie)
        self.assertEqual(sr.verifier_horodatages(sortie, segs)["invalides"], [])

    def test_diarisation_retombe_sur_les_etiquettes_du_direct(self):
        segs = [{"debut_s": 0.0, "fin_s": 3.0, "locuteur": "x"}, {"debut_s": 3.0, "fin_s": 6.0, "locuteur": "y"},
                {"debut_s": 6.0, "fin_s": 9.0, "locuteur": "x"}]
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["PORTEPAROLE_SESSIONS"] = tmp
            s = Session.creer("test", "faux", "d")
            s.ecrire_audio(np.zeros(16000 * 9, dtype=np.float32))
            avert = []
            with mock.patch.object(sr, "diariser_pyannote", side_effect=RuntimeError("non")),                     mock.patch.object(sr, "diariser_ecapa", side_effect=RuntimeError("non")):
                etiquettes, methode = sr.etape_diarisation(s, segs, {"diarisation": "auto"}, avert)
        self.assertEqual(etiquettes, [1, 2, 1])
        self.assertEqual(methode, "empreinte ECAPA en direct")

    def test_relecture_par_lots_et_ciblee(self):
        class LLMLots:
            nom = "faux"
            def __init__(self): self.appels_batch, self.appels_seuls = [], 0
            def batch(self, prompts, max_new_tokens=0):
                self.appels_batch.append(len(prompts))
                return [p[-1]["content"].split("Traduction à relire : ", 1)[1] for p in prompts]
            def __call__(self, messages, max_new_tokens=0):
                self.appels_seuls += 1
                return messages[-1]["content"].split("Traduction à relire : ", 1)[1]
        segs = [{"id": i, "debut_s": 4.0 * i, "fin_s": 4.0 * i + 3, "ar": "", "fr": f"Il y avait {i + 3} blessés ce jour."}
                for i in range(10)]
        segs += [{"id": 10 + i, "debut_s": 50.0 + i, "fin_s": 51.0 + i, "ar": "", "fr": "ils ont parlé longuement ensuite."} for i in range(3)]
        llm = LLMLots()
        sortie, bilan = sr.etape_revision(segs, llm, [], {"taille_lot_revision": 8})
        self.assertEqual(llm.appels_batch, [8, 2])          # 10 segments à chiffres, deux generate au lieu de dix
        self.assertEqual(bilan["segments_relus"], 10)
        self.assertEqual(sortie[12]["raison"], "non relu (rien d'évident à corriger)")
        self.assertEqual(len(sortie), 13)

    def test_segment_isole_rejoint_le_groupe_le_plus_proche(self):
        emb = np.array([[1, 0], [0.9, 0.1], [0.95, 0.05], [0, 1], [0.1, 0.9], [0.5, 0.5]], dtype=float)
        et = sr.absorber_singletons(emb, [0, 0, 0, 1, 1, 2])
        self.assertEqual(len(set(et)), 2)
        self.assertEqual(sr.absorber_singletons(emb[:3], [0, 1, 2]), [0, 1, 2])      # trop peu de segments pour juger

    def test_segment_isole_rejoint_le_groupe_le_plus_proche(self):
        emb = np.array([[1, 0], [0.9, 0.1], [0.95, 0.05], [0, 1], [0.1, 0.9], [0.6, 0.55]], dtype=float)
        sortie = sr.absorber_singletons(emb, [0, 0, 0, 1, 1, 2])
        self.assertEqual(len(set(sortie)), 2)
        self.assertEqual(sr.absorber_singletons(emb[:3], [0, 1, 2]), [0, 1, 2])     # trop peu de segments pour juger


if __name__ == "__main__":
    unittest.main()
