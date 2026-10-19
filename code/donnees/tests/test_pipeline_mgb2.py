#!/usr/bin/env python3
"""Tests unitaires exhaustifs pour pipeline_mgb2.py.

Ces tests s'exécutent entièrement HORS LIGNE sans GPU ni réseau.
Les tests nécessitant un accès réel à Hugging Face ou un GPU sont décorés
avec @unittest.skipUnless(..., "Test réel désactivé par défaut").
"""

import os
import sys
import json
import csv
import shutil
import tempfile
import unittest
from pathlib import Path

DOSSIER_CODE = Path(__file__).resolve().parents[1]
if str(DOSSIER_CODE) not in sys.path:
    sys.path.insert(0, str(DOSSIER_CODE))

from pipeline_mgb2 import (
    executer_d1_selection,
    generer_grille_aveugle_d2,
    executer_d3_traduction_paquets,
    executer_d4_filtres,
    nettoyer_nom_fichier,
    sauvegarder_et_verifier_audio_flac,
)
from seuils import charger_seuils
from validation import valider_fiche


class TestPipelineMgb2(unittest.TestCase):
    def setUp(self):
        self.seuils = charger_seuils()
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="test_pipeline_mgb2_"))
        self.chemin_d = self.tmp_dir / "d_selection_v1.jsonl"
        self.chemin_a = self.tmp_dir / "a_v1.jsonl"
        self.chemin_rejets = self.tmp_dir / "rejets_d1.csv"
        self.chemin_stats = self.tmp_dir / "stats_d1.json"
        self.dossier_audio = self.tmp_dir / "audio"

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_d1_conservation_et_persistance(self):
        """Vérifie la création effective de tous les fichiers et la stricte conservation."""
        res = executer_d1_selection(
            mode_reel=False,
            max_segments=20,
            repartition_mode="alterne",
            chemin_sortie_d=self.chemin_d,
            chemin_sortie_a=self.chemin_a,
            chemin_rejets=self.chemin_rejets,
            chemin_stats=self.chemin_stats,
            dossier_audio=self.dossier_audio,
        )
        stats = res["stats"]

        # 1. Égalité stricte de conservation
        self.assertEqual(
            stats["entree"],
            stats["gardes_d"] + stats["gardes_a"] + stats["rejets"] + stats["deja_traites"],
            "Conservation stricte non respectée.",
        )

        # 2. Persistance effective de tous les fichiers
        self.assertTrue(self.chemin_d.exists(), "d_selection_v1.jsonl doit exister sur le disque.")
        self.assertTrue(self.chemin_a.exists(), "a_v1.jsonl doit exister sur le disque.")
        self.assertTrue(self.chemin_rejets.exists(), "rejets_d1.csv doit exister sur le disque.")
        self.assertTrue(self.chemin_stats.exists(), "stats_d1.json doit exister sur le disque.")

        # 3. Répartition équilibrée (les deux jeux reçoivent des segments en mode alterne)
        self.assertTrue(stats["gardes_d"] > 0, "Le jeu D doit contenir des segments.")
        self.assertTrue(stats["gardes_a"] > 0, "Le jeu A doit contenir des segments.")

        # 4. Fichiers audio créés et valides
        audios_generes = list(self.dossier_audio.glob("*.flac"))
        self.assertEqual(
            len(audios_generes),
            stats["gardes_d"] + stats["gardes_a"],
            "Chaque segment retenu doit avoir son audio FLAC sur disque.",
        )

    def test_d1_reprise_sans_doublon_et_idempotence(self):
        """Test reprise : interruption après 10 segments, reprise jusqu'à 20, aucun doublon."""
        # 1ère passe : arrêt à 10 segments
        res1 = executer_d1_selection(
            mode_reel=False,
            max_segments=10,
            repartition_mode="alterne",
            chemin_sortie_d=self.chemin_d,
            chemin_sortie_a=self.chemin_a,
            chemin_rejets=self.chemin_rejets,
            chemin_stats=self.chemin_stats,
            dossier_audio=self.dossier_audio,
        )
        nb_d_p1 = res1["stats"]["gardes_d"]
        nb_a_p1 = res1["stats"]["gardes_a"]

        # 2ème passe : relance avec max_segments=20 (reprise sur les mêmes fichiers)
        res2 = executer_d1_selection(
            mode_reel=False,
            max_segments=20,
            repartition_mode="alterne",
            chemin_sortie_d=self.chemin_d,
            chemin_sortie_a=self.chemin_a,
            chemin_rejets=self.chemin_rejets,
            chemin_stats=self.chemin_stats,
            dossier_audio=self.dossier_audio,
        )
        # La 2ème passe a dû détecter les segments déjà traités
        self.assertTrue(res2["stats"]["deja_traites"] > 0, "Des segments déjà traités doivent être sautés.")

        # Vérification qu'il n'y a AUCUN doublon d'identifiant dans les fichiers finaux
        def lire_ids(chemin):
            ids = []
            with open(chemin, "r", encoding="utf-8") as f:
                for l in f:
                    if l.strip():
                        ids.append(json.loads(l)["id_source_mgb2"])
            return ids

        ids_d = lire_ids(self.chemin_d)
        ids_a = lire_ids(self.chemin_a)

        self.assertEqual(len(ids_d), len(set(ids_d)), "Doublons détectés dans le fichier D.")
        self.assertEqual(len(ids_a), len(set(ids_a)), "Doublons détectés dans le fichier A.")
        self.assertTrue(set(ids_d).isdisjoint(set(ids_a)), "Les jeux D et A doivent être strictement disjoints.")

        # 3ème passe : exécuter le même lot une nouvelle fois ne change rien (idempotence)
        res3 = executer_d1_selection(
            mode_reel=False,
            max_segments=20,
            repartition_mode="alterne",
            chemin_sortie_d=self.chemin_d,
            chemin_sortie_a=self.chemin_a,
            chemin_rejets=self.chemin_rejets,
            chemin_stats=self.chemin_stats,
            dossier_audio=self.dossier_audio,
        )
        self.assertEqual(res3["stats"]["gardes_d"], 0, "Aucun nouveau segment D ne doit être ajouté.")
        self.assertEqual(res3["stats"]["gardes_a"], 0, "Aucun nouveau segment A ne doit être ajouté.")

    def test_d1_schema_intermediaire_vs_final(self):
        """Vérifie que la sortie D1 respecte SCHEMA_D_SELECTION et pas SCHEMA_D (pas encore traduit)."""
        res = executer_d1_selection(
            mode_reel=False,
            max_segments=6,
            chemin_sortie_d=self.chemin_d,
            chemin_sortie_a=self.chemin_a,
            chemin_rejets=self.chemin_rejets,
            chemin_stats=self.chemin_stats,
            dossier_audio=self.dossier_audio,
        )
        fiche_d = res["gardes_d"][0]

        # Doit être 100% conforme à SCHEMA_D_SELECTION
        errs_sel, _ = valider_fiche(fiche_d, "d_selection", self.seuils)
        self.assertEqual(len(errs_sel), 0, f"Erreurs dans SCHEMA_D_SELECTION: {errs_sel}")

        # Ne doit PAS être conforme à SCHEMA_D car il manque fr, modele_fr et score_qe (étape D3/D4)
        errs_d_final, _ = valider_fiche(fiche_d, "d", self.seuils)
        champs_manquants = [e[1] for e in errs_d_final if "champ requis absent" in e[1]]
        self.assertTrue(any("fr" in c for c in champs_manquants), "Le schéma D final doit exiger 'fr'.")

    def test_nettoyage_noms_fichiers(self):
        """Vérifie que les identifiants contenant des caractères interdits sous Windows sont nettoyés."""
        id_complexe = "mgb2:ep/001\\seg*25?test|ok"
        nettoye = nettoyer_nom_fichier(id_complexe)
        self.assertNotIn(":", nettoye)
        self.assertNotIn("/", nettoye)
        self.assertNotIn("\\", nettoye)
        self.assertNotIn("*", nettoye)
        self.assertNotIn("?", nettoye)
        self.assertNotIn("|", nettoye)

    def test_erreur_sauvegarde_audio_rejet_explicite(self):
        """Vérifie que l'impossibilité d'écrire ou relire un audio conduit à un rejet explicite."""
        import numpy as np
        # Pointer vers un dossier existant au lieu d'un fichier provoque une erreur d'écriture
        chemin_invalide = self.tmp_dir
        succes, motif = sauvegarder_et_verifier_audio_flac(np.zeros(100, dtype=np.float32), 16000, chemin_invalide)
        self.assertFalse(succes)
        self.assertIn("erreur_sauvegarde_audio", motif)

    def test_mode_reel_refus_sans_token(self):
        """Vérifie qu'aucun repli silencieux vers le mock n'a lieu en mode réel sans token."""
        with self.assertRaises(PermissionError):
            executer_d1_selection(mode_reel=True, hf_token="")

    def test_refus_explicite_d2_d3_d4_en_mode_reel(self):
        """Vérifie que D2, D3 et D4 refusent explicitement le mode réel tant qu'ils ne sont pas implémentés."""
        with self.assertRaises(NotImplementedError):
            generer_grille_aveugle_d2([], [], mode_reel=True)

        with self.assertRaises(NotImplementedError):
            executer_d3_traduction_paquets([], Path("dummy.csv"), mode_reel=True)

        with self.assertRaises(NotImplementedError):
            executer_d4_filtres([], mode_reel=True)

    def test_d2_d3_d4_simulation(self):
        """Vérifie que D2, D3, D4 fonctionnent en mode simulation pour les tests hors ligne."""
        segments = [{"id": f"d-{i:06d}", "episode": "ep_01", "ar_ecoute": f"نص {i}"} for i in range(1, 11)]
        grille, cle = generer_grille_aveugle_d2(segments, ["Qwen/Qwen2.5-7B-Instruct"], mode_reel=False)
        self.assertEqual(len(grille), 10)
        self.assertEqual(len(cle), 10)

        trads, rejets_d3 = executer_d3_traduction_paquets(segments, Path("dummy.csv"), mode_reel=False)
        self.assertEqual(len(trads) + len(rejets_d3), 10)

        gardes, rejets_d4 = executer_d4_filtres(trads, mode_reel=False)
        self.assertEqual(len(gardes) + len(rejets_d4), len(trads))


if __name__ == "__main__":
    unittest.main()
