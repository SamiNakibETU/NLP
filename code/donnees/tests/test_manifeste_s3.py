#!/usr/bin/env python3
"""Tests de manifeste_s3.py, hors ligne, sur un petit corpus MGB-2 factice."""

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

DOSSIER_CODE = Path(__file__).resolve().parents[1]
if str(DOSSIER_CODE) not in sys.path:
    sys.path.insert(0, str(DOSSIER_CODE))

from manifeste_s3 import (  # noqa: E402
    ErreurManifeste,
    charger_config,
    construire_manifestes,
    ecrire_corpus_factice,
    filtrer_segments,
    lire_bornes,
    main,
    motif_rejet,
    tirer_segments,
)
from seuils import charger_seuils  # noqa: E402

BORNES = {"duree_min_s": 5, "duree_max_s": 25, "mots_min": 6, "wmer_max": 15, "max_par_episode": 200}


def segment(identifiant, episode="E1", duree=10.0, nb_mots=10, wmer=0.0, debut=0.0):
    return {"id": identifiant, "episode": episode, "debut": debut, "fin": debut + duree,
            "duree_s": duree, "nb_mots": nb_mots, "wmer": wmer}


class TestFiltres(unittest.TestCase):
    def test_bornes_incluses(self):
        for duree, nb_mots, wmer in ((5.0, 6, 15.0), (25.0, 6, 0.0)):
            self.assertIsNone(motif_rejet(segment("a", duree=duree, nb_mots=nb_mots, wmer=wmer), BORNES))

    def test_motifs_dans_l_ordre(self):
        self.assertEqual(motif_rejet(segment("a", duree=4.99), BORNES), "duree")
        self.assertEqual(motif_rejet(segment("a", duree=25.01), BORNES), "duree")
        self.assertEqual(motif_rejet(segment("a", nb_mots=5), BORNES), "mots")
        self.assertEqual(motif_rejet(segment("a", wmer=15.1), BORNES), "wmer")
        # Plusieurs filtres échouent : seul le premier compte.
        self.assertEqual(motif_rejet(segment("a", duree=2.0, nb_mots=1, wmer=90.0), BORNES), "duree")

    def test_max_par_episode(self):
        bornes = dict(BORNES, max_par_episode=3)
        segments = [segment(f"E1_{k}", debut=k * 20.0) for k in range(10)] + [segment("E2_0", episode="E2")]
        gardes, rejets = filtrer_segments(segments, bornes, graine=42)
        self.assertEqual(sum(s["episode"] == "E1" for s in gardes), 3)
        self.assertEqual(sum(s["episode"] == "E2" for s in gardes), 1)
        self.assertEqual([motif for _, motif in rejets], ["max_par_episode"] * 7)
        # Même graine, même choix ; et ce ne sont pas forcément les 3 premiers.
        gardes_bis, _ = filtrer_segments(segments, bornes, graine=42)
        self.assertEqual([s["id"] for s in gardes], [s["id"] for s in gardes_bis])

    def test_seuils_lus_dans_le_yaml(self):
        seuils = charger_seuils()
        bornes = lire_bornes(seuils)
        self.assertEqual(bornes["duree_min_s"], seuils["constantes"]["duree_segment_min_s"])
        self.assertEqual(bornes["wmer_max"], seuils["proposes"]["ecart_ecoute_max_pct"])
        self.assertEqual(bornes["max_par_episode"], seuils["constantes"]["segments_max_par_episode"])


class TestTirage(unittest.TestCase):
    def test_tirage_emboite(self):
        segments = [segment(f"s{k:03d}") for k in range(300)]
        petit = tirer_segments(segments, 50, 42, "train")
        grand = tirer_segments(list(reversed(segments)), 200, 42, "train")
        self.assertEqual([s["id"] for s in petit], [s["id"] for s in grand[:50]])

    def test_graine_change_le_tirage(self):
        segments = [segment(f"s{k:03d}") for k in range(300)]
        self.assertNotEqual(
            [s["id"] for s in tirer_segments(segments, 50, 42, "train")],
            [s["id"] for s in tirer_segments(segments, 50, 43, "train")],
        )


class TestConstruction(unittest.TestCase):
    def setUp(self):
        self.dossier = Path(tempfile.mkdtemp(prefix="test_manifeste_s3_"))
        self.racine = self.dossier / "mgb2"
        ecrire_corpus_factice(self.racine)
        self.config = charger_config()
        self.config["sources"]["racine_mgb2"] = str(self.racine)
        self.config["selection"].update(nb_segments_train=50, nb_segments_test=2, pct_episodes_test=10)
        self.seuils = charger_seuils()
        self.sortie = self.dossier / "sortie"

    def tearDown(self):
        shutil.rmtree(self.dossier, ignore_errors=True)

    def lire(self, partie):
        texte = (self.sortie / f"{partie}.jsonl").read_text(encoding="utf-8")
        return texte, [json.loads(ligne) for ligne in texte.splitlines()]

    def test_manifestes_et_resume(self):
        resume = construire_manifestes(self.config, self.seuils, self.sortie)
        texte_train, train = self.lire("train")
        _, test = self.lire("test")
        self.assertEqual((len(train), len(test)), (50, 2))
        self.assertFalse({l["episode"] for l in train} & {l["episode"] for l in test})
        bornes = lire_bornes(self.seuils)
        for ligne in train + test:
            self.assertTrue(bornes["duree_min_s"] <= ligne["duree_s"] <= bornes["duree_max_s"])
            self.assertGreaterEqual(ligne["nb_mots"], bornes["mots_min"])
            self.assertLessEqual(ligne["wmer"], bornes["wmer_max"])
            self.assertEqual(ligne["wav"], f"wav/{ligne['episode']}.wav")
            self.assertTrue((self.racine / ligne["wav"]).is_file())
        self.assertEqual({l["partie"] for l in train}, {"train"})
        self.assertIn("كلمة", texte_train)  # ensure_ascii=False : arabe lisible dans le fichier
        self.assertNotIn("texte_fr", train[0])

        for partie in ("train", "test"):
            bloc = resume["parties"][partie]
            self.assertEqual(bloc["lus"]["segments"], bloc["selection_complete"]["segments"] + sum(bloc["rejets"].values()))
        relu = json.loads((self.sortie / "resume_manifeste.json").read_text(encoding="utf-8"))
        self.assertEqual(relu["parties"], resume["parties"])

    def test_meme_graine_meme_manifeste(self):
        construire_manifestes(self.config, self.seuils, self.sortie)
        premier, _ = self.lire("train")
        construire_manifestes(self.config, self.seuils, self.sortie)
        self.assertEqual(premier, self.lire("train")[0])

    def test_erreur_si_dossier_xml_absent(self):
        self.config["sources"]["racine_mgb2"] = str(self.dossier / "absent")
        with self.assertRaises(ErreurManifeste):
            construire_manifestes(self.config, self.seuils, self.sortie)

    def test_erreur_si_wav_absent(self):
        for wav in (self.racine / "wav").glob("*.wav"):
            wav.unlink()
        with self.assertRaises(ErreurManifeste):
            construire_manifestes(self.config, self.seuils, self.sortie)

    def test_erreur_si_trop_de_segments_demandes(self):
        self.config["selection"]["nb_segments_train"] = 10_000
        with self.assertRaises(ErreurManifeste):
            construire_manifestes(self.config, self.seuils, self.sortie)

    def test_mode_essai(self):
        with contextlib.redirect_stdout(io.StringIO()):
            resume = main(["--essai"])
        self.assertEqual(resume["parties"]["train"]["tirage"]["segments"], 50)


if __name__ == "__main__":
    unittest.main()
