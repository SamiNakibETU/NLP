"""Tests de la lecture des extraits (audio_s3.py) et de l'encodage S3 (encoder_s3.py), hors ligne.

Le petit Cohere Transcribe aléatoire de ``code/projecteur/factices_s3.py`` remplace le vrai modèle.
"""

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

RACINE_PROJET = Path(__file__).resolve().parents[3]
for _dossier in (RACINE_PROJET / "code" / "ecoute", RACINE_PROJET / "code" / "projecteur"):
    if str(_dossier) not in sys.path:
        sys.path.insert(0, str(_dossier))

from audio_s3 import ErreurAudio, charger_extrait  # noqa: E402
from encoder_s3 import (  # noqa: E402
    EncodeurS3,
    ErreurEncodage,
    charger_config,
    ecrire_audio_factice,
    executer,
    former_lots,
    lire_jsonl,
    lire_vecteurs,
    main,
    nb_vecteurs_theoriques,
)
from factices_s3 import D_ENC, petit_modele_asr, petit_preparateur  # noqa: E402

SR = 16000


def petit_encodeur():
    return EncodeurS3(petit_modele_asr(), petit_preparateur(), torch.device("cpu"), torch.float32, "essai")


class TestChargerExtrait(unittest.TestCase):
    def setUp(self):
        self.dossier = Path(tempfile.mkdtemp(prefix="test_audio_s3_"))
        (self.dossier / "wav").mkdir()
        # Rampe : l'échantillon i vaut i / 2**15 (exact en PCM 16 bits), on retrouve donc le morceau lu.
        self.signal = (np.arange(5 * SR) % 30000).astype(np.int16)
        sf.write(str(self.dossier / "wav" / "e.wav"), self.signal, SR, subtype="PCM_16")

    def tearDown(self):
        shutil.rmtree(self.dossier, ignore_errors=True)

    def test_longueur_et_contenu(self):
        entree = {"wav": "wav/e.wav", "debut": 1.23, "fin": 3.71}
        extrait = charger_extrait(entree, self.dossier)
        self.assertLessEqual(abs(len(extrait) - (3.71 - 1.23) * SR), 1)
        attendu = self.signal[round(1.23 * SR): round(3.71 * SR)].astype(np.float32) / 32768
        np.testing.assert_array_equal(extrait, attendu)
        self.assertEqual(extrait.dtype, np.float32)

    def test_tolerance_de_fin(self):
        """Une fin un peu après la fin du wav est coupée à la fin du fichier, dans la tolérance seulement."""
        entree = {"wav": "wav/e.wav", "debut": 4.5, "fin": 5.06}
        with self.assertRaises(ErreurAudio):
            charger_extrait(entree, self.dossier)
        extrait = charger_extrait(entree, self.dossier, tolerance_fin_s=0.1)
        self.assertEqual(len(extrait), 5 * SR - round(4.5 * SR))
        with self.assertRaises(ErreurAudio):
            charger_extrait({"wav": "wav/e.wav", "debut": 4.5, "fin": 5.2}, self.dossier, tolerance_fin_s=0.1)

    def test_erreurs_explicites(self):
        with self.assertRaises(ErreurAudio):
            charger_extrait({"wav": "wav/absent.wav", "debut": 0, "fin": 1}, self.dossier)
        with self.assertRaises(ErreurAudio):
            charger_extrait({"wav": "wav/e.wav", "debut": 4, "fin": 6}, self.dossier)  # au-delà de la fin
        sf.write(str(self.dossier / "wav" / "8k.wav"), np.zeros(8000, dtype=np.int16), 8000)
        with self.assertRaises(ErreurAudio):
            charger_extrait({"wav": "wav/8k.wav", "debut": 0, "fin": 0.5}, self.dossier)
        sf.write(str(self.dossier / "wav" / "stereo.wav"), np.zeros((SR, 2), dtype=np.int16), SR)
        with self.assertRaises(ErreurAudio):
            charger_extrait({"wav": "wav/stereo.wav", "debut": 0, "fin": 0.5}, self.dossier)


class TestEncodeur(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encodeur = petit_encodeur()
        generateur = np.random.default_rng(0)
        cls.extraits = [(0.1 * generateur.standard_normal(int(d * SR))).astype(np.float32) for d in (1.3, 2.0, 3.77)]

    def test_encodeur_gele_et_en_eval(self):
        self.assertFalse(self.encodeur.encodeur.training)
        self.assertTrue(all(not p.requires_grad for p in self.encodeur.encodeur.parameters()))
        self.assertEqual(type(self.encodeur.encodeur).__name__, "ParakeetEncoder")
        self.assertEqual(self.encodeur.d_enc, D_ENC)  # lu dans la config, pas écrit en dur

    def test_formule_du_nombre_de_vecteurs(self):
        for extrait, v in zip(self.extraits, self.encodeur.encoder_lot(self.extraits)):
            self.assertEqual(v.shape, (self.encodeur.nb_vecteurs_attendus(len(extrait)), D_ENC))
            self.assertEqual(v.dtype, np.float16)
        # n // 160 trames, puis 3 fois ceil(L / 2) : 10 s -> 1000 -> 500 -> 250 -> 125 (12,5 Hz)
        config = self.encodeur.encodeur.config
        self.assertEqual(nb_vecteurs_theoriques(10 * SR, self.encodeur.preparateur, config), 125)
        self.assertEqual(nb_vecteurs_theoriques(5 * SR, self.encodeur.preparateur, config), 63)

    def test_seul_et_en_lot_identiques(self):
        en_lot = self.encodeur.encoder_lot(self.extraits)
        for extrait, v in zip(self.extraits, en_lot):
            seul = self.encodeur.encoder_lot([extrait])[0]
            self.assertEqual(seul.shape, v.shape)
            np.testing.assert_allclose(seul.astype(np.float32), v.astype(np.float32), atol=1e-2, rtol=1e-2)

    def test_pas_de_decoupage_a_25_s(self):
        """Le préparateur découpe au-delà de max_audio_clip_s - overlap (30 s) : jamais pour nos 25 s max."""
        self.assertEqual(self.encodeur.preparateur.max_audio_clip_s - self.encodeur.preparateur.overlap_chunk_second, 30.0)
        long = np.zeros(25 * SR, dtype=np.float32)
        entree = self.encodeur.preparateur([long], sampling_rate=SR, return_attention_mask=True, return_tensors="pt")
        self.assertEqual(entree["audio_chunk_index"], [(0, None)])
        with self.assertRaises(ErreurEncodage):
            self.encodeur.encoder_lot([np.zeros(32 * SR, dtype=np.float32)])

    def test_former_lots(self):
        durees = [10, 3, 24, 5, 7, 9]
        lots = former_lots(durees, 20)
        self.assertEqual(sorted(i for lot in lots for i in lot), list(range(len(durees))))
        for lot in lots:
            self.assertTrue(len(lot) == 1 or sum(durees[i] for i in lot) <= 20)
        self.assertEqual([[durees[i] for i in lot] for lot in lots], [[3, 5, 7], [9, 10], [24]])


class TestEncodagePartie(unittest.TestCase):
    def setUp(self):
        self.dossier = Path(tempfile.mkdtemp(prefix="test_encoder_s3_"))
        self.racine = self.dossier / "mgb2"
        self.manifeste = ecrire_audio_factice(self.racine, nb_segments=11)
        self.config = charger_config()
        self.config["encodage"].update(segments_par_fichier_npy=4, secondes_audio_par_lot=8)
        self.sortie = self.dossier / "sortie"
        self.encodeur = petit_encodeur()
        self.messages = []

    def tearDown(self):
        shutil.rmtree(self.dossier, ignore_errors=True)

    def lancer(self, manifeste=None):
        return executer(self.config, self.encodeur, manifeste or self.manifeste, self.racine, self.sortie, 42,
                        journal=self.messages.append)

    def test_stockage_index_et_rapport(self):
        rapport = self.lancer()
        index = lire_jsonl(self.sortie / "index.jsonl")
        self.assertEqual([l["id"] for l in index], [e["id"] for e in self.manifeste])
        self.assertEqual(sorted(p.name for p in self.sortie.glob("vecteurs_*.npy")),
                         ["vecteurs_00000.npy", "vecteurs_00001.npy", "vecteurs_00002.npy"])
        self.assertFalse(list(self.sortie.glob("*.tmp")))
        for ligne in index:
            self.assertEqual((ligne["encodeur"], ligne["d_enc"]), ("essai", D_ENC))
            self.assertEqual(ligne["texte_ar"], "نص تجريبي")
            stocke = lire_vecteurs(self.sortie, ligne)
            self.assertEqual(stocke.dtype, np.float16)
            seul = self.encodeur.encoder_lot([charger_extrait(ligne, self.racine)])[0]
            np.testing.assert_allclose(stocke.astype(np.float32), seul.astype(np.float32), atol=1e-2, rtol=1e-2)
        self.assertTrue(rapport["conforme"])
        self.assertEqual(rapport["valeurs_non_finies"], 0)
        self.assertEqual(rapport["somme_nb_vecteurs_index"], rapport["somme_lignes_fichiers"])
        self.assertEqual(len(rapport["reencodage_seul"]), 3)
        self.assertTrue(json.loads((self.sortie / "rapport_encodage.json").read_text(encoding="utf-8"))["conforme"])

    def test_reprise_apres_coupure(self):
        self.lancer()
        npy_1 = (self.sortie / "vecteurs_00001.npy").read_bytes()
        index = lire_jsonl(self.sortie / "index.jsonl")
        # Coupure : dernier fichier absent, lignes d'index du fichier 1 incomplètes.
        (self.sortie / "vecteurs_00002.npy").unlink()
        with open(self.sortie / "index.jsonl", "w", encoding="utf-8") as fichier:
            for ligne in index[:6]:
                fichier.write(json.dumps(ligne, ensure_ascii=False) + "\n")
        self.messages.clear()
        self.lancer()
        self.assertIn("1/3 fichier(s) déjà faits", self.messages)
        self.assertEqual(lire_jsonl(self.sortie / "index.jsonl"), index)
        self.assertEqual((self.sortie / "vecteurs_00001.npy").read_bytes(), npy_1)

    def test_tirage_plus_grand_reutilise_les_fichiers(self):
        """Tirage emboîté : le manifeste plus long commence par le plus court, ses fichiers complets sont repris."""
        self.lancer(self.manifeste[:8])
        self.messages.clear()
        self.lancer()
        self.assertIn("2/3 fichier(s) déjà faits", self.messages)

    def test_autre_encodeur_refuse(self):
        self.lancer()
        self.encodeur.modele_id = "autre"
        with self.assertRaises(ErreurEncodage):
            self.lancer()

    def test_arret_si_disque_insuffisant(self):
        self.config["controles"]["marge_disque_go"] = 10 ** 9
        with self.assertRaises(ErreurEncodage):
            self.lancer()
        self.assertFalse(list(self.sortie.glob("vecteurs_*.npy")))

    def test_mode_essai(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rapport = main(["--essai"])
        self.assertTrue(rapport["conforme"])


if __name__ == "__main__":
    unittest.main()
