"""Tests du flux incrémental (paquets, gris puis blanc, retard, contexte) avec le moteur factice, sans modèle."""

import sys
import unittest
from pathlib import Path

import numpy as np

RACINE_PROJET = Path(__file__).resolve().parents[3]
for dossier in (RACINE_PROJET / "code" / "direct", RACINE_PROJET / "speech_app"):
    if str(dossier) not in sys.path:
        sys.path.insert(0, str(dossier))

from flux_incremental import FluxIncremental  # noqa: E402
from live_moteurs import PHRASES, SR, MoteurFactice, signal_demo  # noqa: E402


class Horloge:
    """Horloge pilotée par l'audio reçu : 1 s d'audio = 1 s d'horloge, calcul instantané."""
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def rejouer(signal, moteur=None, paquet_s=0.25, horloge=None):
    horloge = horloge or Horloge()
    moteur = moteur or MoteurFactice()
    flux = FluxIncremental(moteur, horloge=horloge)
    messages, pas = [], int(paquet_s * SR)
    for debut in range(0, len(signal), pas):
        horloge.t += paquet_s
        flux.ajouter(signal[debut: debut + pas])
        messages += flux.traiter()
    messages += flux.terminer()
    return flux, moteur, messages


class TestFluxIncremental(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flux, cls.moteur, cls.messages = rejouer(signal_demo([0, 1, 2], duree_phrase_s=6.0))
        cls.definitifs = [m for m in cls.messages if m["type"] == "definitif"]
        cls.provisoires = [m for m in cls.messages if m["type"] == "provisoire"]

    def test_une_phrase_definitive_par_phrase(self):
        self.assertEqual([m["fr"] for m in self.definitifs], [PHRASES[i][2] for i in range(3)])
        self.assertEqual([m["id_segment"] for m in self.definitifs], ["seg_0001", "seg_0002", "seg_0003"])
        self.assertTrue(all(m["ar"] for m in self.definitifs))

    def test_gris_puis_blanc_dans_chaque_segment(self):
        for d in self.definitifs:
            avant = [m for m in self.provisoires if m["id_segment"] == d["id_segment"]]
            self.assertGreaterEqual(len(avant), 2, "plusieurs passes grises avant la clôture")
            index_def = self.messages.index(d)
            self.assertTrue(all(self.messages.index(p) < index_def for p in avant), "aucun gris après le blanc")
            self.assertNotEqual(avant[-1]["fr"], d["fr"], "le blanc diffère de la traduction rapide")

    def test_provisoire_grandit(self):
        premier = [m for m in self.provisoires if m["id_segment"] == "seg_0001"]
        longueurs = [len(m["ar"].split()) for m in premier]
        self.assertEqual(longueurs, sorted(longueurs))
        self.assertGreater(longueurs[-1], longueurs[0])

    def test_pas_de_doublon_et_segments_ordonnes(self):
        ids = [m["id_segment"] for m in self.definitifs]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(b["debut_s"] >= a["fin_s"] - 1.0 for a, b in zip(self.definitifs, self.definitifs[1:])))
        texte = " ".join(m["fr"] for m in self.definitifs)
        self.assertEqual(texte.count("Bonsoir"), 1)

    def test_contexte_des_deux_phrases_precedentes(self):
        appels = self.moteur.appels_definitif
        self.assertEqual(len(appels), 3)
        self.assertEqual(appels[0][1], [])
        self.assertEqual(appels[1][1], [appels[0][0]])
        self.assertEqual(appels[2][1], [appels[0][0], appels[1][0]][-2:])

    def test_retard_mesure_a_l_horloge(self):
        for d in self.definitifs:
            # silence de clôture (0,5 s) + au plus un paquet (0,25 s) ; calcul nul avec l'horloge simulée
            self.assertGreaterEqual(d["retard_s"], 0.4)
            self.assertLessEqual(d["retard_s"], 1.0)
        bilan = self.flux.bilan()
        self.assertEqual(bilan["nb_segments"], 3)
        self.assertLessEqual(bilan["retard_max_s"], 1.0)
        self.assertGreater(bilan["taux_reecriture_pct"], 0.0)   # le gris (glose) est réécrit par le blanc

    def test_fin_de_flux_clot_le_segment_en_cours(self):
        signal = np.concatenate([np.zeros(SR, dtype=np.float32), signal_demo([1], 4.0)[SR:]])[: 5 * SR]
        _, _, messages = rejouer(signal)
        definitifs = [m for m in messages if m["type"] == "definitif"]
        self.assertEqual(len(definitifs), 1)
        self.assertTrue(definitifs[0]["fr"])

    def test_silence_seul_ne_produit_rien(self):
        flux, _, messages = rejouer(np.zeros(6 * SR, dtype=np.float32))
        self.assertEqual(messages, [])
        self.assertEqual(flux.bilan(), {"nb_segments": 0})

    def test_coupure_d_office_sans_pause(self):
        _, _, messages = rejouer(np.concatenate([np.zeros(SR // 2, dtype=np.float32), signal_demo([2], 20.0)]))
        definitifs = [m for m in messages if m["type"] == "definitif"]
        self.assertGreaterEqual(len(definitifs), 2)
        self.assertLessEqual(definitifs[0]["fin_s"] - definitifs[0]["debut_s"], 15.5)

    def test_paquets_pcm16(self):
        flux = FluxIncremental(MoteurFactice(), horloge=Horloge())
        pcm = (np.sin(np.arange(4000) / 10) * 8000).astype("<i2").tobytes()
        flux.ajouter_pcm16(pcm + b"\x01")        # octet impair ignoré
        self.assertEqual(flux.n, 4000)
        self.assertAlmostEqual(float(flux.signal().max()), 8000 / 32768, places=2)

    def test_paquets_de_tailles_irregulieres_donnent_le_meme_resultat(self):
        signal = signal_demo([0, 1], 6.0)
        _, _, a = rejouer(signal, paquet_s=0.25)
        _, _, b = rejouer(signal, paquet_s=0.5)
        fr = lambda ms: [m["fr"] for m in ms if m["type"] == "definitif"]  # noqa: E731
        self.assertEqual(fr(a), fr(b))


if __name__ == "__main__":
    unittest.main()
