"""Le direct n'affiche jamais d'écriture arabe dans le sous-titre français, sans modèle."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_moteurs import MoteurDirectS3, part_arabe, sans_arabe  # noqa: E402


class FauxTranscripteur:
    def __init__(self, sortie):
        self.sortie = sortie

    def traduire_avec_contexte(self, ar, contexte):
        return self.sortie


class FauxOpus:
    def traduire(self, ar):
        return "Le ministre est arrivé à Beyrouth."


def moteur(sortie_qwen):
    m = MoteurDirectS3.__new__(MoteurDirectS3)
    m.t, m.opus, m.lora_actif = FauxTranscripteur(sortie_qwen), FauxOpus(), False
    return m


class TestNettoyage(unittest.TestCase):
    def test_retire_un_nom_recopie(self):
        self.assertEqual(sans_arabe("Le ministre نجيب ميقاتي a parlé ."), "Le ministre a parlé.")

    def test_part_arabe(self):
        self.assertEqual(part_arabe("Bonjour"), 0.0)
        self.assertGreater(part_arabe("أهلا بكم في نشرة الأخبار"), 0.9)

    def test_qwen_qui_recopie_passe_la_main_a_opus(self):
        fr = moteur("وصل الوزير إلى بيروت").traduire_definitif("وصل الوزير إلى بيروت", [])
        self.assertEqual(fr, "Le ministre est arrivé à Beyrouth.")

    def test_bonne_traduction_gardee(self):
        fr = moteur("Le ministre est arrivé hier soir à Beyrouth.").traduire_definitif("...", [])
        self.assertEqual(fr, "Le ministre est arrivé hier soir à Beyrouth.")

    def test_gris_sans_arabe(self):
        m = moteur("")
        m.opus = type("O", (), {"traduire": lambda self, ar: "Le président بري a dit"})()
        self.assertEqual(m.traduire_provisoire("..."), "Le président a dit")


if __name__ == "__main__":
    unittest.main()
