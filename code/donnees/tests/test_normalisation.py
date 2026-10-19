"""Tests de la normalisation du texte arabe pour les mesures."""

import sys
import unittest
from pathlib import Path

DOSSIER_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOSSIER_CODE))

from normalisation import (  # noqa: E402
    compter_mots_arabe,
    distance_mots,
    normaliser_arabe,
    normaliser_fr,
    taux_ecart_ecoute,
)


class NormalisationArabeComplet(unittest.TestCase):

    def test_retire_diacritiques(self):
        self.assertEqual(normaliser_arabe("مُحَمَّد"), "محمد")

    def test_retire_tatweel(self):
        self.assertEqual(normaliser_arabe("السلامـــــ عليكم"), "السلام عليكم")

    def test_unifie_hamzas_sur_alef(self):
        # أ إ آ tombent sur alef au niveau complet (hamza sans support distingue).
        self.assertEqual(normaliser_arabe("أ"), "ا")
        self.assertEqual(normaliser_arabe("إ"), "ا")
        self.assertEqual(normaliser_arabe("آ"), "ا")
        self.assertEqual(normaliser_arabe("ٱ"), "ا")

    def test_unifie_ta_marbuta_et_alef_maqsura(self):
        self.assertEqual(normaliser_arabe("مدرسة"), "مدرسه")
        self.assertEqual(normaliser_arabe("على"), "علي")

    def test_convertit_les_chiffres_orientaux(self):
        self.assertEqual(normaliser_arabe("في عام ٢٠٢٦"), "في عام 2026")

    def test_retire_la_ponctuation_arabe_et_latine(self):
        self.assertEqual(normaliser_arabe("قال هذا؟"), "قال هذا")
        self.assertEqual(normaliser_arabe("واحد، اثنان."), "واحد اثنان")
        self.assertEqual(normaliser_arabe("«قال-هذا»"), "قال هذا")


    def test_reduit_les_espaces_et_marques_bidi(self):
        self.assertEqual(normaliser_arabe("\u200f طقس  اليوم \n"), "طقس اليوم")

    def test_deux_graphies_egales_donnennt_le_meme_texte(self):
        # meme mot, ecrit avec ou sans diacritiques et hamza supportee
        self.assertEqual(
            normaliser_arabe("أَحْمَد"),
            normaliser_arabe("احمد"),
        )

    def test_niveau_inconnu_refuse(self):
        with self.assertRaises(ValueError):
            normaliser_arabe("texte", niveau="bizarre")

    def test_entree_non_texte_refusee(self):
        with self.assertRaises(TypeError):
            normaliser_arabe(12345)


class NormalisationArabeLeger(unittest.TestCase):

    def test_garde_les_hamzas_distinctes(self):
        # Au niveau leger, alef avec hamza reste distincte d'alef nue.
        self.assertNotEqual(normaliser_arabe("أ", niveau="leger"), "ا")

    def test_retire_quand_meme_diacritiques(self):
        self.assertEqual(normaliser_arabe("مُحَمَّد", niveau="leger"), "محمد")


class ComptageEtEcart(unittest.TestCase):

    REF = "ذهب الولد الى المدرسة صباح اليوم"
    # 6 mots une fois normalises : ذهب الولد الي المدرسه صباح اليوم

    def test_comptage_de_mots(self):
        self.assertEqual(compter_mots_arabe(self.REF), 6)

    def test_comptage_ignores_vide(self):
        self.assertEqual(compter_mots_arabe(""), 0)

    def test_ecart_nulle_si_identique(self):
        self.assertEqual(taux_ecart_ecoute(self.REF, self.REF), 0.0)

    def test_ecart_un_mot_sur_six(self):
        ecoute = "ذهب البنت الى المدرسة صباح اليوم"  # الولد -> البنت
        self.assertAlmostEqual(taux_ecart_ecoute(self.REF, ecoute), 1 / 6)

    def test_ecart_avec_un_mot_en_plus(self):
        ecoute = "ذهب الولد الى المدرسة صباح اليوم السابق"  # un mot insere
        self.assertAlmostEqual(taux_ecart_ecoute(self.REF, ecoute), 1 / 6)

    def test_ecart_normalise_les_graphies_avant_comparaison(self):
        # meme texte ecrit avec diacritiques et ta marbuta differente : ecart nul
        self.assertEqual(
            taux_ecart_ecoute("المدرسة", "المدرسَة"), 0.0
        )

    def test_ecart_zero_avec_ponctuation_et_hamzas_defaut_1(self):
        # Defaut 1 du lot 1 : ponctuation collee aux mots retiree pour les mesures
        ref = "قال الوزير ان المفاوضات مستمرة حتى نهاية الشهر"
        candidat = "قال الوزير، إن المفاوضات مستمرة حتى نهاية الشهر."
        self.assertEqual(taux_ecart_ecoute(ref, candidat), 0.0)


    def test_ecart_reference_vide_refusee(self):
        with self.assertRaises(ValueError):
            taux_ecart_ecoute("", "un mot")

    def test_distance_mots_directe(self):
        self.assertEqual(distance_mots("a b c", "a x c"), 1)
        self.assertEqual(distance_mots("", "a b"), 2)


class NormalisationFrancais(unittest.TestCase):

    def test_minuscules_apostrophe_espaces(self):
        self.assertEqual(normaliser_fr("L’ONU   a dit : \"non\" "), "l'onu a dit : \"non\"")

    def test_entree_non_texte_refusee(self):
        with self.assertRaises(TypeError):
            normaliser_fr(None)


if __name__ == "__main__":
    unittest.main()
