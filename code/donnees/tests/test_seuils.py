"""Tests du fichier de seuils : c'est une garde-fou du plan.

Les valeurs de seuils.yaml sont celles de plan-donnees.md. Si quelqu'un
modifie le fichier sans passer par le plan, ces tests echouent.
"""

import sys
import tempfile
import unittest
from pathlib import Path

DOSSIER_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOSSIER_CODE))

from seuils import ErreurSeuils, charger_seuils  # noqa: E402

SEUILS = charger_seuils()


class ValeursDuPlan(unittest.TestCase):
    """Les seuils 'proposes' doivent garder les valeurs du plan, rien d'autre."""

    def test_seuils_proposes_non_modifies(self):
        attends = {
            "ecart_ecoute_max_pct": 15,        # D1 filtre 3
            "rapport_longueur_min": 0.6,       # D4 filtre 3
            "rapport_longueur_max": 2.5,       # D4 filtre 3
            "qe_rejet_pire_pct": 20,           # D4 filtre 7
            "taux_erreur_sens_max_pct": 5,     # D5
        }
        for cle, valeur in attends.items():
            with self.subTest(cle=cle):
                self.assertEqual(SEUILS["proposes"][cle], valeur)

    def test_constantes_essentielles(self):
        constantes = SEUILS["constantes"]
        self.assertEqual(constantes["mots_min_segment"], 6)
        self.assertEqual(constantes["taille_paquet_traduction"], 5)
        self.assertEqual(constantes["segments_objectif"], 50000)
        self.assertEqual(constantes["unpc_max_phrases"], 20000)
        self.assertEqual(constantes["date_min_extrait"], "2026-10-02")
        self.assertEqual(constantes["duree_extrait_min_s"], 20)
        self.assertEqual(constantes["duree_extrait_max_s"], 30)
        self.assertEqual(constantes["extraits_t"], 60)
        self.assertEqual(constantes["extraits_r"], 20)
        self.assertEqual(constantes["segments_max_par_episode_a"], 600)
        self.assertEqual(constantes["heures_alignement_min"], 100)
        self.assertEqual(constantes["heures_alignement_max"], 200)


class FichierIncomplet(unittest.TestCase):

    def ecrire(self, texte):
        fichier = Path(tempfile.gettempdir()) / "_test_seuils.yaml"
        fichier.write_text(texte, encoding="utf-8")
        self.addCleanup(fichier.unlink)
        return fichier

    def test_cle_manquante_leve_erreur(self):
        fichier = self.ecrire("proposes:\n  ecart_ecoute_max_pct: 15\nconstantes: {}\n")
        with self.assertRaises(ErreurSeuils):
            charger_seuils(fichier)

    def test_mauvais_type_leve_erreur(self):
        fichier = self.ecrire(
            "proposes:\n  ecart_ecoute_max_pct: quinze\n"
            "constantes:\n  duree_segment_min_s: 5\n  duree_segment_max_s: 25\n"
            "  mots_min_segment: 6\n  segments_max_par_episode: 200\n"
            "  segments_objectif: 50000\n  segments_comparaison_modeles: 100\n"
            "  segments_mesure_vitesse: 500\n  taille_paquet_traduction: 5\n"
            "  relances_paquet_max: 1\n  extraits_t: 60\n  extraits_r: 20\n"
            "  duree_extrait_min_s: 20\n  duree_extrait_max_s: 30\n"
            "  date_min_extrait: '2026-10-02'\n  extraits_avec_fr2: 15\n"
            "  unpc_max_phrases: 20000\n  noms_min: 300\n  noms_max: 500\n"
            "  segments_controle_humain: 200\n  segments_debuts_phrase: 5000\n"
            "  coupe_debuts_pct: [30, 50, 70]\n  debuts_relecture: 100\n"
            "  episodes_validation_pct: 5\n"
            "  segments_max_par_episode_a: 600\n"
            "  heures_alignement_min: 100\n  heures_alignement_max: 200\n"
        )

        with self.assertRaises(ErreurSeuils):
            charger_seuils(fichier)

    def test_fichier_absent_leve_erreur(self):
        with self.assertRaises(ErreurSeuils):
            charger_seuils("chemin-qui-n-existe-pas.yaml")


if __name__ == "__main__":
    unittest.main()
