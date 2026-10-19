"""Tests unitaires pour le mode direct (confirmation 2 passes, probabilité, secours, 10 min simulées et serveur)."""

from pathlib import Path
import sys
import tempfile
import unittest

RACINE_PROJET = Path(__file__).resolve().parents[3]
DOSSIER_DIRECT = RACINE_PROJET / "code" / "direct"
if str(DOSSIER_DIRECT) not in sys.path:
    sys.path.insert(0, str(DOSSIER_DIRECT))

from confirmation import (  # noqa: E402
    ConfirmateurDeuxPasses,
    DecideurProbabiliteOui,
    ModeSecoursParMorceaux,
)
from flux_simule import simuler_flux_direct  # noqa: E402
from serveur_direct import PAGE_HTML_DIRECT, construire_serveur_direct  # noqa: E402


class TestModeDirect(unittest.TestCase):
    """Vérifie le mode direct, les 10 minutes de flux simulé et le serveur WebSocket."""

    def test_solution_secours_par_morceaux(self):
        """La solution de secours n'affiche le sous-titre qu'à la clôture du segment."""
        secours = ModeSecoursParMorceaux()
        m1 = secours.traiter_passe("seg_1", "Début provisoire", est_clos=False, horodatage_s=1.0)
        self.assertEqual(m1["texte_confirme"], "")
        self.assertEqual(m1["texte_provisoire"], "")

        m2 = secours.traiter_passe("seg_1", "Phrase complète traduite.", est_clos=True, horodatage_s=3.0)
        self.assertEqual(m2["texte_confirme"], "Phrase complète traduite.")
        self.assertEqual(m2["texte_provisoire"], "")
        self.assertTrue(m2["est_clos"])

    def test_regle_deux_passes_immuabilite_mots_confirmes(self):
        """Un mot occupé à la même place dans deux passes successives est confirmé et ne change plus."""
        conf = ConfirmateurDeuxPasses()
        # Passe 1 : tout est provisoire
        r1 = conf.mettre_a_jour("seg_1", "Le ministre annonce un", est_clos=False, horodatage_s=1.0)
        self.assertEqual(r1["texte_confirme"], "")
        self.assertEqual(r1["texte_provisoire"], "Le ministre annonce un")

        # Passe 2 : "Le ministre annonce" reste identique -> confirmé ; "une mesure" est provisoire
        r2 = conf.mettre_a_jour("seg_1", "Le ministre annonce une mesure", est_clos=False, horodatage_s=2.0)
        self.assertEqual(r2["texte_confirme"], "Le ministre annonce")
        self.assertEqual(r2["texte_provisoire"], "une mesure")

        # Passe 3 : même si le modèle change le 1er mot en "Un", "Le ministre annonce" reste figé
        r3 = conf.mettre_a_jour("seg_1", "Un ministre annonce une mesure urgente", est_clos=False, horodatage_s=3.0)
        self.assertTrue(r3["texte_confirme"].startswith("Le ministre annonce"))

    def test_variante_probabilite_oui(self):
        """Vérifie le calcul softmax de P('oui') et le déclenchement au-dessus du seuil."""
        dec = DecideurProbabiliteOui(seuil_probabilite_oui=0.65)
        p_haute = dec.calculer_probabilite_oui_depuis_logits(logit_oui=2.0, logit_non=0.0)
        self.assertGreater(p_haute, 0.85)
        self.assertTrue(dec.doit_traduire(p_haute, est_clos=False)["traduire_maintenant"])
        self.assertFalse(dec.doit_traduire(0.40, est_clos=False)["traduire_maintenant"])
        self.assertTrue(dec.doit_traduire(0.40, est_clos=True)["traduire_maintenant"])

    def test_simulation_10_minutes_sans_plantage_et_ecriture_csv(self):
        """Dix minutes (600 s) de son simulé passent sans plantage avec retard <= 5 s et écriture CSV."""
        with tempfile.TemporaryDirectory() as tmp:
            chemin_csv = Path(tmp) / "resultats_direct.csv"
            res = simuler_flux_direct(
                duree_totale_s=600.0,
                mode_decision="deux_passes",
                systeme="S2",
                enregistrer_dans_csv=True,
                chemin_csv=chemin_csv,
            )
            self.assertEqual(res["duree_simulee_s"], 600.0)
            self.assertEqual(res["nb_segments_clos"], 100)
            self.assertLessEqual(res["retard_moyen_s"], 3.0)
            self.assertTrue(res["sous_plafond_5s"])
            self.assertEqual(len(res["dernier_contexte_2_phrases"]), 2)
            self.assertIn("Paris", res["noms_propres_memorises"])
            self.assertTrue(chemin_csv.exists())

    def test_page_web_et_serveur_fastapi(self):
        """Vérifie que la page web respecte les 2 lignes, blanc/gris, LTR et que le serveur se construit."""
        self.assertIn('dir="ltr"', PAGE_HTML_DIRECT)
        self.assertIn("#ffffff", PAGE_HTML_DIRECT)
        self.assertIn("#9ca3af", PAGE_HTML_DIRECT)
        self.assertIn("-webkit-line-clamp: 2", PAGE_HTML_DIRECT)
        app = construire_serveur_direct()
        chemins = [getattr(r, "path", "") for r in app.routes]
        self.assertIn("/", chemins)
        self.assertIn("/ws/direct", chemins)


if __name__ == "__main__":
    unittest.main()
