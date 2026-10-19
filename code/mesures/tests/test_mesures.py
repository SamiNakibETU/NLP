"""Tests unitaires complets pour l'Étape 2 : module code/mesures.

Vérifie :
1. Les scores WER et CER calculés à la main après normaliser_arabe.
2. Les scores chrF++ et BLEU sur exemples exacts, nuls et partiels.
3. Le facteur temps réel (RTF), le retard en direct et le taux de réécriture.
4. Le rééchantillonnage apparié (bootstrap 1 000 tirages) :
   - un système comparé à lui-même donne un intervalle qui contient zéro ([0.0, 0.0]) ;
   - un système systématiquement meilleur donne un intervalle qui exclut zéro.
5. L'écriture normalisée dans resultats/resultats.csv et le mode --essai.
"""

from pathlib import Path
import sys
import tempfile
import unittest

DOSSIER_MESURES = Path(__file__).resolve().parents[1]
if str(DOSSIER_MESURES) not in sys.path:
    sys.path.insert(0, str(DOSSIER_MESURES))

from reechantillonnage import comparer_systemes_bootstrap  # noqa: E402
from scores import (  # noqa: E402
    calculer_chrf_bleu,
    calculer_comet,
    calculer_retard_et_reecriture,
    calculer_rtf,
    calculer_wer_cer,
)
from tableau_resultats import (  # noqa: E402
    COLONNES_RESULTATS,
    enregistrer_resultat,
    executer_mode_essai,
    lire_resultats,
    main as cli_main,
)


class TestScoresManuels(unittest.TestCase):
    """Vérifie chaque formule de score sur des petits exemples calculables à la main."""

    def test_wer_cer_identique_apres_normalisation(self):
        """Deux graphies arabes équivalentes (hamza, ta marbuta, harakat) ont WER = 0 et CER = 0."""
        ref = ["أَعْلَنَ الرَّئِيسُ الْفَرَنْسِيُّ خطةً جَدِيدَةً."]
        hyp = ["اعلن الرئيس الفرنسي خطه جديده"]
        res = calculer_wer_cer(ref, hyp, niveau="complet")
        self.assertEqual(res["wer"], 0.0)
        self.assertEqual(res["cer"], 0.0)
        self.assertEqual(res["wer_pct"], 0.0)
        self.assertEqual(res["cer_pct"], 0.0)

    def test_wer_cer_calcul_a_la_main(self):
        """Sur 4 mots de référence dont 1 est substitué, WER = 1/4 = 0.25 (25.0 %)."""
        ref = ["ذهب الولد إلى المدرسة"]
        hyp = ["ذهب الرجل الى المدرسه"]
        res = calculer_wer_cer(ref, hyp, niveau="complet")
        self.assertAlmostEqual(res["wer"], 0.25, places=6)
        self.assertAlmostEqual(res["wer_pct"], 25.0, places=2)
        self.assertAlmostEqual(res["cer"], round(3.0 / 21.0, 6), places=5)

    def test_wer_moitie_faux(self):
        """Sur 4 mots dont 2 supprimés, WER = 2/4 = 0.50 (50.0 %)."""
        ref = ["قال الوزير في المؤتمر"]
        hyp = ["قال الوزير"]
        res = calculer_wer_cer(ref, hyp, niveau="complet")
        self.assertAlmostEqual(res["wer"], 0.5, places=6)
        self.assertAlmostEqual(res["wer_pct"], 50.0, places=2)

    def test_chrf_bleu_identique_et_different(self):
        """Une traduction identique à la référence donne chrF++ = 100 et BLEU = 100."""
        refs = ["Le président a annoncé un accord économique majeur à Paris."]
        hyps_exact = ["Le président a annoncé un accord économique majeur à Paris."]
        res_exact = calculer_chrf_bleu(refs, hyps_exact)
        self.assertAlmostEqual(res_exact["chrf_pp"], 100.0, places=1)
        self.assertAlmostEqual(res_exact["bleu"], 100.0, places=1)
        self.assertIn("nrefs:1", res_exact["signature_chrf_pp"])
        self.assertIn("tok:13a", res_exact["signature_bleu"])

        hyps_partiel = ["Le président a annoncé un accord économique important à Paris."]
        res_partiel = calculer_chrf_bleu(refs, hyps_partiel)
        self.assertGreater(res_partiel["chrf_pp"], 70.0)
        self.assertLess(res_partiel["chrf_pp"], 100.0)
        self.assertGreater(res_partiel["bleu"], 50.0)
        self.assertLess(res_partiel["bleu"], 100.0)

    def test_rtf_exact(self):
        """2,5 s de calcul pour 10 s d'audio donne RTF = 0,25."""
        self.assertAlmostEqual(calculer_rtf(2.5, 10.0), 0.25, places=4)
        self.assertAlmostEqual(calculer_rtf(0.0, 5.0), 0.0, places=4)
        with self.assertRaises(ValueError):
            calculer_rtf(1.0, 0.0)
        with self.assertRaises(ValueError):
            calculer_rtf(-1.0, 5.0)

    def test_retard_et_reecriture_a_la_main(self):
        """Calcul exact du retard et du taux de réécriture sur 2 segments simulés."""
        segments = [
            {
                "fin_parole_s": 3.0,
                "affichage_definitif_s": 5.5,
                "passes_provisoires": ["le ministre", "le président parle"],
                "texte_final": "le président parle",
            },
            {
                "fin_parole_s": 10.0,
                "affichage_definitif_s": 13.5,
                "passes_provisoires": [],
                "texte_final": "merci à tous",
            },
        ]
        res = calculer_retard_et_reecriture(segments)
        self.assertAlmostEqual(res["retard_moyen_s"], 3.0, places=3)
        self.assertAlmostEqual(res["retard_max_s"], 3.5, places=3)
        self.assertAlmostEqual(res["taux_reecriture"], 0.2, places=4)
        self.assertAlmostEqual(res["taux_reecriture_pct"], 20.0, places=2)

    def test_comet_repli_explicite_non_mesure(self):
        """Si COMET ne peut pas tourner localement, renvoie explicitement 'non mesuré'."""
        res = calculer_comet(
            sources_ar=["مرحبا"],
            hypotheses_fr=["Bonjour"],
            references_fr=["Bonjour"],
        )
        self.assertIn("comet", res)
        self.assertTrue(isinstance(res["comet"], float) or res["comet"] == "non mesuré")


class TestReechantillonnageApparie(unittest.TestCase):
    """Vérifie le rééchantillonnage apparié (bootstrap à 1 000 tirages)."""

    def test_systeme_compare_a_lui_meme_contient_zero(self):
        """Exigence de l'Étape 2 : comparer un système à lui-même rend un intervalle qui contient zéro."""
        refs = [
            "Le gouvernement a présenté son budget annuel devant le parlement.",
            "Les ministres européens se réunissent à Bruxelles ce matin.",
            "La croissance économique devrait atteindre trois pour cent cette année.",
            "Les négociations diplomatiques se poursuivent sous l'égide des Nations unies.",
            "Un nouvel accord commercial a été signé entre les deux pays.",
        ]
        hyps = [
            "Le gouvernement a présenté le budget annuel au parlement.",
            "Les ministres européens se réunissent à Bruxelles aujourd'hui.",
            "La croissance économique va atteindre trois pour cent cette année.",
            "Les discussions diplomatiques continuent sous l'égide de l'ONU.",
            "Un accord commercial nouveau a été signé entre les deux pays.",
        ]
        res = comparer_systemes_bootstrap(
            references=refs,
            hypotheses_a=hyps,
            hypotheses_b=hyps,
            metrique="chrf++",
            n_tirages=1000,
            graine=42,
        )
        self.assertEqual(res["ecart_observe"], 0.0)
        self.assertEqual(res["ic_95_bas"], 0.0)
        self.assertEqual(res["ic_95_haut"], 0.0)
        self.assertTrue(res["contient_zero"])
        self.assertFalse(res["exclut_zero"])

    def test_systeme_nettement_meilleur_exclut_zero(self):
        """Si B est exact partout et A est médiocre partout, l'intervalle à 95 % exclut zéro."""
        refs = [
            "Le président français a annoncé une réforme importante ce soir.",
            "La réunion internationale a débuté au Caire en présence des ministres.",
            "Les exportations de pétrole ont augmenté au cours du dernier trimestre.",
            "Le conseil de sécurité examine la résolution proposée par l'Algérie.",
            "Les élections législatives auront lieu le mois prochain dans tout le pays.",
            "La banque centrale a décidé de maintenir ses taux d'intérêt inchangés.",
        ]
        hyps_a = [
            "Président annonce chose.",
            "Réunion commence Caire.",
            "Pétrole monte trimestre.",
            "Conseil regarde texte.",
            "Élections mois prochain.",
            "Banque garde taux.",
        ]
        hyps_b = list(refs)

        res = comparer_systemes_bootstrap(
            references=refs,
            hypotheses_a=hyps_a,
            hypotheses_b=hyps_b,
            metrique="chrf++",
            n_tirages=1000,
            graine=42,
        )
        self.assertGreater(res["ecart_observe"], 30.0)
        self.assertGreater(res["ic_95_bas"], 0.0)
        self.assertFalse(res["contient_zero"])
        self.assertTrue(res["exclut_zero"])


class TestTableauResultats(unittest.TestCase):
    """Vérifie l'écriture et la lecture de resultats/resultats.csv."""

    def test_enregistrement_et_lecture_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chemin_csv = Path(tmpdir) / "resultats.csv"
            enregistrer_resultat(
                systeme="S0",
                version_modele="whisper-small+opus-mt-ar-fr",
                jeu="fleurs_ar_fr",
                mesure="chrF++",
                valeur=45.67,
                intervalle="[+0.00, +0.00]",
                configuration="configs/s0.yaml",
                chemin_csv=chemin_csv,
            )
            lignes = lire_resultats(chemin_csv)
            self.assertEqual(len(lignes), 1)
            self.assertEqual(list(lignes[0].keys()), COLONNES_RESULTATS)
            self.assertEqual(lignes[0]["systeme"], "S0")
            self.assertEqual(lignes[0]["valeur"], "45.67")

    def test_evaluer_sorties_et_mode_essai(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chemin_csv = Path(tmpdir) / "resultats_essai.csv"
            res = executer_mode_essai(chemin_csv=chemin_csv)
            self.assertTrue(res["bootstrap_auto_contient_zero"])
            lignes = lire_resultats(chemin_csv)
            self.assertGreaterEqual(len(lignes), 5)
            mesures_trouvees = {l["mesure"] for l in lignes}
            self.assertIn("WER (%)", mesures_trouvees)
            self.assertIn("CER (%)", mesures_trouvees)
            self.assertIn("chrF++", mesures_trouvees)
            self.assertIn("BLEU", mesures_trouvees)
            self.assertIn("RTF", mesures_trouvees)

            code_retour = cli_main(["--essai", "--csv", str(chemin_csv)])
            self.assertEqual(code_retour, 0)

    def test_evaluation_finale_grille_aveugle_et_protection_jeu_t(self):
        from evaluation_finale import comparer_tous_les_systemes, executer_essai_evaluation_finale

        with tempfile.TemporaryDirectory() as tmpdir:
            res = executer_essai_evaluation_finale(dossier_sortie=tmpdir)
            self.assertEqual(res["grille_aveugle"]["nb_extraits_anonymises"], 30)
            self.assertEqual(res["taxonomie_erreurs"]["nb_erreurs_classees"], 21)

            # Le lancement sur T sans autorisation explicite doit lever PermissionError
            with self.assertRaises(PermissionError):
                comparer_tous_les_systemes(
                    fiches_reference=[{"id": "1", "ar": "مرحبا", "fr": "Bonjour", "duree_s": 2.0}],
                    sorties_par_systeme={"S2": [{"id": "1", "fr_traduit": "Bonjour", "temps_calcul_s": 0.5}]},
                    jeu="T",
                    dossier_sortie=tmpdir,
                    autoriser_jeu_t=False,
                )


if __name__ == "__main__":
    unittest.main()
