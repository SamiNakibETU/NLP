"""Tests du validateur de fiches (jeux T/R et D).

Fichiers d'exemples : tests/exemples/*.jsonl — fiches FACTICES (voir
tests/exemples/lisezmoi.md). Exigence du lot 1 : le validateur accepte un
exemple correct et refuse trois exemples fautifs, pour chacun des deux schemas.
"""

import sys
import unittest
from pathlib import Path

DOSSIER_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOSSIER_CODE))

from schemas import fiche_vide  # noqa: E402
from seuils import charger_seuils  # noqa: E402
from validation import valider_fiche, valider_fichier_jsonl  # noqa: E402

EXEMPLES = DOSSIER_CODE / "tests" / "exemples"

SEUILS = charger_seuils()


def codes(resultat):
    return [code for code, _ in resultat["erreurs"]]


def charger_jsonl(chemin):
    import json
    with open(chemin, "r", encoding="utf-8") as lecteur:
        return [json.loads(ligne) for ligne in lecteur if ligne.strip()]


class ValidationFichesTR(unittest.TestCase):
    """Schema des jeux T et R."""

    def test_fiche_correcte_acceptee(self):
        resultats = valider_fichier_jsonl(EXEMPLES / "tr-ok-factice.jsonl", "tr", SEUILS)
        self.assertEqual(len(resultats), 1)
        self.assertTrue(resultats[0]["valide"], msg=resultats[0]["erreurs"])
        self.assertEqual(resultats[0]["erreurs"], [])

    def test_trois_fiches_fautives_rejetees(self):
        resultats = valider_fichier_jsonl(EXEMPLES / "tr-ko-factice.jsonl", "tr", SEUILS)
        self.assertEqual(len(resultats), 3)
        for resultat in resultats:
            self.assertFalse(resultat["valide"], msg=f"ligne {resultat['ligne']} acceptee a tort")
        self.assertIn("champ_manquant", codes(resultats[0]))
        self.assertIn("enum_invalide", codes(resultats[1]))
        self.assertIn("coherence_debut_fin", codes(resultats[2]))

    def test_champ_vide_refuse(self):
        fiche = charger_jsonl(EXEMPLES / "tr-ok-factice.jsonl")[0]
        fiche["ar"] = "   "
        erreurs, _ = valider_fiche(fiche, "tr", SEUILS)
        self.assertIn("champ_vide", [c for c, _ in erreurs])

    def test_champ_null_refuse(self):
        fiche = charger_jsonl(EXEMPLES / "tr-ok-factice.jsonl")[0]
        fiche["annotateurs"] = None
        erreurs, _ = valider_fiche(fiche, "tr", SEUILS)
        self.assertIn("champ_manquant", [c for c, _ in erreurs])

    def test_date_avant_le_premier_octobre_refusee(self):
        fiche = charger_jsonl(EXEMPLES / "tr-ok-factice.jsonl")[0]
        fiche["date"] = "2026-09-30"
        erreurs, _ = valider_fiche(fiche, "tr", SEUILS)
        self.assertIn("date_avant_minimum", [c for c, _ in erreurs])

    def test_date_le_premier_octobre_refusee(self):
        # "après le 1er octobre" : le 1er octobre lui-meme est refuse.
        fiche = charger_jsonl(EXEMPLES / "tr-ok-factice.jsonl")[0]
        fiche["date"] = "2026-10-01"
        erreurs, _ = valider_fiche(fiche, "tr", SEUILS)
        self.assertIn("date_avant_minimum", [c for c, _ in erreurs])

    def test_date_mal_formatee_refusee(self):
        fiche = charger_jsonl(EXEMPLES / "tr-ok-factice.jsonl")[0]
        fiche["date"] = "05/10/2026"
        erreurs, _ = valider_fiche(fiche, "tr", SEUILS)
        self.assertIn("format_date", [c for c, _ in erreurs])

    def test_date_impossible_refusee(self):
        fiche = charger_jsonl(EXEMPLES / "tr-ok-factice.jsonl")[0]
        fiche["date"] = "2026-13-40"
        erreurs, _ = valider_fiche(fiche, "tr", SEUILS)
        self.assertIn("date_invalide", [c for c, _ in erreurs])

    def test_url_invalide_refusee(self):
        fiche = charger_jsonl(EXEMPLES / "tr-ok-factice.jsonl")[0]
        fiche["url"] = "ftp://exemple.test/vid"
        erreurs, _ = valider_fiche(fiche, "tr", SEUILS)
        self.assertIn("url_invalide", [c for c, _ in erreurs])

    def test_id_mauvais_format_refuse(self):
        fiche = charger_jsonl(EXEMPLES / "tr-ok-factice.jsonl")[0]
        fiche["id"] = "x-01"
        erreurs, _ = valider_fiche(fiche, "tr", SEUILS)
        self.assertIn("format_invalide", [c for c, _ in erreurs])

    def test_id_incoherent_avec_jeu_refuse(self):
        fiche = charger_jsonl(EXEMPLES / "tr-ok-factice.jsonl")[0]
        fiche["jeu"] = "R"
        erreurs, _ = valider_fiche(fiche, "tr", SEUILS)
        self.assertIn("id_jeu_incoherent", [c for c, _ in erreurs])

    def test_locuteurs_zero_refuse(self):
        fiche = charger_jsonl(EXEMPLES / "tr-ok-factice.jsonl")[0]
        fiche["locuteurs"] = 0
        erreurs, _ = valider_fiche(fiche, "tr", SEUILS)
        self.assertIn("valeur_invalide", [c for c, _ in erreurs])

    def test_locuteurs_texte_refuse(self):
        fiche = charger_jsonl(EXEMPLES / "tr-ok-factice.jsonl")[0]
        fiche["locuteurs"] = "deux"
        erreurs, _ = valider_fiche(fiche, "tr", SEUILS)
        self.assertIn("type_invalide", [c for c, _ in erreurs])

    def test_duree_hors_fenetre_est_un_avertissement(self):
        # La fenetre 20-30 s est une regle du lot 4 : ici, simple avertissement.
        fiche = charger_jsonl(EXEMPLES / "tr-ok-factice.jsonl")[0]
        fiche["fin"] = fiche["debut"] + 45
        erreurs, avertissements = valider_fiche(fiche, "tr", SEUILS)
        self.assertEqual(erreurs, [])
        self.assertTrue(any(c == "duree_hors_fenetre" for c, _ in avertissements))

    def test_champ_inconnu_est_un_avertissement(self):
        fiche = charger_jsonl(EXEMPLES / "tr-ok-factice.jsonl")[0]
        fiche["commentaire"] = "note libre"
        erreurs, avertissements = valider_fiche(fiche, "tr", SEUILS)
        self.assertEqual(erreurs, [])
        self.assertTrue(any(c == "champ_inconnu" for c, _ in avertissements))

    def test_fr2_optionnel_accepte_si_present(self):
        fiche = charger_jsonl(EXEMPLES / "tr-ok-factice.jsonl")[0]
        fiche["fr_2"] = "Deuxieme traduction factice, independante de la premiere."
        erreurs, _ = valider_fiche(fiche, "tr", SEUILS)
        self.assertEqual(erreurs, [])


class ValidationFichesD(unittest.TestCase):
    """Schema du jeu distille."""

    def test_fiche_correcte_acceptee(self):
        resultats = valider_fichier_jsonl(EXEMPLES / "d-ok-factice.jsonl", "d", SEUILS)
        self.assertEqual(len(resultats), 1)
        self.assertTrue(resultats[0]["valide"], msg=resultats[0]["erreurs"])
        self.assertEqual(resultats[0]["erreurs"], [])

    def test_trois_fiches_fautives_rejetees(self):
        resultats = valider_fichier_jsonl(EXEMPLES / "d-ko-factice.jsonl", "d", SEUILS)
        self.assertEqual(len(resultats), 3)
        for resultat in resultats:
            self.assertFalse(resultat["valide"], msg=f"ligne {resultat['ligne']} acceptee a tort")
        self.assertIn("champ_manquant", codes(resultats[0]))
        self.assertIn("enum_invalide", codes(resultats[1]))
        self.assertIn("type_invalide", codes(resultats[2]))

    def test_id_mauvais_format_refuse(self):
        fiche = charger_jsonl(EXEMPLES / "d-ok-factice.jsonl")[0]
        fiche["id"] = "d-12"
        erreurs, _ = valider_fiche(fiche, "d", SEUILS)
        self.assertIn("format_invalide", [c for c, _ in erreurs])

    def test_ecart_ecoute_negatif_refuse(self):
        fiche = charger_jsonl(EXEMPLES / "d-ok-factice.jsonl")[0]
        fiche["ecart_ecoute"] = -0.1
        erreurs, _ = valider_fiche(fiche, "d", SEUILS)
        self.assertIn("valeur_invalide", [c for c, _ in erreurs])

    def test_ecart_ecoute_au_dessus_du_seuil_est_un_avertissement(self):
        # Le filtre D1 aurait du ecarter ce segment ; la fiche finale ne doit
        # pas l'avoir. Ici : avertissement, la decision appartient a D1.
        fiche = charger_jsonl(EXEMPLES / "d-ok-factice.jsonl")[0]
        fiche["ecart_ecoute"] = 0.4
        erreurs, avertissements = valider_fiche(fiche, "d", SEUILS)
        self.assertEqual(erreurs, [])
        self.assertTrue(any(c == "ecart_ecoute_haut" for c, _ in avertissements))

    def test_audio_pas_flac_est_un_avertissement(self):
        fiche = charger_jsonl(EXEMPLES / "d-ok-factice.jsonl")[0]
        fiche["audio"] = "audio/factice-episode-001.wav"
        erreurs, avertissements = valider_fiche(fiche, "d", SEUILS)
        self.assertEqual(erreurs, [])
        self.assertTrue(any(c == "audio_pas_flac" for c, _ in avertissements))

    def test_signalements_pas_une_liste_refuse(self):
        fiche = charger_jsonl(EXEMPLES / "d-ok-factice.jsonl")[0]
        fiche["signalements"] = "chiffres"
        erreurs, _ = valider_fiche(fiche, "d", SEUILS)
        self.assertIn("liste_invalide", [c for c, _ in erreurs])

    def test_score_qe_negatif_accepte(self):
        # COMETKiwi peut donner des notes negatives.
        fiche = charger_jsonl(EXEMPLES / "d-ok-factice.jsonl")[0]
        fiche["score_qe"] = -0.3
        erreurs, _ = valider_fiche(fiche, "d", SEUILS)
        self.assertEqual(erreurs, [])

    def test_nan_dans_champ_nombre_refuse_defaut_2(self):
        # Defaut 2 du lot 1 : NaN refuse dans les champs numeriques
        fiche = charger_jsonl(EXEMPLES / "d-ok-factice.jsonl")[0]
        fiche["debut"] = float("nan")
        erreurs, _ = valider_fiche(fiche, "d", SEUILS)
        self.assertIn("valeur_invalide", [c for c, _ in erreurs])

    def test_infini_dans_champ_nombre_refuse_defaut_2(self):
        # Defaut 2 du lot 1 : infini refuse dans les champs numeriques
        fiche = charger_jsonl(EXEMPLES / "d-ok-factice.jsonl")[0]
        fiche["fin"] = float("inf")
        erreurs, _ = valider_fiche(fiche, "d", SEUILS)
        self.assertIn("valeur_invalide", [c for c, _ in erreurs])

    def test_duree_hors_fenetre_est_un_avertissement_defaut_4(self):
        # Defaut 4 : avertissement si la duree sort de 5 a 25 s
        fiche = charger_jsonl(EXEMPLES / "d-ok-factice.jsonl")[0]
        fiche["fin"] = fiche["debut"] + 30  # duree = 30 s > 25 s
        erreurs, avertissements = valider_fiche(fiche, "d", SEUILS)
        self.assertEqual(erreurs, [])
        self.assertTrue(any(c == "duree_hors_fenetre" for c, _ in avertissements))

    def test_ecart_ecoute_superieur_a_cinq_refuse_defaut_4(self):
        # Defaut 4 : refuse une valeur ecart_ecoute > 5
        fiche = charger_jsonl(EXEMPLES / "d-ok-factice.jsonl")[0]
        fiche["ecart_ecoute"] = 5.1
        erreurs, _ = valider_fiche(fiche, "d", SEUILS)
        self.assertIn("valeur_invalide", [c for c, _ in erreurs])


class ValidationFichesA(unittest.TestCase):
    """Schema du jeu d'alignement (Jeu A)."""

    def test_fiche_correcte_acceptee(self):
        resultats = valider_fichier_jsonl(EXEMPLES / "a-ok-factice.jsonl", "a", SEUILS)
        self.assertEqual(len(resultats), 1)
        self.assertTrue(resultats[0]["valide"], msg=resultats[0]["erreurs"])
        self.assertEqual(resultats[0]["erreurs"], [])

    def test_trois_fiches_fautives_rejetees(self):
        resultats = valider_fichier_jsonl(EXEMPLES / "a-ko-factice.jsonl", "a", SEUILS)
        self.assertEqual(len(resultats), 3)
        for resultat in resultats:
            self.assertFalse(resultat["valide"], msg=f"ligne {resultat['ligne']} acceptee a tort")
        self.assertIn("champ_manquant", codes(resultats[0]))
        self.assertIn("enum_invalide", codes(resultats[1]))
        self.assertIn("type_invalide", codes(resultats[2]))


class ValidationFichier(unittest.TestCase):
    """Lecture JSONL, lignes illisibles, doublons."""

    def test_ligne_json_invalide_rejetee(self):
        import json
        fiche = charger_jsonl(EXEMPLES / "d-ok-factice.jsonl")[0]
        fiche["id"] = "d-000009"
        texte = json.dumps(fiche, ensure_ascii=False) + "\nligne cassee\n"
        chemin = DOSSIER_CODE / "tests" / "_tmp_ligne_cassee.jsonl"
        chemin.write_text(texte, encoding="utf-8")
        try:
            resultats = valider_fichier_jsonl(chemin, "d", SEUILS)
        finally:
            chemin.unlink()
        self.assertFalse(resultats[1]["valide"])
        self.assertIn("json_invalide", codes(resultats[1]))

    def test_id_doublon_refuse_defaut_3(self):
        # Defaut 3 : un identifiant en double devient une erreur
        import json
        fiche = charger_jsonl(EXEMPLES / "d-ok-factice.jsonl")[0]
        chemin = DOSSIER_CODE / "tests" / "_tmp_doublon.jsonl"
        chemin.write_text(
            json.dumps(fiche, ensure_ascii=False) + "\n" + json.dumps(fiche, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        try:
            resultats = valider_fichier_jsonl(chemin, "d", SEUILS)
        finally:
            chemin.unlink()
        self.assertTrue(resultats[0]["valide"])
        self.assertFalse(resultats[1]["valide"])
        self.assertIn("id_doublon", codes(resultats[1]))

    def test_fiche_pas_un_dict_rejetee(self):
        erreurs, _ = valider_fiche(["pas", "une", "fiche"], "tr", SEUILS)
        self.assertEqual([c for c, _ in erreurs], ["fiche_invalide"])


class ValidationFichesP(unittest.TestCase):
    """Validation des fiches du jeu P (presse OPUS)."""

    def test_fiche_p_valide(self):
        fiche = {
            "id": "p-000001",
            "source": "News-Commentary",
            "id_origine": 42,
            "ar": "قال الوزير إن المفاوضات مستمرة حتى نهاية الشهر",
            "fr": "Le ministre a affirmé que les négociations se poursuivent jusqu'à la fin du mois.",
        }
        erreurs, avertissements = valider_fiche(fiche, "p", SEUILS)
        self.assertEqual(erreurs, [])
        self.assertEqual(avertissements, [])

    def test_fiche_p_champ_manquant(self):
        fiche = {
            "id": "p-000001",
            "source": "News-Commentary",
            "id_origine": 42,
            "ar": "قال الوزير إن المفاوضات مستمرة",
            # "fr" manquant
        }
        erreurs, _ = valider_fiche(fiche, "p", SEUILS)
        self.assertIn("champ_manquant", codes({"erreurs": erreurs}))

    def test_fiche_p_format_id_invalide(self):
        fiche = {
            "id": "mauvais-id",
            "source": "News-Commentary",
            "id_origine": 42,
            "ar": "نص عربي",
            "fr": "Texte français.",
        }
        erreurs, _ = valider_fiche(fiche, "p", SEUILS)
        self.assertIn("format_invalide", codes({"erreurs": erreurs}))


class ModeleFicheVide(unittest.TestCase):
    """Le modele de fiche vide doit etre incomplet donc refuse tel quel."""

    def test_fiche_vide_refusee_telle_quelle(self):
        for type_jeu in ("tr", "d", "a", "p"):
            with self.subTest(type_jeu=type_jeu):
                erreurs, _ = valider_fiche(fiche_vide(type_jeu), type_jeu, SEUILS)
                self.assertTrue(erreurs, f"une fiche vide ({type_jeu}) doit etre refusee")
                self.assertTrue(all(c == "champ_manquant" for c, _ in erreurs))


if __name__ == "__main__":
    unittest.main()


