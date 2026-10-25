"""Tests du benchmark de traduction, sans modèle ni réseau."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

DOSSIER = Path(__file__).resolve().parents[1]
if str(DOSSIER) not in sys.path:
    sys.path.insert(0, str(DOSSIER))

from benchmark_traduction import (  # noqa: E402
    RACINE_PROJET,
    _motif_nom_arabe,
    _racine_nom_francais,
    charger_traductions_supplementaires,
    choisir_modele,
    controler_traduction,
    controles_automatiques,
    decrire_corpus,
    ecrire_grille_aveugle,
    ecrire_jsonl,
    geler_selection,
    charger_jeu,
    lire_dossier_xml,
    lire_marque,
    lire_xml_mgb2,
    nettoyer_sortie,
    noter_grilles,
    scores_avec_reference,
    segments_marque,
    selectionner_segments,
    trouver_marque_francais,
    traduire_candidat,
)

MARQUE_EQUIPE = RACINE_PROJET / "donnees" / "marque" / "5331FFF2-358E-445E-BA71-74B745387304.marque.txt"

MARQUE_AR = """<<EP-9_utt_1 | 00:00:18-00:00:27>>
أهلا بكم، دعا الرئيس الفرنسي مواطنيه إلى الوحدة

<<EP-9_utt_2 | 00:01:00-00:01:06>>
على جميع الأطراف
في ظل الحديث عن 3 مسلمين
"""

MARQUE_FR = """<<EP-9_utt_1 | 00:00:18-00:00:27>>
Bienvenue, le président français a appelé ses concitoyens à l'unité

<<EP-9_utt_2 | 00:01:00-00:01:06>>
à toutes les parties, alors qu'on parle de 3 musulmans
"""

XML_EQUIPE = RACINE_PROJET / "donnees" / "00BA15A7-341E-4F65-BA88-EACB688D9746.xml"

XML_MINI = """<?xml version="1.0" encoding="UTF-8"?>
<transcript>
  <head><recording date="2015/06/29" filename="EP-1" title="mn wA$nTn"/></head>
  <body><segments>
    <segment AWD="2.0" PMER="0.0" WMER="0.0" starttime="0.0" endtime="6.0" id="u1" who="EP_1_speaker1_align">
      <element type="word">أهلا</element><element type="word">بكم</element><element type="word">في</element>
      <element type="word">حلقة</element><element type="word">جديدة</element><element type="word">من</element>
      <element type="word">البرنامج</element><element type="word">،</element>
    </segment>
    <segment AWD="2.0" PMER="0.0" WMER="40.0" starttime="6.0" endtime="12.0" id="u2" who="EP_1_speaker2_align">
      <element type="word">نص</element><element type="word">مقطوع</element><element type="word">جدا</element>
      <element type="word">هنا</element><element type="word">الآن</element><element type="word">اليوم</element>
    </segment>
    <segment AWD="2.0" PMER="0.0" WMER="0.0" starttime="12.0" endtime="20.0" id="u3" who="EP_1_speaker2_align">
      <element type="word">زار</element><element type="word">الرئيس</element><element type="word">مصر</element>
      <element type="word">عام</element><element type="word">٢٠١٣</element><element type="word">مع</element>
      <element type="word">وفد</element>
    </segment>
  </segments></body>
</transcript>
"""


def ecrire_xml(dossier: Path, nom: str = "EP-1.xml") -> Path:
    chemin = Path(dossier) / nom
    chemin.write_text(XML_MINI, encoding="utf-8")
    return chemin


class TestLectureXml(unittest.TestCase):
    def test_segments_texte_et_indicateurs(self):
        with tempfile.TemporaryDirectory() as dossier:
            segments = lire_xml_mgb2(ecrire_xml(Path(dossier)))
        self.assertEqual([s["id"] for s in segments], ["u1", "u2", "u3"])
        self.assertEqual(segments[0]["ar"], "أهلا بكم في حلقة جديدة من البرنامج ،")
        self.assertEqual(segments[0]["nb_mots"], 7)
        self.assertEqual(segments[1]["wmer"], 40.0)
        self.assertEqual(segments[0]["episode"], "EP-1")
        self.assertEqual(segments[2]["duree_s"], 8.0)

    def test_fichier_de_l_equipe(self):
        if not XML_EQUIPE.is_file():
            self.skipTest("XML de l'équipe absent")
        segments = lire_dossier_xml(XML_EQUIPE.parent)
        resume = decrire_corpus(segments)
        self.assertEqual(resume["segments"], 227)
        self.assertEqual(resume["episodes"], 1)
        choisis = selectionner_segments(segments, nb=60)
        self.assertEqual(len(choisis), 60)
        self.assertTrue(all(5 <= s["duree_s"] <= 25 and s["wmer"] <= 15 for s in choisis))
        self.assertGreater(len({s["locuteur"] for s in choisis}), 3)


class TestLectureMarque(unittest.TestCase):
    def _ecrire(self, dossier: Path) -> Path:
        (dossier / "EP-9.fr.txt").write_text(MARQUE_FR, encoding="utf-8")
        chemin = dossier / "EP-9.marque.txt"
        chemin.write_text(MARQUE_AR, encoding="utf-8")
        return chemin

    def test_blocs_horodatages_et_texte_sur_plusieurs_lignes(self):
        with tempfile.TemporaryDirectory() as dossier:
            blocs = lire_marque(self._ecrire(Path(dossier)))
        self.assertEqual([b["id"] for b in blocs], ["EP-9_utt_1", "EP-9_utt_2"])
        self.assertEqual((blocs[0]["debut"], blocs[0]["fin"]), (18.0, 27.0))
        self.assertEqual(blocs[1]["texte"], "على جميع الأطراف في ظل الحديث عن 3 مسلمين")

    def test_reference_francaise_trouvee_et_alignee(self):
        with tempfile.TemporaryDirectory() as dossier:
            chemin_ar = self._ecrire(Path(dossier))
            chemin_fr = trouver_marque_francais(chemin_ar)
            self.assertEqual(chemin_fr.name, "EP-9.fr.txt")
            segments = segments_marque(chemin_ar, chemin_fr)
        self.assertEqual(segments[1]["episode"], "EP-9")
        self.assertEqual(segments[1]["duree_s"], 6.0)
        self.assertEqual(segments[1]["contexte_ar"], [segments[0]["ar"]])
        self.assertTrue(segments[0]["fr_reference"].startswith("Bienvenue"))

    def test_scores_contre_la_reference(self):
        with tempfile.TemporaryDirectory() as dossier:
            chemin_ar = self._ecrire(Path(dossier))
            segments = segments_marque(chemin_ar, trouver_marque_francais(chemin_ar))
        exact = [{"id": s["id"], "fr": s["fr_reference"]} for s in segments]
        faux = [{"id": s["id"], "fr": "Le présentateur parle."} for s in segments]
        scores = scores_avec_reference(segments, {"exact": exact, "faux": faux}, avec_comet=False)
        self.assertEqual(scores["exact"]["nb"], 2)
        self.assertGreater(scores["exact"]["chrf_pp"], scores["faux"]["chrf_pp"])

    def test_fichier_de_l_equipe(self):
        if not MARQUE_EQUIPE.is_file():
            self.skipTest("fichier marqué de l'équipe absent")
        config = {
            "donnees": {"marque_ar": str(MARQUE_EQUIPE.relative_to(RACINE_PROJET))},
            "selection": {"nb_phrases_contexte": 2},
        }
        segments, _ = charger_jeu(config, Path(tempfile.gettempdir()))
        self.assertEqual(len(segments), 217)
        self.assertEqual({s["episode"] for s in segments}, {"5331FFF2-358E-445E-BA71-74B745387304"})
        self.assertTrue(all(s["ar"] and s["fin"] > s["debut"] for s in segments))


class TestSelection(unittest.TestCase):
    def test_filtre_wmer_et_contexte(self):
        with tempfile.TemporaryDirectory() as dossier:
            segments = lire_xml_mgb2(ecrire_xml(Path(dossier)))
        choisis = selectionner_segments(segments, nb=10)
        self.assertEqual([s["id"] for s in choisis], ["u1", "u3"])
        self.assertEqual(len(choisis[1]["contexte_ar"]), 2)
        self.assertTrue(choisis[1]["contexte_ar"][1].startswith("نص"))

    def test_selection_gelee(self):
        with tempfile.TemporaryDirectory() as dossier:
            chemin = Path(dossier) / "segments.jsonl"
            premier = geler_selection([{"id": "a", "ar": "x"}], chemin)
            second = geler_selection([{"id": "b", "ar": "y"}], chemin)
        self.assertEqual(premier, second)


class TestControles(unittest.TestCase):
    SEGMENT = {"id": "u3", "ar": "زار الرئيس مصر عام ٢٠١٣ مع وفد", "duree_s": 8.0}

    def test_traduction_correcte(self):
        lexique = [([_motif_nom_arabe("مصر")], _racine_nom_francais("Égypte"))]
        controle = controler_traduction(self.SEGMENT, "Le président a visité l'Égypte en 2013 avec une délégation.", 0.6, 2.5, lexique)
        self.assertEqual(controle["signaux"], [])
        self.assertFalse(controle["echec_dur"])

    def test_echecs_durs_et_signaux(self):
        lexique = [([_motif_nom_arabe("مصر")], _racine_nom_francais("Égypte"))]
        self.assertTrue(controler_traduction(self.SEGMENT, "", 0.6, 2.5)["echec_dur"])
        reste = controler_traduction(self.SEGMENT, "Le président زار a visité le pays en 2013.", 0.6, 2.5)
        self.assertIn("arabe_restant", reste["signaux"])
        meta = controler_traduction(self.SEGMENT, "Voici la traduction : le président est venu en 2013.", 0.6, 2.5)
        self.assertIn("meta_texte", meta["signaux"])
        chiffre = controler_traduction(self.SEGMENT, "Le président a visité l'Égypte avec une délégation.", 0.6, 2.5, lexique)
        self.assertIn("chiffre_manquant", chiffre["signaux"])
        nom = controler_traduction(self.SEGMENT, "Le président a visité le pays en 2013 avec une délégation.", 0.6, 2.5, lexique)
        self.assertIn("nom_propre_different", nom["signaux"])

    def test_nom_dans_un_mot_plus_long_ne_compte_pas(self):
        motif = _motif_nom_arabe("مصر")
        self.assertIsNone(motif.search("الشعب المصري"))
        self.assertIsNotNone(motif.search("وصل إلى ومصر"))

    def test_nettoyage(self):
        self.assertEqual(nettoyer_sortie("Traduction : « Bonjour à tous. »\n\nNote : rien"), "Bonjour à tous.")


class TestChaineComplete(unittest.TestCase):
    def setUp(self):
        self.segments = [
            {"id": f"s{i}", "ar": "زار الرئيس البلد مع وفد كبير", "duree_s": 6.0, "contexte_ar": []}
            for i in range(6)
        ]
        self.traductions = {
            "bon": [{"id": s["id"], "fr": "Le président a visité le pays avec une grande délégation.", "temps_s": 1.0} for s in self.segments],
            "moyen": [{"id": s["id"], "fr": "Le président est allé dans le pays avec un groupe.", "temps_s": 0.2} for s in self.segments],
            "casse": [{"id": s["id"], "fr": "", "temps_s": 0.1} for s in self.segments],
        }

    def test_reprise_sans_recharger_le_modele(self):
        class Faux:
            appels = 0

            def traduire(self, segments):
                Faux.appels += 1
                return [("Bonjour.", 0.5) for _ in segments]

            def fermer(self):
                pass

        with tempfile.TemporaryDirectory() as dossier:
            candidat = {"nom": "faux", "modele_id": "x"}
            traduire_candidat(candidat, self.segments, Path(dossier), lambda c: Faux())
            lignes = traduire_candidat(candidat, self.segments, Path(dossier), lambda c: Faux())
        self.assertEqual(Faux.appels, 1)
        self.assertEqual(len(lignes), 6)

    def test_grille_aveugle_notation_et_decision(self):
        with tempfile.TemporaryDirectory() as dossier:
            dossier = Path(dossier)
            grille, cle_chemin = ecrire_grille_aveugle(self.segments, self.traductions, dossier)
            cle = json.loads(cle_chemin.read_text(encoding="utf-8"))
            self.assertEqual(len(cle), 6)
            self.assertGreater(len({tuple(v.values()) for v in cle.values()}), 1)

            notes = {"bon": "5", "moyen": "3", "casse": "1"}
            for annotateur in ("fa", "ha"):
                with open(grille, encoding="utf-8-sig", newline="") as fichier:
                    lignes = list(csv.DictReader(fichier))
                for ligne in lignes:
                    for lettre, nom in cle[ligne["id"]].items():
                        ligne[f"sens_{lettre}"] = notes[nom]
                        ligne[f"fluidite_{lettre}"] = notes[nom]
                        ligne[f"noms_chiffres_{lettre}"] = "ok"
                        if nom == "bon":
                            ligne["prefere"] = lettre
                    ligne["fr_reference"] = "Le président a visité le pays avec une grande délégation."
                with open(dossier / f"grille_{annotateur}.csv", "w", encoding="utf-8-sig", newline="") as fichier:
                    ecrivain = csv.DictWriter(fichier, fieldnames=list(lignes[0].keys()))
                    ecrivain.writeheader()
                    ecrivain.writerows(lignes)

            humain = noter_grilles(
                [dossier / "grille_fa.csv", dossier / "grille_ha.csv"], cle_chemin,
                self.segments, self.traductions, nb_tirages=200, avec_comet=False,
            )
        self.assertEqual(humain["humain"]["bon"]["sens_moyen"], 5.0)
        self.assertEqual(humain["humain"]["bon"]["prefere_pct"], 100.0)
        self.assertEqual(humain["accord_preference_pct"], 100.0)
        self.assertTrue(humain["humain"]["moyen"]["ecart_sens_vs_meilleur"]["significatif"])
        self.assertEqual(humain["avec_reference"]["bon"]["chrf_pp"], 100.0)

        automatique = controles_automatiques(self.segments, self.traductions, lexique=[])
        self.assertEqual(automatique["casse"]["echec_dur_pct"], 100.0)
        decision = choisir_modele(automatique, humain, nb_segments_cible=3600)
        self.assertEqual(decision["choix"], "bon")
        self.assertFalse(decision["provisoire"])
        self.assertIn("casse", decision["ecartes"])
        self.assertEqual(decision["heures_projetees_corpus"], 1.0)

    def test_grille_notee_protege_la_cle(self):
        with tempfile.TemporaryDirectory() as dossier:
            dossier = Path(dossier)
            grille, cle_chemin = ecrire_grille_aveugle(self.segments, self.traductions, dossier)
            (dossier / "grille_fa.csv").write_text(grille.read_text(encoding="utf-8-sig"), encoding="utf-8")
            cle_avant = cle_chemin.read_text(encoding="utf-8")
            ecrire_grille_aveugle(self.segments, self.traductions, dossier, graine=7)
            self.assertEqual(cle_chemin.read_text(encoding="utf-8"), cle_avant)
            avec_seamless = {**self.traductions, "seamless": self.traductions["bon"]}
            with self.assertRaises(RuntimeError):
                ecrire_grille_aveugle(self.segments, avec_seamless, dossier)

    def test_traductions_d_un_autre_notebook(self):
        with tempfile.TemporaryDirectory() as dossier:
            dossier = Path(dossier)
            ecrire_jsonl(dossier / "traductions_seamless-m4t-v2.jsonl", self.traductions["bon"])
            ecrire_jsonl(dossier / "traductions_bon.jsonl", self.traductions["bon"])
            ecrire_jsonl(dossier / "traductions_ancien.jsonl", self.traductions["bon"][:3])
            trouves = charger_traductions_supplementaires(dossier, self.segments, ["bon"])
        self.assertEqual(list(trouves), ["seamless-m4t-v2"])

    def test_sans_notes_le_choix_est_provisoire(self):
        automatique = controles_automatiques(self.segments, self.traductions, lexique=[])
        decision = choisir_modele(automatique, None, {"bon": 0.71, "moyen": 0.80})
        self.assertTrue(decision["provisoire"])
        self.assertEqual(decision["choix"], "moyen")


if __name__ == "__main__":
    unittest.main()
