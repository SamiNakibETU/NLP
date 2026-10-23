"""Rapport de session avec des faux : diarisation, LLM et synthèse vocale simulés (aucun modèle, aucun GPU).

    python -m unittest speech_app/tests/test_session_report.py     (ou pytest)
"""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import session_report as sr
import tts
from sessions import Session

SEGMENTS = [
    (0.0, 4.0, "هارولد ويلسون يتحدث اليوم", "Harold Wilsone parle aujourd'hui de l'économie."),
    (4.0, 8.0, "قال إن النمو بلغ 3 في المئة", "Il a dit que la croissance a atteint 3 pour cent."),
    (8.0, 12.0, "ردت المعارضة", "L'opposition a répondu."),
    (12.0, 16.0, "وطالبت بالإصلاح", "Elle a réclamé des réformes."),
    (16.0, 20.0, "ثم عاد الوزير", "Puis le ministre est revenu au micro."),
    (20.0, 24.0, "وانتهت الجلسة", "La séance s'est terminée tard."),
]
LOCUTEURS_FAUX = ["A", "A", "B", "B", "A", "B"]


def faux_diariseur(session, segments, options):
    return LOCUTEURS_FAUX[:len(segments)], "faux diariseur (test)"


class FauxLLM:
    """Relecture : applique le glossaire reçu dans le prompt ; résumé : puces datées tirées du transcript."""
    nom = "faux-llm"

    def __init__(self):
        self.prompts = []

    def __call__(self, messages, max_new_tokens=256):
        self.prompts.append(messages[-1]["content"])
        user = messages[-1]["content"]
        if messages[0]["content"] == sr.SYSTEME_REVISION:
            fr = user.split("Traduction à relire : ", 1)[1]
            for nom in re.findall(r"^- (.+?) \(", user, flags=re.M):
                fr = fr.replace(nom + "e", nom)           # « Harold Wilsone » -> « Harold Wilson »
            return fr
        stamps = re.findall(r"^\[(\d\d:\d\d)\]", user, flags=re.M)
        return ("## Sujet\nUn débat sur l'économie.\n\n## Points clés\n"
                + "\n".join(f"- point {i} [{t}]" for i, t in enumerate(stamps[:4]))
                + "\n\n## Personnes, organisations, lieux cités\n- Harold Wilson\n\n## Chiffres\n- 3 pour cent [00:04]")


def creer_session(racine: str) -> Session:
    os.environ["PORTEPAROLE_SESSIONS"] = racine
    s = Session.creer("test", "faux", "Journal de test")
    s.ecrire_audio(np.zeros(24 * 16000, dtype=np.float32))
    for i, (d, f, ar, fr) in enumerate(SEGMENTS):
        s.ajouter_segment({"id": i, "debut_s": d, "fin_s": f, "ar": ar, "fr": fr})
    return s


class TestRapport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.session = creer_session(self._tmp.name)
        self.lexique = sr.charger_lexique()

    def rapport(self, **opt):
        return sr.generer_rapport(self.session, {"diariseur": faux_diariseur, "lexique": self.lexique, **opt})

    def test_rapport_complet_avec_faux_llm(self):
        llm = FauxLLM()
        r = self.rapport(llm=llm, inclure_arabe=True)
        d = self.session.dossier_rapport
        for nom in ("transcript.md", "resume.md", "sous_titres.srt", "rapport.json"):
            self.assertTrue((d / nom).is_file(), nom)
        self.assertEqual(r["n_locuteurs"], 2)
        self.assertEqual(r["methode_diarisation"], "faux diariseur (test)")
        self.assertEqual([l["locuteur"] for l in r["segments"]], [1, 1, 2, 2, 1, 2])
        transcript = (d / "transcript.md").read_text(encoding="utf-8")
        self.assertIn("**[00:08] Locuteur 2** : L'opposition a répondu.", transcript)
        self.assertIn("> هارولد ويلسون يتحدث اليوم", transcript)            # arabe joint sur demande
        self.assertEqual(r["methode_resume"], "LLM")
        self.assertEqual(r["controle_horodatages"]["invalides"], [])
        self.assertIn("[00:04]", (d / "resume.md").read_text(encoding="utf-8"))
        self.assertEqual(json.loads((d / "rapport.json").read_text(encoding="utf-8"))["sid"], self.session.sid)

    def test_lexique_applique_et_original_conserve(self):
        llm = FauxLLM()
        r = self.rapport(llm=llm)
        premier = r["segments"][0]
        self.assertEqual(premier["fr_original"], "Harold Wilsone parle aujourd'hui de l'économie.")
        self.assertEqual(premier["fr_revise"], "Harold Wilson parle aujourd'hui de l'économie.")
        self.assertTrue(premier["modifie"])
        self.assertIn("Harold Wilson", r["lexique"]["entites_detectees"])
        self.assertTrue(any("- Harold Wilson (personne)" in p for p in llm.prompts))   # glossaire transmis au LLM
        self.assertFalse(r["segments"][2]["modifie"])

    def test_sous_titres_srt_valides(self):
        self.rapport(llm=FauxLLM())
        srt = (self.session.dossier_rapport / "sous_titres.srt").read_text(encoding="utf-8")
        blocs = [b for b in srt.strip().split("\n\n") if b.strip()]
        self.assertGreaterEqual(len(blocs), 6)
        fin_prec = -1.0
        for k, b in enumerate(blocs, 1):
            lignes = b.split("\n")
            self.assertEqual(lignes[0], str(k))
            m = re.fullmatch(r"(\d\d):(\d\d):(\d\d),(\d{3}) --> (\d\d):(\d\d):(\d\d),(\d{3})", lignes[1])
            self.assertIsNotNone(m, lignes[1])
            g = list(map(int, m.groups()))
            deb, fin = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000, g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
            self.assertLess(deb, fin)
            self.assertGreaterEqual(deb, fin_prec - 1e-6)
            fin_prec = fin
            self.assertTrue("".join(lignes[2:]).strip())
        self.assertIn("[Locuteur 1] Harold Wilson", srt)

    def test_repli_extractif_sans_llm(self):
        with mock.patch.dict(os.environ, {"PORTEPAROLE_LLM_RAPPORT": "off"}):
            r = self.rapport()
        self.assertTrue(r["methode_resume"].startswith("extractif"))
        self.assertEqual(r["methode_revision"], "aucune (désactivée)")
        resume = (self.session.dossier_rapport / "resume.md").read_text(encoding="utf-8")
        self.assertIn("Résumé extractif de secours", resume)
        self.assertIn("## Points clés", resume)
        self.assertRegex(resume, r"\[\d\d:\d\d\]")
        self.assertIn("3 pour cent", resume)                                    # chiffre repéré
        self.assertIn("Harold Wilson", resume)                                  # entité du lexique repérée dans l'arabe
        self.assertEqual(r["segments"][0]["fr_revise"], r["segments"][0]["fr_original"])

    def test_llm_en_panne_ne_casse_pas_le_rapport(self):
        def llm_casse(messages, max_new_tokens=0):
            raise RuntimeError("OOM")
        r = self.rapport(llm=llm_casse)
        self.assertTrue(r["methode_resume"].startswith("extractif"))
        self.assertTrue(all(not s["modifie"] for s in r["segments"]))

    def test_session_vide_refusee(self):
        vide = Session.creer("test", "faux")
        with self.assertRaises(ValueError):
            sr.generer_rapport(vide, {})


class TestEtapes(unittest.TestCase):
    def test_regroupement_deux_voix(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=16), rng.normal(size=16)
        emb = np.stack([a + 0.05 * rng.normal(size=16) for _ in range(3)] + [b + 0.05 * rng.normal(size=16) for _ in range(3)])
        self.assertEqual(sr.regrouper_embeddings(emb, 0.4), [0, 0, 0, 1, 1, 1])
        self.assertEqual(len(set(sr.regrouper_embeddings(emb, 0.4, nb_locuteurs=1))), 1)
        self.assertEqual(sr.regrouper_embeddings(np.zeros((0, 4))), [])

    def test_attribution_majoritaire(self):
        segs = [{"debut_s": 0, "fin_s": 10}, {"debut_s": 10, "fin_s": 12}, {"debut_s": 30, "fin_s": 32}]
        tours = [(0, 4, "x"), (4, 10.5, "y"), (10.5, 13, "x")]
        self.assertEqual(sr.attribuer_locuteurs(segs, tours), ["y", "x", "x"])        # le dernier : tour le plus proche

    def test_numerotation_par_ordre_d_apparition(self):
        self.assertEqual(sr._numeroter(["z", "k", "z", None]), [1, 2, 1, None])

    def test_lexique_charge_et_detecte(self):
        lex = sr.charger_lexique()
        self.assertGreater(len(lex), 300)
        self.assertEqual([e["fr"] for e in sr.entites_du_texte("قال هارولد ويلسون", lex)][:1], ["Harold Wilson"])
        self.assertEqual(sr.entites_du_texte("جملة بلا أسماء", lex), [])

    def test_garde_fou_contre_l_invention(self):
        o = "Le Premier ministre a annoncé 3 mesures."
        self.assertTrue(sr.sortie_acceptable(o, "Le Premier ministre a annoncé 3 mesures nouvelles."))
        self.assertFalse(sr.sortie_acceptable(o, o + " Il a aussi promis de démissionner avant la fin de l'année."))
        self.assertFalse(sr.sortie_acceptable(o, "Le Premier ministre a annoncé 4 mesures."))
        self.assertFalse(sr.sortie_acceptable(o, "رئيس الوزراء أعلن ثلاثة إجراءات."))
        self.assertFalse(sr.sortie_acceptable(o, ""))


class TestTTS(unittest.TestCase):
    def test_cache_par_hash_et_repli(self):
        appels = []

        def faux_piper(texte, sortie):
            appels.append("piper")
            raise tts.TTSIndisponible("piper absent")

        def faux_edge(texte, sortie):
            appels.append("edge")
            Path(sortie).write_bytes(b"ID3fake")

        moteurs = {"piper": (faux_piper, ".wav", "audio/wav"), "edge": (faux_edge, ".mp3", "audio/mpeg")}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(tts, "RACINE", Path(d)), \
                mock.patch.dict(os.environ, {"PORTEPAROLE_TTS": "piper"}):
            chemin, mime = tts.synthese("**Bonjour** [00:12] le monde", moteurs)
            self.assertEqual((chemin.suffix, mime, appels), (".mp3", "audio/mpeg", ["piper", "edge"]))
            chemin2, _ = tts.synthese("Bonjour le monde", moteurs)                     # même texte nettoyé -> cache
            self.assertEqual(chemin, chemin2)
            self.assertEqual(len(appels), 2)
            with mock.patch.dict(os.environ, {"PORTEPAROLE_TTS": "off"}), self.assertRaises(tts.TTSIndisponible):
                tts.synthese("Bonjour", moteurs)
            with self.assertRaises(ValueError):
                tts.synthese("[00:01]", moteurs)


class TestAPI(unittest.TestCase):
    def test_routes(self):
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
        except Exception as exc:                          # noqa: BLE001
            self.skipTest(f"fastapi/httpx indisponible : {exc}")
        import session_api
        with tempfile.TemporaryDirectory() as d:
            session = creer_session(d)
            app = FastAPI()
            app.include_router(session_api.router)
            c = TestClient(app)
            self.assertEqual([s["sid"] for s in c.get("/api/sessions").json()["sessions"]], [session.sid])
            self.assertEqual(c.get(f"/api/sessions/{session.sid}/rapport").json()["etat"], "absent")
            self.assertEqual(c.post("/api/sessions/inconnue/rapport").status_code, 404)
            self.assertEqual(c.post(f"/api/sessions/{session.sid}/rapport", json={"x": 1}).status_code, 422)
            with mock.patch.dict(os.environ, {"PORTEPAROLE_LLM_RAPPORT": "off"}):
                session_api.lancer_rapport(session.sid, {"diarisation": "off", "docx": False}, synchrone=True)
            etat = c.get(f"/api/sessions/{session.sid}/rapport").json()
            self.assertEqual(etat["etat"], "pret")
            self.assertEqual(c.get(f"/api/sessions/{session.sid}/fichiers/resume.md").status_code, 200)
            self.assertEqual(c.get(f"/api/sessions/{session.sid}/fichiers/..%2Fmeta.json").status_code, 404)
            with mock.patch.dict(os.environ, {"PORTEPAROLE_TTS": "off"}):
                self.assertEqual(c.get("/api/tts", params={"texte": "Bonjour"}).status_code, 503)


if __name__ == "__main__":
    unittest.main()
