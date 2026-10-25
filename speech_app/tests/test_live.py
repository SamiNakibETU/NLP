"""Direct : session complète rejouée avec le moteur factice (ASR et traducteurs simulés), sans GPU ni modèle.

Vérifie par WebSocket et par fichier : gris puis blanc, pas de doublon, segments persistés, spectateurs, bilan.
"""
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.environ["PORTEPAROLE_LIVE_DELAI"] = "0"
_TMP = tempfile.mkdtemp(prefix="pp_sessions_")
os.environ["PORTEPAROLE_SESSIONS"] = _TMP

from starlette.testclient import TestClient  # noqa: E402

import live  # noqa: E402
import live_moteurs  # noqa: E402
from sessions import Session  # noqa: E402

SR = 16000


def pcm16(signal: np.ndarray) -> bytes:
    return (np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes()


def ecrire_wav(chemin: Path, signal: np.ndarray) -> None:
    with wave.open(str(chemin), "wb") as f:
        f.setnchannels(1), f.setsampwidth(2), f.setframerate(SR)
        f.writeframes(pcm16(signal))


class TestLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        live.configurer(factice=True)
        live.TEMPS_REEL = False
        cls.client = TestClient(live.creer_app())
        cls.signal = live_moteurs.signal_demo([0, 1, 2], duree_phrase_s=6.0)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(_TMP, ignore_errors=True)

    def rejouer_ws(self):
        """Envoie l'audio par paquets de 250 ms comme le navigateur ; renvoie (sid, tous les messages reçus)."""
        messages = []
        with self.client.websocket_connect("/ws/live?source=micro&titre=essai") as ws:
            while True:                                          # attendre « prêt » : avant, le serveur jette l'audio
                m = json.loads(ws.receive_text())
                messages.append(m)
                if m["type"] == "etat" and m["etat"] in ("pret", "en_cours"):
                    break
            sid = m["sid"]
            pas = SR // 4
            for debut in range(0, len(self.signal), pas):
                ws.send_bytes(pcm16(self.signal[debut: debut + pas]))
            ws.send_text("stop")
            while True:
                m = json.loads(ws.receive_text())
                messages.append(m)
                if m["type"] == "fin":
                    break
        return sid, messages

    def test_page_et_ressources(self):
        r = self.client.get("/live")
        self.assertEqual(r.status_code, 200)
        self.assertIn("marque.js", r.text)          # le nom vit dans static/marque.js
        for nom in ("live.js", "live.css", "pcm-worklet.js"):
            self.assertEqual(self.client.get(f"/live/assets/{nom}").status_code, 200, nom)
        self.assertEqual(self.client.get("/live/assets/../live.py").status_code, 404)
        statut = self.client.get("/api/live/statut").json()
        self.assertTrue(statut["moteur_disponible"] and statut["factice"])

    def test_session_websocket_complete(self):
        sid, messages = self.rejouer_ws()
        definitifs = [m for m in messages if m["type"] == "definitif"]
        self.assertEqual([m["fr"] for m in definitifs], [live_moteurs.PHRASES[i][2] for i in range(3)])
        # gris puis blanc : des provisoires arrivent, et aucun après le définitif de son segment
        # (l'audio part en rafale : le serveur peut regrouper des paquets, donc pas de minimum par segment)
        self.assertTrue([m for m in messages if m["type"] == "provisoire"])
        for d in definitifs:
            apres = [i for i, m in enumerate(messages) if m["type"] == "provisoire" and m["id_segment"] == d["id_segment"]
                     and i > messages.index(d)]
            self.assertEqual(apres, [])
        ids = [m["id_segment"] for m in definitifs]
        self.assertEqual(len(ids), len(set(ids)))
        # segments persistés, audio écrit, bilan
        session = Session.ouvrir(sid)
        segs = session.segments()
        self.assertEqual([s["fr"] for s in segs], [m["fr"] for m in definitifs])
        self.assertTrue(all(s["ar"] and s["debut_s"] is not None for s in segs))
        self.assertTrue(session.chemin_audio.is_file())
        self.assertAlmostEqual(session.meta()["duree_s"], len(self.signal) / SR, delta=0.3)
        fin = messages[-1]
        self.assertEqual(fin["bilan"]["nb_segments"], 3)
        self.assertIn("retard_moyen_s", fin["bilan"])
        self.assertIn("taux_reecriture_pct", fin["bilan"])

    def test_spectateur_rejoint_apres_coup(self):
        sid, _ = self.rejouer_ws()
        with self.client.websocket_connect(f"/ws/regarder/{sid}") as ws:
            vus = []
            while True:
                m = json.loads(ws.receive_text())
                vus.append(m)
                if m["type"] == "fin":
                    break
        self.assertEqual(len([m for m in vus if m["type"] == "definitif"]), 3)

    def test_spectateur_session_inconnue(self):
        with self.client.websocket_connect("/ws/regarder/inconnue") as ws:
            self.assertEqual(json.loads(ws.receive_text())["etat"], "erreur")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg absent")
    def test_fichier_rejoue(self):
        wav = Path(_TMP) / "journal.wav"
        ecrire_wav(wav, self.signal)
        with open(wav, "rb") as f:
            r = self.client.post("/api/live/fichier", files={"file": ("journal.wav", f, "audio/wav")})
        self.assertEqual(r.status_code, 200)
        sid = r.json()["sid"]
        self.assertEqual(self.client.get(r.json()["media"]).status_code, 200)
        with self.client.websocket_connect(f"/ws/regarder/{sid}") as ws:
            while json.loads(ws.receive_text()).get("etat") not in ("pret", "en_cours"):
                pass
            self.assertEqual(self.client.post(f"/api/live/{sid}/demarrer").status_code, 200)
            vus = []
            while True:
                m = json.loads(ws.receive_text())
                vus.append(m)
                if m["type"] == "fin":
                    break
        self.assertEqual(len([m for m in vus if m["type"] == "definitif"]), 3)
        arret = self.client.post(f"/api/live/{sid}/arreter").json()
        self.assertTrue(arret["termine"])
        self.assertEqual(arret["bilan"]["nb_segments"], 3)
        self.assertEqual(len(Session.ouvrir(sid).segments()), 3)

    def test_url_invalide(self):
        self.assertEqual(self.client.post("/api/live/url", json={"url": "pas un lien"}).status_code, 422)

    def test_commande_ffmpeg(self):
        c = live.commande_ffmpeg("https://x/y/index.m3u8")
        self.assertIn("-re", c)
        self.assertEqual(c[c.index("-live_start_index") + 1], "-1")
        self.assertEqual(c[c.index("-ar") + 1], "16000")
        self.assertNotIn("-re", live.commande_ffmpeg("a.wav", temps_reel=False))
        self.assertNotIn("-live_start_index", live.commande_ffmpeg("a.wav"))

    def test_resoudre_url_directe(self):
        self.assertEqual(live.resoudre_url("https://x/flux.m3u8?a=1"), "https://x/flux.m3u8?a=1")


class TestJeton(unittest.TestCase):
    def test_jeton_exige(self):
        os.environ["PORTEPAROLE_API_TOKEN"] = "secret"
        try:
            live.configurer(factice=True)
            client = TestClient(live.creer_app())
            self.assertEqual(client.get("/api/live/statut").status_code, 401)
            self.assertEqual(client.get("/api/live/statut?token=secret").status_code, 200)
            self.assertEqual(client.get("/api/live/statut", headers={"Authorization": "Bearer secret"}).status_code, 200)
            self.assertEqual(client.get("/live").status_code, 200)          # la page elle-même reste publique
        finally:
            del os.environ["PORTEPAROLE_API_TOKEN"]


class TestGpuLock(unittest.TestCase):
    def test_un_seul_modele_a_la_fois(self):
        import threading

        import lazy_engines
        actifs, maxi, verrou = [0], [0], threading.Lock()

        class Faux:
            def run(self, *a, **k):
                with verrou:
                    actifs[0] += 1
                    maxi[0] = max(maxi[0], actifs[0])
                time.sleep(0.05)
                with verrou:
                    actifs[0] -= 1
                return "ok"

        pool = lazy_engines.Pool()
        moteurs = [lazy_engines.LazyEngine(pool, "g", "m", "M", "", lambda: {"m": Faux()}, True) for _ in range(4)]
        fils = [threading.Thread(target=m.run, args=("x",)) for m in moteurs]
        [f.start() for f in fils]
        [f.join() for f in fils]
        self.assertEqual(maxi[0], 1)

    def test_un_nom_un_moteur(self):
        import lazy_engines
        moteurs = lazy_engines.build_lazy_engines("/inexistant", only={"s3", "s3_direct"})
        self.assertEqual(set(moteurs), {"s3", "s3_direct"})
        self.assertTrue(all(name == m.name for name, m in moteurs.items()))
        self.assertTrue(moteurs["s3"].produces_arabic and not moteurs["s3_direct"].produces_arabic)


if __name__ == "__main__":
    unittest.main()
