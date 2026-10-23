"""Pages de l'interface (accueil, vidéo, rapport, atelier) et session créée par POST /api/translate, en mode démo."""
import math
import os
import shutil
import struct
import sys
import tempfile
import time
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PORTEPAROLE_SESSIONS", tempfile.mkdtemp(prefix="pp_pages_"))
os.environ.pop("PORTEPAROLE_ENGINES", None)

from starlette.testclient import TestClient  # noqa: E402

import api  # noqa: E402
import pages  # noqa: E402
from sessions import Session  # noqa: E402

client = TestClient(api.app)


def test_pages_servies():
    for chemin, mot in (("/", "live.js"), ("/live", "live.js"), ("/sessions", "sessions.js"), ("/video", "video.js"), ("/rapport?sid=x", "rapport.js")):
        r = client.get(chemin)
        assert r.status_code == 200 and mot in r.text, chemin
    assert client.get("/atelier/").status_code == 200


def test_statiques_en_liste_blanche():
    for nom in pages.ASSETS:
        assert client.get(f"/static/{nom}").status_code == 200, nom
    assert client.get("/static/../api.py").status_code == 404
    assert client.get("/static/inconnu.js").status_code == 404


def test_manifeste_pwa():
    r = client.get("/manifest.webmanifest")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/manifest+json")
    m = r.json()
    assert m["display"] == "standalone" and m["start_url"] == "/"
    for icone in m["icons"]:
        assert client.get(icone["src"]).status_code == 200, icone["src"]


def test_le_nom_vit_dans_une_seule_constante():
    marque = (pages.STATIC / "marque.js").read_text(encoding="utf-8")
    assert "Kalam" in marque
    for html in [pages.STATIC / PAGE for PAGE in pages.PAGES.values()]:
        # le texte statique sert de repli ; marque.js doit être chargé par chaque page
        assert "marque.js" in html.read_text(encoding="utf-8"), html.name


def test_moteurs_exposent_leur_groupe():
    j = client.get("/api/engines").json()
    assert j["engines"] and all("group" in e for e in j["engines"])


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg absent")
def test_translate_renvoie_un_sid(tmp_path):
    wav = tmp_path / "essai.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(i / 20))) for i in range(16000 * 8)))
    with open(wav, "rb") as f:
        r = client.post("/api/translate", files={"file": ("essai.wav", f, "audio/wav")}, data={"wait": "false"})
    assert r.status_code == 200
    job = r.json()["job_id"]
    for _ in range(100):
        j = client.get(f"/api/jobs/{job}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert j["status"] == "done", j
    sid = j["result"]["sid"]
    assert sid
    session = Session.ouvrir(sid)
    assert session.chemin_audio.is_file() and session.segments()
    assert any(s["sid"] == sid for s in client.get("/api/sessions").json()["sessions"])
