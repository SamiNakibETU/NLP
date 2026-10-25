"""Une session = un dossier qui garde l'audio et les segments d'une écoute en direct ou d'un fichier déposé.

C'est le contrat entre le direct (live.py), qui écrit la session au fil de l'eau, et le rapport de
fin de session (session_report.py), qui la relit pour produire transcript, locuteurs et résumé.

    <PORTEPAROLE_SESSIONS>/<sid>/
        meta.json        {"sid", "source", "moteur", "cree_le", "duree_s", "titre"}
        audio.wav        16 kHz mono int16, tout l'audio reçu
        segments.jsonl   une ligne par segment clos : {"id", "debut_s", "fin_s", "ar", "fr", "locuteur"}
        rapport/         sorties du rapport (transcript.md, resume.md, sous_titres.srt, rapport.json)
"""
from __future__ import annotations

import json
import os
import time
import uuid
import wave
from pathlib import Path

import numpy as np

FREQUENCE = 16000


def dossier_sessions() -> Path:
    racine = Path(os.environ.get("PORTEPAROLE_SESSIONS", Path(__file__).resolve().parent / "sessions"))
    racine.mkdir(parents=True, exist_ok=True)
    return racine


class Session:
    def __init__(self, dossier: Path):
        self.dossier = dossier
        self.sid = dossier.name

    @classmethod
    def creer(cls, source: str, moteur: str, titre: str = "") -> "Session":
        sid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        dossier = dossier_sessions() / sid
        dossier.mkdir(parents=True)
        session = cls(dossier)
        session.ecrire_meta({"sid": sid, "source": source, "moteur": moteur,
                             "cree_le": time.strftime("%Y-%m-%dT%H:%M:%S"), "duree_s": 0.0, "titre": titre})
        return session

    @classmethod
    def ouvrir(cls, sid: str) -> "Session":
        dossier = dossier_sessions() / sid
        if not dossier.is_dir() or Path(sid).name != sid:
            raise FileNotFoundError(f"session inconnue : {sid}")
        return cls(dossier)

    @property
    def chemin_audio(self) -> Path:
        return self.dossier / "audio.wav"

    @property
    def dossier_rapport(self) -> Path:
        d = self.dossier / "rapport"
        d.mkdir(exist_ok=True)
        return d

    def meta(self) -> dict:
        return json.loads((self.dossier / "meta.json").read_text(encoding="utf-8"))

    def ecrire_meta(self, meta: dict) -> None:
        (self.dossier / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def ecrire_audio(self, signal: np.ndarray) -> None:
        """Signal float32 [-1, 1] à 16 kHz ; remplace l'audio déjà écrit."""
        pcm = (np.clip(signal, -1.0, 1.0) * 32767).astype("<i2")
        with wave.open(str(self.chemin_audio), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(FREQUENCE)
            f.writeframes(pcm.tobytes())
        meta = self.meta()
        meta["duree_s"] = round(len(signal) / FREQUENCE, 2)
        self.ecrire_meta(meta)

    def ajouter_segment(self, segment: dict) -> None:
        ligne = {k: segment.get(k) for k in ("id", "debut_s", "fin_s", "ar", "fr", "locuteur")}
        with open(self.dossier / "segments.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(ligne, ensure_ascii=False) + "\n")

    def segments(self) -> list[dict]:
        chemin = self.dossier / "segments.jsonl"
        if not chemin.is_file():
            return []
        return [json.loads(l) for l in chemin.read_text(encoding="utf-8").splitlines() if l.strip()]
