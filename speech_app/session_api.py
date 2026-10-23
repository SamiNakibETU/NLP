"""Routes des sessions : liste, rapport de fin de session (locuteurs + résumé), fichiers produits, lecture vocale.

    GET  /api/sessions                          sessions existantes (les plus récentes d'abord)
    POST /api/sessions/{sid}/rapport            lance la génération en tâche de fond (corps JSON : options facultatives)
    GET  /api/sessions/{sid}/rapport            état (absent|en_cours|pret|erreur) et, une fois prêt, le rapport
    GET  /api/sessions/{sid}/fichiers/{nom}     transcript.md, resume.md, sous_titres.srt, rapport.json, transcript.docx, audio.wav
    GET  /api/tts?texte=...                     lecture vocale du français (wav ou mp3), voir tts.py

À inclure dans api.py : `api.include_router(session_api.router)`. Même jeton facultatif que le reste de l'API.
"""
from __future__ import annotations

import json
import os
import threading
import time

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import FileResponse

import session_report
import tts
from sessions import Session, dossier_sessions

router = APIRouter()
ETATS: dict[str, dict] = {}
VERROU = threading.Lock()
OPTIONS_AUTORISEES = {"diarisation", "nb_locuteurs", "seuil_similarite", "revision", "resume", "inclure_arabe", "docx"}
FICHIERS_PUBLICS = set(session_report.FICHIERS) | {"audio.wav", "segments.jsonl", "meta.json"}


def verifier_jeton(request: Request) -> None:
    jeton = os.environ.get("PORTEPAROLE_API_TOKEN") or None
    if jeton and request.headers.get("authorization", "") != f"Bearer {jeton}":
        raise HTTPException(401, "Jeton manquant ou invalide")


def _ouvrir(sid: str) -> Session:
    try:
        return Session.ouvrir(sid)
    except FileNotFoundError:
        raise HTTPException(404, f"Session inconnue : {sid}")


def _pool():
    """Pool des moteurs de l'interface, à décharger avant le LLM ; None si introuvable."""
    try:
        import app as ui
        return next((e.pool for e in ui.ENGINES.values() if hasattr(e, "pool")), None)
    except Exception:                                      # noqa: BLE001
        return None


def lancer_rapport(sid: str, options: dict | None = None, *, synchrone: bool = False) -> dict:
    """Démarre (ou renvoie l'état de) la génération du rapport pour `sid`."""
    session = _ouvrir(sid)
    with VERROU:
        if ETATS.get(sid, {}).get("etat") == "en_cours":
            return ETATS[sid]
        ETATS[sid] = {"sid": sid, "etat": "en_cours", "etape": "En attente du GPU", "progression": 0.0,
                      "debut": time.time()}

    def progression(f, d=""):
        ETATS[sid].update(progression=round(float(f), 2), etape=d)

    def travail():
        try:
            rapport = session_report.generer_rapport(
                session, {**(options or {}), "progress": progression, "pool": _pool()})
            ETATS[sid].update(etat="pret", progression=1.0, etape="Rapport prêt", fichiers=rapport["fichiers"])
        except Exception as exc:                           # noqa: BLE001 : l'erreur est renvoyée au client
            ETATS[sid].update(etat="erreur", erreur=f"{type(exc).__name__}: {exc}")

    if synchrone:
        travail()
    else:
        threading.Thread(target=travail, name=f"rapport-{sid}", daemon=True).start()
    return ETATS[sid]


@router.get("/api/sessions", dependencies=[Depends(verifier_jeton)])
def lister_sessions():
    out = []
    for d in sorted(dossier_sessions().iterdir(), reverse=True):
        if not (d / "meta.json").is_file():
            continue
        s = Session(d)
        meta = s.meta()
        out.append({**meta, "n_segments": len(s.segments()), "rapport": (d / "rapport" / "rapport.json").is_file()})
    return {"sessions": out}


@router.post("/api/sessions/{sid}/rapport", dependencies=[Depends(verifier_jeton)], status_code=202)
def demarrer_rapport(sid: str, options: dict = Body(default_factory=dict)):
    inconnues = set(options) - OPTIONS_AUTORISEES
    if inconnues:
        raise HTTPException(422, f"Options inconnues : {', '.join(sorted(inconnues))}")
    if not _ouvrir(sid).segments():
        raise HTTPException(409, "La session ne contient encore aucun segment")
    return lancer_rapport(sid, options)


@router.get("/api/sessions/{sid}/rapport", dependencies=[Depends(verifier_jeton)])
def lire_rapport(sid: str):
    session = _ouvrir(sid)
    etat = ETATS.get(sid)
    if etat and etat["etat"] in ("en_cours", "erreur"):
        return etat
    fichier = session.dossier_rapport / "rapport.json"
    if fichier.is_file():                                  # prêt, y compris après un redémarrage du serveur
        return {"sid": sid, "etat": "pret", "rapport": json.loads(fichier.read_text(encoding="utf-8"))}
    return {"sid": sid, "etat": "absent"}


@router.get("/api/sessions/{sid}/fichiers/{nom}", dependencies=[Depends(verifier_jeton)])
def fichier_session(sid: str, nom: str):
    session = _ouvrir(sid)
    if nom not in FICHIERS_PUBLICS:
        raise HTTPException(404, "Fichier inconnu")
    chemin = (session.dossier if nom in ("audio.wav", "segments.jsonl", "meta.json") else session.dossier / "rapport") / nom
    if not chemin.is_file():
        raise HTTPException(404, f"{nom} n'existe pas encore pour cette session")
    return FileResponse(chemin, filename=nom)


@router.get("/api/tts", dependencies=[Depends(verifier_jeton)])
def lecture_vocale(texte: str):
    try:
        chemin, mime = tts.synthese(texte)
    except ValueError:
        raise HTTPException(422, "Le texte à lire est vide")
    except tts.TTSIndisponible as exc:
        raise HTTPException(503, f"Lecture vocale indisponible : {exc}")
    return FileResponse(chemin, media_type=mime)
