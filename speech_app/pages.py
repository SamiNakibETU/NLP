"""Pages de l'interface : accueil, dépôt de vidéo, rapport. Statiques, servies depuis static/.

    GET /            accueil (trois entrées, état des moteurs, sessions)
    GET /sessions    renvoie vers la liste des sessions de l'accueil
    GET /video       dépôt d'un fichier, sous-titres synchronisés
    GET /rapport     rapport d'une session (?sid=...)
    GET /static/{nom}  feuilles de style et scripts partagés (liste blanche)

Le nom du produit vit dans static/marque.js, et nulle part ailleurs.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

STATIC = Path(__file__).resolve().parent / "static"
PAGES = {"/": "live.html", "/sessions": "sessions.html", "/video": "video.html", "/rapport": "rapport.html"}
ASSETS = {
    "base.css": "text/css", "marque.js": "text/javascript", "commun.js": "text/javascript",
    "sessions.js": "text/javascript", "video.css": "text/css", "video.js": "text/javascript",
    "rapport.css": "text/css", "rapport.js": "text/javascript",
    "manifest.webmanifest": "application/manifest+json", "icone.svg": "image/svg+xml",
    "icone-180.png": "image/png", "icone-192.png": "image/png", "icone-512.png": "image/png",
}

router = APIRouter()


def _page(nom: str) -> HTMLResponse:
    chemin = STATIC / nom
    if not chemin.is_file():
        raise HTTPException(404, f"{nom} introuvable")
    return HTMLResponse(chemin.read_text(encoding="utf-8"))


@router.get("/", include_in_schema=False)
def accueil():
    return _page(PAGES["/"])


@router.get("/sessions", include_in_schema=False)
def sessions():
    return _page(PAGES["/sessions"])


@router.get("/video", include_in_schema=False)
def video():
    return _page(PAGES["/video"])


@router.get("/rapport", include_in_schema=False)
def rapport():
    return _page(PAGES["/rapport"])


@router.get("/manifest.webmanifest", include_in_schema=False)
def manifeste():
    return FileResponse(STATIC / "manifest.webmanifest", media_type="application/manifest+json")


@router.get("/static/{nom}", include_in_schema=False)
def asset(nom: str):
    if nom not in ASSETS:
        raise HTTPException(404)
    return FileResponse(STATIC / nom, media_type=ASSETS[nom], headers={"Cache-Control": "no-cache"})
