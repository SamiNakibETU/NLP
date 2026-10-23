"""Récupération d'un extrait de vidéo en ligne pour test local de recherche.

Dépend de yt-dlp et ffmpeg (présents sur Colab). Le téléchargement contrevient aux conditions d'utilisation de
YouTube et aux droits des diffuseurs : usage limité à un essai de recherche, sans republication de la vidéo ni des
sous-titres produits. `missing_tools()` dit ce qui manque, pour afficher un message clair dans l'interface.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

MAX_SECONDS = 180
VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def missing_tools() -> list[str]:
    missing = []
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        missing.append("yt-dlp")
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg")
    return missing


def fetch_clip(video_id: str, out_dir: str | Path, seconds: int = 90, start: int = 0) -> tuple[str, str]:
    """Télécharge `seconds` secondes de la vidéo (basse définition) et en extrait un WAV 16 kHz mono.
    Retourne (chemin de la vidéo mp4, chemin du wav)."""
    if not VIDEO_ID.match(video_id):
        raise ValueError("identifiant de vidéo invalide")
    if (m := missing_tools()):
        raise RuntimeError("Outils manquants : " + ", ".join(m))
    import yt_dlp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seconds = max(5, min(int(seconds), MAX_SECONDS))
    opts = {
        "format": "best[height<=480][ext=mp4]/best[height<=480]/best",
        "outtmpl": str(out_dir / f"{video_id}.%(ext)s"),
        "download_ranges": yt_dlp.utils.download_range_func(None, [(start, start + seconds)]),
        "force_keyframes_at_cuts": True, "quiet": True, "no_warnings": True, "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    video = next(iter(sorted(out_dir.glob(f"{video_id}.*"))), None)
    if not video:
        raise RuntimeError("Téléchargement impossible (vidéo indisponible ou bloquée).")
    wav = out_dir / f"{video_id}.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                   check=True)
    return str(video), str(wav)
