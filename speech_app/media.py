"""Conversion d'un fichier audio ou vidéo en wav mono 16 kHz (ffmpeg), partagée par l'API et l'interface."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

NATIVE_AUDIO = (".wav", ".flac", ".ogg")                       # lisibles sans conversion
VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")


class MediaError(Exception):
    """Fichier illisible ou ffmpeg absent ; le message est destiné à l'utilisateur."""


def is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXT


def to_wav(src: str, folder: str) -> str:
    """Audio ou vidéo -> wav mono 16 kHz dans `folder`."""
    if not shutil.which("ffmpeg"):
        if Path(src).suffix.lower() in NATIVE_AUDIO:
            return src
        raise MediaError("ffmpeg n'est pas installé sur le serveur (requis pour les vidéos et le mp3)")
    dst = str(Path(folder) / "audio.wav")
    proc = subprocess.run(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", dst],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not Path(dst).is_file():
        raise MediaError(f"Fichier illisible par ffmpeg : {proc.stderr[-300:]}")
    return dst
