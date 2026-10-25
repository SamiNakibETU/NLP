"""Lecture vocale du français (résumé, transcript) : Piper en local et gratuit, edge-tts en repli.

    PORTEPAROLE_TTS = piper | edge | off        (défaut : piper, puis edge-tts si Piper est indisponible)
    PORTEPAROLE_TTS_VOIX = fr_FR-siwis-medium   (ou fr_FR-upmc-medium)

Piper (paquet `piper-tts`, voix ONNX) tourne sur CPU : aucune VRAM, pas de clé. La voix est téléchargée une fois
depuis le dépôt `rhasspy/piper-voices` de Hugging Face, puis gardée dans `speech_app/cache/tts/`.
edge-tts utilise les voix en ligne de Microsoft Edge (réseau requis, sans clé).
ElevenLabs serait l'option payante de qualité supérieure ; elle n'est PAS implémentée (pas de clé de service).
Les audios sont mis en cache d'après le hash du texte, de la voix et du moteur.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import urllib.request
import wave
from pathlib import Path

RACINE = Path(os.environ.get("PORTEPAROLE_CACHE", Path(__file__).resolve().parent / "cache")) / "tts"
VOIX_PIPER = os.environ.get("PORTEPAROLE_TTS_VOIX", "fr_FR-siwis-medium")
VOIX_EDGE = "fr-FR-DeniseNeural"
URL_VOIX = "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/{nom}/{qualite}/fr_FR-{nom}-{qualite}"
TEXTE_MAX = 4000
_voix_chargee: dict = {}


class TTSIndisponible(RuntimeError):
    """Aucun moteur de synthèse utilisable (désactivé, paquet absent, réseau coupé)."""


def mode() -> str:
    return os.environ.get("PORTEPAROLE_TTS", "piper").strip().lower()


def nettoyer(texte: str) -> str:
    """Retire la mise en forme Markdown et les horodatages [mm:ss], qui ne se lisent pas à voix haute."""
    import re
    t = re.sub(r"\[\d{1,3}:\d{2}(?::\d{2})?\]", "", texte)
    t = re.sub(r"^[#>\-\*\s]+", "", t, flags=re.M)
    return re.sub(r"[*_`]+", "", re.sub(r"\s+", " ", t)).strip()[:TEXTE_MAX]


def _cle(texte: str, moteur: str, voix: str) -> str:
    return hashlib.sha1(f"{moteur}|{voix}|{texte}".encode("utf-8")).hexdigest()[:20]


def _fichier_voix_piper(nom: str) -> Path:
    onnx = RACINE / f"{nom}.onnx"
    if onnx.is_file() and Path(str(onnx) + ".json").is_file():
        return onnx
    _, voix_nom, qualite = nom.split("-")                 # fr_FR-siwis-medium
    base = URL_VOIX.format(nom=voix_nom, qualite=qualite)
    RACINE.mkdir(parents=True, exist_ok=True)
    for ext in (".onnx.json", ".onnx"):
        tmp = RACINE / (nom + ext + ".part")
        urllib.request.urlretrieve(base + ext, tmp)
        tmp.replace(RACINE / (nom + ext))
    return onnx


def _piper(texte: str, sortie: Path) -> None:
    try:
        from piper import PiperVoice
    except ImportError as exc:
        raise TTSIndisponible("piper-tts n'est pas installé") from exc
    try:
        if VOIX_PIPER not in _voix_chargee:
            _voix_chargee[VOIX_PIPER] = PiperVoice.load(str(_fichier_voix_piper(VOIX_PIPER)))
    except OSError as exc:
        raise TTSIndisponible(f"voix Piper {VOIX_PIPER} introuvable ou téléchargement impossible : {exc}") from exc
    voix = _voix_chargee[VOIX_PIPER]
    with wave.open(str(sortie), "wb") as f:
        if hasattr(voix, "synthesize_wav"):               # piper-tts >= 1.3
            voix.synthesize_wav(texte, f)
        else:                                             # piper-tts 1.2
            voix.synthesize(texte, f)


def _edge(texte: str, sortie: Path) -> None:
    try:
        import edge_tts
    except ImportError as exc:
        raise TTSIndisponible("edge-tts n'est pas installé") from exc
    try:
        asyncio.run(edge_tts.Communicate(texte, VOIX_EDGE).save(str(sortie)))
    except Exception as exc:                              # noqa: BLE001 : réseau, service
        raise TTSIndisponible(f"edge-tts a échoué : {exc}") from exc


def synthese(texte: str, moteurs: dict | None = None) -> tuple[Path, str]:
    """Texte français -> (fichier audio, type MIME). Lève TTSIndisponible. `moteurs` : faux pour les tests."""
    m = mode()
    if m == "off":
        raise TTSIndisponible("la lecture vocale est désactivée (PORTEPAROLE_TTS=off)")
    texte = nettoyer(texte)
    if not texte:
        raise ValueError("texte vide")
    moteurs = moteurs or {"piper": (_piper, ".wav", "audio/wav"), "edge": (_edge, ".mp3", "audio/mpeg")}
    ordre = ["edge"] if m == "edge" else ["piper", "edge"]
    erreurs = []
    RACINE.mkdir(parents=True, exist_ok=True)
    def chemin_de(nom):
        return RACINE / f"{_cle(texte, nom, VOIX_PIPER if nom == 'piper' else VOIX_EDGE)}{moteurs[nom][1]}"

    for nom in ordre:                                     # un audio déjà synthétisé par n'importe quel moteur sert
        if chemin_de(nom).is_file() and chemin_de(nom).stat().st_size > 0:
            return chemin_de(nom), moteurs[nom][2]
    for nom in ordre:
        fonction, ext, mime = moteurs[nom]
        chemin = chemin_de(nom)
        try:
            tmp = chemin.with_suffix(ext + ".tmp")
            fonction(texte, tmp)
            tmp.replace(chemin)
            return chemin, mime
        except TTSIndisponible as exc:
            erreurs.append(str(exc))
    raise TTSIndisponible(" ; ".join(erreurs))
