"""Génération et lecture de fichiers de sous-titres au format SubRip (.srt).

Format standard SubRip :
1
00:00:01,250 --> 00:00:04,800
Texte du sous-titre en français.
"""

from pathlib import Path
from typing import Dict, List, Sequence, Union


def formater_horodatage_srt(secondes: float) -> str:
    """Convertit un temps en secondes (float >= 0) au format SRT `HH:MM:SS,mmm`."""
    if secondes < 0:
        raise ValueError(f"L'horodatage ne peut pas être négatif : {secondes}")
    total_ms = int(round(float(secondes) * 1000.0))
    heures = total_ms // 3_600_000
    reste = total_ms % 3_600_000
    minutes = reste // 60_000
    reste = reste % 60_000
    secs = reste // 1000
    ms = reste % 1000
    return f"{heures:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def generer_contenu_srt(segments: Sequence[Dict[str, Union[float, str]]]) -> str:
    """Construit le contenu complet d'un fichier `.srt` à partir d'une liste de segments.

    Chaque segment doit contenir :
    - `debut_s` (float) : début en secondes
    - `fin_s` (float) : fin en secondes (strictement supérieur ou égal à `debut_s`)
    - `texte_fr` ou `texte` (str) : texte du sous-titre
    """
    blocs: List[str] = []
    numero = 1
    for seg in segments:
        debut = float(seg["debut_s"])
        fin = float(seg["fin_s"])
        if fin < debut:
            raise ValueError(f"Segment invalide : fin_s ({fin}) < debut_s ({debut}).")
        texte = str(seg.get("texte_fr", seg.get("texte", ""))).strip()
        if not texte:
            continue
        debut_str = formater_horodatage_srt(debut)
        fin_str = formater_horodatage_srt(fin)
        blocs.append(f"{numero}\n{debut_str} --> {fin_str}\n{texte}")
        numero += 1
    return "\n\n".join(blocs) + ("\n" if blocs else "")


def sauvegarder_srt(
    segments: Sequence[Dict[str, Union[float, str]]],
    chemin_srt: Union[str, Path],
) -> Path:
    """Écrit les segments au format `.srt` dans `chemin_srt` et renvoie le chemin."""
    cible = Path(chemin_srt)
    cible.parent.mkdir(parents=True, exist_ok=True)
    contenu = generer_contenu_srt(segments)
    cible.write_text(contenu, encoding="utf-8")
    return cible
