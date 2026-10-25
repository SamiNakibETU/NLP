"""Lecture du passage utile d'un segment MGB-2 (étape 2 du plan S3).

Repris de ``projecteur_v1/speech_llm/audio.py`` (dépôt MLP), sans la limite de
30 s ni les features Whisper : on lit seulement les échantillons du segment,
jamais l'émission entière (souvent 25 min ou plus).
"""

from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

# Format des wav MGB-2 et format d'entrée du préparateur Cohere Transcribe.
FREQUENCE_ATTENDUE = 16000


class ErreurAudio(Exception):
    """Fichier audio absent ou au mauvais format."""


def charger_extrait(entree: Dict[str, Any], racine: Union[str, Path],
                    frequence_attendue: int = FREQUENCE_ATTENDUE, tolerance_fin_s: float = 0.0) -> np.ndarray:
    """Échantillons float32 de ``racine/entree["wav"]`` entre ``debut`` et ``fin`` (secondes).

    Quelques horodatages MGB-2 finissent un peu après la fin du wav (0,06 s au plus sur
    50 000 segments) : jusqu'à ``tolerance_fin_s``, on coupe à la fin du fichier ; au-delà,
    c'est une erreur.
    """
    import soundfile as sf

    chemin = Path(racine) / entree["wav"]
    if not chemin.is_file():
        raise ErreurAudio(f"wav introuvable : {chemin}")
    infos = sf.info(str(chemin))
    if infos.samplerate != frequence_attendue:
        raise ErreurAudio(f"{chemin} : {infos.samplerate} Hz, {frequence_attendue} Hz attendus")
    if infos.channels != 1:
        raise ErreurAudio(f"{chemin} : {infos.channels} canaux, mono attendu")
    debut = round(float(entree["debut"]) * infos.samplerate)
    fin = round(float(entree["fin"]) * infos.samplerate)
    if infos.frames < fin <= infos.frames + round(tolerance_fin_s * infos.samplerate):
        fin = infos.frames
    if not 0 <= debut < fin <= infos.frames:
        raise ErreurAudio(f"{chemin} : passage [{entree['debut']}, {entree['fin']}] s hors du fichier "
                          f"({infos.frames / infos.samplerate:.2f} s)")
    extrait, _ = sf.read(str(chemin), start=debut, stop=fin, dtype="float32")
    return extrait
