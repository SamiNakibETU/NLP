"""Module du système S3 (regroupement audio, projecteur MLP et entraînement en deux temps)."""

from .architecture_s3 import (
    ProjecteurAudioLLM,
    assembler_entree_s3,
    charger_config_s3,
    regrouper_vecteurs_audio,
)
from .entrainement_s3 import (
    DecodeurLLMPourInputsEmbeds,
    construire_exemples_audio_s3,
    diagnostiquer_stagnation_s3,
    entrainer_s3_temps1,
    entrainer_s3_temps2_ou_un_temps,
)

__all__ = [
    "ProjecteurAudioLLM",
    "assembler_entree_s3",
    "charger_config_s3",
    "regrouper_vecteurs_audio",
    "DecodeurLLMPourInputsEmbeds",
    "construire_exemples_audio_s3",
    "diagnostiquer_stagnation_s3",
    "entrainer_s3_temps1",
    "entrainer_s3_temps2_ou_un_temps",
]
