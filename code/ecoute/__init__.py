"""Module d'écoute (ASR) et d'extraction des représentations de l'encodeur audio."""

from .encodeur import (
    ExtracteurEncodeurAudio,
    calculer_nb_vecteurs_utiles,
    extraire_vecteurs_sans_silence,
    lire_dimension_encodeur,
    lire_dimension_llm,
)
from .comparer_briques import choisir_meilleur_modele_ecoute, choisir_meilleur_llm

__all__ = [
    "ExtracteurEncodeurAudio",
    "calculer_nb_vecteurs_utiles",
    "extraire_vecteurs_sans_silence",
    "lire_dimension_encodeur",
    "lire_dimension_llm",
    "choisir_meilleur_modele_ecoute",
    "choisir_meilleur_llm",
]
