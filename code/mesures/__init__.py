"""Module de mesures et d'évaluation pour le projet de sous-titrage arabe -> français.

Contenu du module :
- Taux d'erreur de mots (WER) et de caractères (CER) après normaliser_arabe
- chrF++ et BLEU (avec signature sacrebleu)
- COMET (Unbabel/wmt22-comet-da)
- Facteur temps réel (RTF)
- Retard et taux de réécriture pour le mode direct
- Rééchantillonnage apparié (bootstrap à 1 000 tirages, intervalle à 95 %)
- Écriture normalisée dans resultats/resultats.csv
"""

from .scores import (
    calculer_wer_cer,
    calculer_chrf_bleu,
    calculer_comet,
    calculer_rtf,
    calculer_retard_et_reecriture,
)
from .reechantillonnage import comparer_systemes_bootstrap
from .tableau_resultats import enregistrer_resultat, lire_resultats

__all__ = [
    "calculer_wer_cer",
    "calculer_chrf_bleu",
    "calculer_comet",
    "calculer_rtf",
    "calculer_retard_et_reecriture",
    "comparer_systemes_bootstrap",
    "enregistrer_resultat",
    "lire_resultats",
]
