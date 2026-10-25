"""Courbes de l'entraînement S3 (matplotlib) : perte, WER/CER de validation, balayage du taux.

Deux mesures d'échelles différentes vont dans deux panneaux, jamais sur deux axes y.
Couleurs : palette catégorielle de référence, dans l'ordre (bleu, orange, aqua).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

COULEURS = ["#2a78d6", "#eb6834", "#1baf7a"]
ENCRE, ENCRE_SECONDAIRE, GRILLE, FOND = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(axe, titre: str, x: str, y: str) -> None:
    axe.set_facecolor(FOND)
    axe.set_title(titre, loc="left", fontsize=11, color=ENCRE)
    axe.set_xlabel(x, color=ENCRE_SECONDAIRE)
    axe.set_ylabel(y, color=ENCRE_SECONDAIRE)
    axe.grid(True, color=GRILLE, linewidth=0.8)
    axe.tick_params(colors=ENCRE_SECONDAIRE)
    for cote in ("top", "right"):
        axe.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        axe.spines[cote].set_color(GRILLE)


def moyenne_glissante(valeurs: Sequence[float], fenetre: int) -> List[float]:
    sortie, somme = [], 0.0
    for i, v in enumerate(valeurs):
        somme += v
        if i >= fenetre:
            somme -= valeurs[i - fenetre]
        sortie.append(somme / min(i + 1, fenetre))
    return sortie


def _etiquette_fin(axe, x, y, texte: str, couleur: str) -> None:
    axe.plot([x], [y], "o", color=couleur, markersize=5, markeredgecolor=FOND, markeredgewidth=1.5)
    axe.annotate(texte, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=ENCRE)


def tracer_courbes(metriques: List[Dict[str, Any]], chemin: Optional[Path] = None, titre: str = "S3 temps 1"):
    """Panneau 1 : perte d'entraînement (moyenne glissante) et de validation. Panneau 2 : WER et CER."""
    import matplotlib.pyplot as plt

    train = [l for l in metriques if l["type"] == "train"]
    validation = [l for l in metriques if l["type"] == "validation"]
    avec_wer = [l for l in validation if "wer_pct" in l]
    figure, axes = plt.subplots(1, 2 if avec_wer else 1, figsize=(12 if avec_wer else 6.5, 4.2), squeeze=False)
    figure.patch.set_facecolor(FOND)
    axe = axes[0][0]
    _style(axe, "Perte (entropie croisée par token)", "étape (lot de 16)", "perte")
    if train:
        x = [l["etape"] for l in train]
        brut = [l["perte"] for l in train]
        axe.plot(x, brut, color=COULEURS[0], linewidth=1, alpha=0.25)
        lisse = moyenne_glissante(brut, 10)
        axe.plot(x, lisse, color=COULEURS[0], linewidth=2, label="entraînement (moyenne sur 100 pas)")
        _etiquette_fin(axe, x[-1], lisse[-1], f"{lisse[-1]:.2f}", COULEURS[0])
    if validation:
        x = [l["etape"] for l in validation]
        y = [l["perte"] for l in validation]
        axe.plot(x, y, color=COULEURS[1], linewidth=2, marker="o", markersize=5, label="validation (200 segments)")
        _etiquette_fin(axe, x[-1], y[-1], f"{y[-1]:.2f}", COULEURS[1])
    axe.legend(frameon=False, fontsize=9, labelcolor=ENCRE)
    if avec_wer:
        axe = axes[0][1]
        _style(axe, "Erreur de transcription en validation", "étape (lot de 16)", "taux d'erreur (%)")
        x = [l["etape"] for l in avec_wer]
        for numero, (cle, nom) in enumerate((("wer_pct", "WER"), ("cer_pct", "CER"))):
            y = [l[cle] for l in avec_wer]
            axe.plot(x, y, color=COULEURS[numero], linewidth=2, marker="o", markersize=5, label=nom)
            _etiquette_fin(axe, x[-1], y[-1], f"{nom} {y[-1]:.1f} %", COULEURS[numero])
        axe.set_ylim(bottom=0)
        axe.legend(frameon=False, fontsize=9, labelcolor=ENCRE)
    figure.suptitle(titre, x=0.01, ha="left", fontsize=12, color=ENCRE)
    figure.tight_layout()
    if chemin:
        figure.savefig(chemin, dpi=150, facecolor=FOND)
    return figure


def tracer_balayage(resultats: List[Dict[str, Any]], chemin: Optional[Path] = None):
    """Perte de validation après le court entraînement, en fonction du taux d'apprentissage (échelle log)."""
    import matplotlib.pyplot as plt

    figure, axe = plt.subplots(figsize=(6.5, 4))
    figure.patch.set_facecolor(FOND)
    _style(axe, "Balayage du taux d'apprentissage", "taux d'apprentissage (échelle log)", "perte de validation")
    x = [r["learning_rate"] for r in resultats]
    y = [r["perte_validation"] for r in resultats]
    axe.plot(x, y, color=COULEURS[0], linewidth=2, marker="o", markersize=6)
    axe.set_xscale("log")
    meilleur = min(resultats, key=lambda r: r["perte_validation"])
    for r in resultats:
        axe.annotate(f"{r['perte_validation']:.3f}", (r["learning_rate"], r["perte_validation"]), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=9,
                     color=ENCRE if r is meilleur else ENCRE_SECONDAIRE,
                     fontweight="bold" if r is meilleur else "normal")
    figure.tight_layout()
    if chemin:
        figure.savefig(chemin, dpi=150, facecolor=FOND)
    return figure
