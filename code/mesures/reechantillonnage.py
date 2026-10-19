"""Rééchantillonnage apparié (paired bootstrap resampling) pour comparer deux systèmes.

Méthode :
- 1 000 tirages avec remise sur les indices des extraits
- Calcul de l'écart (Système B - Système A) à chaque tirage
- Intervalle de confiance à 95 % (percentiles 2,5 % et 97,5 %)
- Un système n'est déclaré "meilleur" que si l'intervalle à 95 % exclut zéro.
"""

import math
from pathlib import Path
import random
import sys
from typing import Dict, List, Sequence, Union

DOSSIER_MESURES = Path(__file__).resolve().parent
if str(DOSSIER_MESURES) not in sys.path:
    sys.path.insert(0, str(DOSSIER_MESURES))

from scores import calculer_chrf_bleu, calculer_wer_cer  # noqa: E402


METRIQUES_SUPPORTEES = ("chrf++", "bleu", "wer", "cer")


def _score_sur_echantillon(
    references: Sequence[str],
    hypotheses: Sequence[str],
    metrique: str,
) -> float:
    m = metrique.lower()
    if m == "chrf++":
        return float(calculer_chrf_bleu(references, hypotheses)["chrf_pp"])
    if m == "bleu":
        return float(calculer_chrf_bleu(references, hypotheses)["bleu"])
    if m == "wer":
        return float(calculer_wer_cer(references, hypotheses)["wer_pct"])
    if m == "cer":
        return float(calculer_wer_cer(references, hypotheses)["cer_pct"])
    raise ValueError(f"Métrique non supportée : {metrique!r}. Attendu : {METRIQUES_SUPPORTEES}")


def _percentile(valeurs_triees: Sequence[float], q: float) -> float:
    """Calcule le percentile q (entre 0 et 1) par interpolation linéaire."""
    if not valeurs_triees:
        raise ValueError("Liste vide pour le calcul de percentile.")
    if len(valeurs_triees) == 1:
        return float(valeurs_triees[0])
    pos = q * (len(valeurs_triees) - 1)
    bas = int(math.floor(pos))
    haut = int(math.ceil(pos))
    if bas == haut:
        return float(valeurs_triees[bas])
    poids = pos - bas
    return float(valeurs_triees[bas] * (1.0 - poids) + valeurs_triees[haut] * poids)


def comparer_systemes_bootstrap(
    references: Sequence[str],
    hypotheses_a: Sequence[str],
    hypotheses_b: Sequence[str],
    metrique: str = "chrf++",
    n_tirages: int = 1000,
    graine: int = 42,
) -> Dict[str, Union[float, bool, str, List[float]]]:
    """Compare deux systèmes A et B par rééchantillonnage apparié (1 000 tirages)."""
    n = len(references)
    if len(hypotheses_a) != n or len(hypotheses_b) != n:
        raise ValueError(
            f"Tailles incohérentes : references={n}, hypotheses_a={len(hypotheses_a)}, hypotheses_b={len(hypotheses_b)}."
        )
    if n == 0:
        raise ValueError("Impossible de faire un rééchantillonnage sur 0 segment.")
    if n_tirages <= 0:
        raise ValueError(f"n_tirages doit être strictement positif (reçu : {n_tirages}).")

    m = metrique.lower()
    if m not in METRIQUES_SUPPORTEES:
        raise ValueError(f"Métrique inconnue : {metrique!r}. Attendu : {METRIQUES_SUPPORTEES}")

    score_a = _score_sur_echantillon(references, hypotheses_a, m)
    score_b = _score_sur_echantillon(references, hypotheses_b, m)
    ecart_obs = round(score_b - score_a, 4)

    generateur = random.Random(graine)
    ecarts: List[float] = []

    for _ in range(n_tirages):
        indices = [generateur.randrange(n) for _ in range(n)]
        refs_tirage = [references[i] for i in indices]
        hyps_a_tirage = [hypotheses_a[i] for i in indices]
        hyps_b_tirage = [hypotheses_b[i] for i in indices]

        s_a = _score_sur_echantillon(refs_tirage, hyps_a_tirage, m)
        s_b = _score_sur_echantillon(refs_tirage, hyps_b_tirage, m)
        ecarts.append(s_b - s_a)

    ecarts.sort()
    ic_bas = round(_percentile(ecarts, 0.025), 4)
    ic_haut = round(_percentile(ecarts, 0.975), 4)

    contient_zero = (ic_bas <= 0.0 <= ic_haut)
    exclut_zero = not contient_zero

    plus_haut_meilleur = m in ("chrf++", "bleu")
    if not exclut_zero:
        conclusion = "écart non significatif (l'intervalle à 95 % contient zéro)"
    else:
        if plus_haut_meilleur:
            gagnant = "B meilleur que A" if ic_bas > 0 else "A meilleur que B"
        else:
            gagnant = "B meilleur que A (erreur plus basse)" if ic_haut < 0 else "A meilleur que B (erreur plus basse)"
        conclusion = f"écart significatif : {gagnant}"

    return {
        "metrique": m,
        "score_a": round(score_a, 4),
        "score_b": round(score_b, 4),
        "ecart_observe": ecart_obs,
        "ic_95_bas": ic_bas,
        "ic_95_haut": ic_haut,
        "intervalle_texte": f"[{ic_bas:+.2f}, {ic_haut:+.2f}]",
        "contient_zero": contient_zero,
        "exclut_zero": exclut_zero,
        "n_tirages": n_tirages,
        "graine": graine,
        "conclusion": conclusion,
    }
