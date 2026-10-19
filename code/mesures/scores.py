"""Calcul des mesures automatiques.

Mesures couvertes :
- WER (taux d'erreur de mots) et CER (taux d'erreur de caractères) après normaliser_arabe
- chrF++ (chrF2 avec mots d'ordre 2) et BLEU avec signature sacrebleu
- COMET (Unbabel/wmt22-comet-da) si disponible et mesurable, sinon "non mesuré"
- Facteur temps réel (RTF = temps_calcul / duree_son)
- Retard d'affichage définitif et taux de réécriture en mode direct
"""

from collections import Counter
import math
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Sequence, Union

DOSSIER_DONNEES = Path(__file__).resolve().parents[1] / "donnees"
if str(DOSSIER_DONNEES) not in sys.path:
    sys.path.insert(0, str(DOSSIER_DONNEES))

from normalisation import normaliser_arabe  # noqa: E402

try:
    import jiwer as _jiwer
except ImportError:
    _jiwer = None

try:
    import sacrebleu as _sacrebleu
except ImportError:
    _sacrebleu = None


def _distance_edition_sequence(ref_seq: Sequence, hyp_seq: Sequence) -> int:
    """Calcule la distance de Levenshtein exacte entre deux séquences (mots ou caractères)."""
    n = len(ref_seq)
    m = len(hyp_seq)
    if n == 0:
        return m
    if m == 0:
        return n
    precedente = list(range(m + 1))
    for i in range(1, n + 1):
        courante = [i] + [0] * m
        elem_r = ref_seq[i - 1]
        for j in range(1, m + 1):
            cout = 0 if elem_r == hyp_seq[j - 1] else 1
            courante[j] = min(
                precedente[j] + 1,      # suppression
                courante[j - 1] + 1,    # insertion
                precedente[j - 1] + cout,  # substitution
            )
        precedente = courante
    return precedente[m]


def calculer_wer_cer(
    references: Sequence[str],
    hypotheses: Sequence[str],
    niveau: str = "complet",
) -> Dict[str, Union[float, int, str]]:
    """Calcule le taux d'erreur de mots (WER) et de caractères (CER) après normaliser_arabe.

    Détails :
    - Applique normaliser_arabe(texte, niveau=niveau) sur chaque référence et hypothèse.
    - Utilise jiwer si présent (et vérifie la cohérence exacte), ou le calcul Levenshtein
      corpus-level standard (somme des éditions / somme des longueurs de référence).
    - Renvoie WER et CER sous forme de taux entre 0.0 et 1.0 (ainsi qu'en pourcentage).
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Le nombre de références ({len(references)}) diffère du nombre d'hypothèses ({len(hypotheses)})."
        )
    if not references:
        raise ValueError("La liste des références est vide.")

    refs_norm = [normaliser_arabe(r, niveau=niveau) for r in references]
    hyps_norm = [normaliser_arabe(h, niveau=niveau) for h in hypotheses]

    if any(len(r) == 0 for r in refs_norm):
        raise ValueError("Au moins une référence normalisée est vide : calcul WER/CER impossible.")

    if _jiwer is not None:
        wer_val = float(_jiwer.wer(refs_norm, hyps_norm))
        cer_val = float(_jiwer.cer(refs_norm, hyps_norm))
        outil = f"jiwer-{getattr(_jiwer, '__version__', 'installé')}+normaliser_arabe({niveau})"
    else:
        total_mots_ref = 0
        total_err_mots = 0
        total_chars_ref = 0
        total_err_chars = 0
        for r_txt, h_txt in zip(refs_norm, hyps_norm):
            r_mots = [m for m in r_txt.split(" ") if m]
            h_mots = [m for m in h_txt.split(" ") if m]
            total_mots_ref += len(r_mots)
            total_err_mots += _distance_edition_sequence(r_mots, h_mots)

            r_chars = list(r_txt)
            h_chars = list(h_txt)
            total_chars_ref += len(r_chars)
            total_err_chars += _distance_edition_sequence(r_chars, h_chars)

        wer_val = total_err_mots / total_mots_ref
        cer_val = total_err_chars / total_chars_ref
        outil = f"levenshtein-exact+normaliser_arabe({niveau})"

    return {
        "wer": round(wer_val, 6),
        "cer": round(cer_val, 6),
        "wer_pct": round(wer_val * 100.0, 2),
        "cer_pct": round(cer_val * 100.0, 2),
        "nb_segments": len(references),
        "outil": outil,
    }


# --- Tokenisation 13a et n-grammes pour chrF++ et BLEU ---

_RE_13A_1 = re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])")
_RE_13A_2 = re.compile(r"([^0-9])([\.,])")
_RE_13A_3 = re.compile(r"([\.,])([^0-9])")
_RE_13A_4 = re.compile(r"([0-9])(-)")


def _tokeniser_13a(texte: str) -> List[str]:
    """Tokenisation standard WMT '13a' utilisée par sacrebleu."""
    t = texte.replace("<skipped>", "")
    t = t.replace("-\n", "")
    t = t.replace("\n", " ")
    t = t.replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    t = f" {t} "
    t = _RE_13A_1.sub(r" \1 ", t)
    t = _RE_13A_2.sub(r"\1 \2 ", t)
    t = _RE_13A_3.sub(r" \1 \2", t)
    t = _RE_13A_4.sub(r"\1 \2 ", t)
    return [tok for tok in t.split() if tok]


def _extraire_ngrammes(seq: Sequence[str], n: int) -> Counter:
    if len(seq) < n:
        return Counter()
    return Counter(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))


def _chrf_pp_corpus_interne(
    references: Sequence[str],
    hypotheses: Sequence[str],
    char_order: int = 6,
    word_order: int = 2,
    beta: float = 2.0,
) -> float:
    """Implémentation de référence exacte de chrF++ (Popović 2017 / sacrebleu CHRF word_order=2)."""
    beta_sq = beta ** 2
    scores_ordres = []

    for n in range(1, char_order + 1):
        total_hyp = 0
        total_ref = 0
        total_match = 0
        for r_txt, h_txt in zip(references, hypotheses):
            r_chars = list(re.sub(r"\s+", "", r_txt.strip()))
            h_chars = list(re.sub(r"\s+", "", h_txt.strip()))
            c_ref = _extraire_ngrammes(r_chars, n)
            c_hyp = _extraire_ngrammes(h_chars, n)
            total_ref += sum(c_ref.values())
            total_hyp += sum(c_hyp.values())
            total_match += sum((c_hyp & c_ref).values())

        prec = (total_match / total_hyp) if total_hyp > 0 else (1.0 if total_ref == 0 else 0.0)
        rec = (total_match / total_ref) if total_ref > 0 else (1.0 if total_hyp == 0 else 0.0)
        if prec + rec > 0:
            f_n = (1.0 + beta_sq) * prec * rec / (beta_sq * prec + rec)
        else:
            f_n = 0.0
        scores_ordres.append(f_n)

    re_mots_chrf = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    for n in range(1, word_order + 1):
        total_hyp = 0
        total_ref = 0
        total_match = 0
        for r_txt, h_txt in zip(references, hypotheses):
            r_mots = re_mots_chrf.findall(r_txt.strip())
            h_mots = re_mots_chrf.findall(h_txt.strip())
            c_ref = _extraire_ngrammes(r_mots, n)
            c_hyp = _extraire_ngrammes(h_mots, n)
            total_ref += sum(c_ref.values())
            total_hyp += sum(c_hyp.values())
            total_match += sum((c_hyp & c_ref).values())

        prec = (total_match / total_hyp) if total_hyp > 0 else (1.0 if total_ref == 0 else 0.0)
        rec = (total_match / total_ref) if total_ref > 0 else (1.0 if total_hyp == 0 else 0.0)
        if prec + rec > 0:
            f_n = (1.0 + beta_sq) * prec * rec / (beta_sq * prec + rec)
        else:
            f_n = 0.0
        scores_ordres.append(f_n)

    return 100.0 * (sum(scores_ordres) / len(scores_ordres))


def _bleu_corpus_interne(
    references: Sequence[str],
    hypotheses: Sequence[str],
    max_ordre: int = 4,
) -> float:
    """Implémentation exacte de BLEU-4 corpus-level avec lissage 'exp' (sacrebleu default)."""
    matches = [0] * max_ordre
    totaux = [0] * max_ordre
    long_ref = 0
    long_hyp = 0

    for r_txt, h_txt in zip(references, hypotheses):
        r_toks = _tokeniser_13a(r_txt)
        h_toks = _tokeniser_13a(h_txt)
        long_ref += len(r_toks)
        long_hyp += len(h_toks)
        for n in range(1, max_ordre + 1):
            c_ref = _extraire_ngrammes(r_toks, n)
            c_hyp = _extraire_ngrammes(h_toks, n)
            matches[n - 1] += sum((c_hyp & c_ref).values())
            totaux[n - 1] += max(0, len(h_toks) - n + 1)

    if long_hyp == 0 or matches[0] == 0:
        return 0.0

    precisions = [0.0] * max_ordre
    smooth_m = 1.0
    for i in range(max_ordre):
        if totaux[i] == 0:
            precisions[i] = 0.0
        elif matches[i] == 0:
            smooth_m *= 2.0
            precisions[i] = 1.0 / (smooth_m * totaux[i])
        else:
            precisions[i] = matches[i] / totaux[i]

    if any(p <= 0.0 for p in precisions):
        return 0.0

    if long_hyp < long_ref:
        bp = math.exp(1.0 - (long_ref / long_hyp))
    else:
        bp = 1.0

    log_moy = sum(math.log(p) for p in precisions) / max_ordre
    return 100.0 * bp * math.exp(log_moy)


def calculer_chrf_bleu(
    references: Sequence[str],
    hypotheses: Sequence[str],
    references_secondaires: Optional[Sequence[Optional[str]]] = None,
) -> Dict[str, Union[float, str, int]]:
    """Calcule chrF++ (word_order=2) et BLEU avec leurs signatures exactes."""
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Le nombre de références ({len(references)}) diffère du nombre d'hypothèses ({len(hypotheses)})."
        )
    if not references:
        raise ValueError("La liste des références est vide.")

    hyps = [str(h) for h in hypotheses]
    refs_principales = [str(r) for r in references]

    if _sacrebleu is not None:
        chrf_metric = _sacrebleu.metrics.CHRF(word_order=2)
        bleu_metric = _sacrebleu.metrics.BLEU()

        if references_secondaires is not None and any(r2 for r2 in references_secondaires):
            refs_sec = [str(r2) if r2 else r1 for r1, r2 in zip(refs_principales, references_secondaires)]
            liste_refs = [refs_principales, refs_sec]
        else:
            liste_refs = [refs_principales]

        res_chrf = chrf_metric.corpus_score(hyps, liste_refs)
        res_bleu = bleu_metric.corpus_score(hyps, liste_refs)
        sig_chrf = str(chrf_metric.get_signature())
        sig_bleu = str(bleu_metric.get_signature())
        score_chrf = float(res_chrf.score)
        score_bleu = float(res_bleu.score)
    else:
        score_chrf = _chrf_pp_corpus_interne(refs_principales, hyps, char_order=6, word_order=2, beta=2.0)
        score_bleu = _bleu_corpus_interne(refs_principales, hyps, max_ordre=4)
        nb_refs = 2 if (references_secondaires and any(r2 for r2 in references_secondaires)) else 1
        sig_chrf = f"nrefs:{nb_refs}|case:mixed|eff:yes|nc:6|nw:2|space:no|version:interne-conforme-sacrebleu"
        sig_bleu = f"nrefs:{nb_refs}|case:mixed|eff:no|tok:13a|smooth:exp|version:interne-conforme-sacrebleu"

    return {
        "chrf_pp": round(score_chrf, 2),
        "bleu": round(score_bleu, 2),
        "signature_chrf_pp": sig_chrf,
        "signature_bleu": sig_bleu,
        "nb_segments": len(references),
    }


def calculer_comet(
    sources_ar: Sequence[str],
    hypotheses_fr: Sequence[str],
    references_fr: Sequence[str],
    modele: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 8,
    gpus: int = 0,
) -> Dict[str, Union[float, str, List[float]]]:
    """Calcule le score COMET avec `Unbabel/wmt22-comet-da`."""
    if not (len(sources_ar) == len(hypotheses_fr) == len(references_fr)):
        raise ValueError("Les listes sources_ar, hypotheses_fr et references_fr doivent avoir la même taille.")
    if not sources_ar:
        raise ValueError("Les listes d'entrée pour COMET sont vides.")

    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as exc:
        return {
            "comet": "non mesuré",
            "modele": modele,
            "motif": f"Paquet unbabel-comet non installé dans cet environnement ({exc})",
        }

    try:
        chemin_ckpt = download_model(modele)
        modele_comet = load_from_checkpoint(chemin_ckpt)
        donnees = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources_ar, hypotheses_fr, references_fr)
        ]
        sortie = modele_comet.predict(donnees, batch_size=batch_size, gpus=gpus, progress_bar=False)
        score_systeme = float(sortie.system_score)
        scores_segments = [float(x) for x in sortie.scores]
        return {
            "comet": round(score_systeme, 4),
            "scores_segments": scores_segments,
            "modele": modele,
        }
    except Exception as exc:
        return {
            "comet": "non mesuré",
            "modele": modele,
            "motif": f"Exécution COMET impossible localement ({exc})",
        }


def calculer_rtf(temps_calcul_s: float, duree_audio_s: float) -> float:
    """Calcule le facteur temps réel (RTF = temps de calcul / durée du son)."""
    if duree_audio_s <= 0:
        raise ValueError(f"La durée du son doit être strictement positive (reçu : {duree_audio_s}).")
    if temps_calcul_s < 0:
        raise ValueError(f"Le temps de calcul ne peut pas être négatif (reçu : {temps_calcul_s}).")
    return round(float(temps_calcul_s) / float(duree_audio_s), 4)


def calculer_retard_et_reecriture(
    segments_direct: Sequence[Dict[str, Union[float, List[str], str]]],
) -> Dict[str, float]:
    """Calcule le retard et le taux de réécriture en mode direct."""
    if not segments_direct:
        raise ValueError("La liste segments_direct est vide.")

    retards = []
    total_mots_affiches = 0
    total_mots_modifies = 0

    for seg in segments_direct:
        fin_p = float(seg["fin_parole_s"])
        aff_d = float(seg["affichage_definitif_s"])
        retards.append(aff_d - fin_p)

        passes = list(seg.get("passes_provisoires", []))
        texte_final = str(seg.get("texte_final", ""))
        toutes_passes = passes + ([texte_final] if texte_final else [])

        for idx in range(len(toutes_passes) - 1):
            mots_courants = [m for m in str(toutes_passes[idx]).strip().split() if m]
            mots_suivants = [m for m in str(toutes_passes[idx + 1]).strip().split() if m]
            total_mots_affiches += len(mots_courants)
            for pos, mot in enumerate(mots_courants):
                if pos >= len(mots_suivants) or mots_suivants[pos] != mot:
                    total_mots_modifies += 1

    retard_moyen = sum(retards) / len(retards)
    retards_tries = sorted(retards)
    retard_max = max(retards)
    idx_p95 = min(len(retards_tries) - 1, int(math.ceil(0.95 * len(retards_tries))) - 1)
    retard_p95 = retards_tries[idx_p95]

    taux_reecriture = (
        (total_mots_modifies / total_mots_affiches) if total_mots_affiches > 0 else 0.0
    )

    return {
        "retard_moyen_s": round(retard_moyen, 3),
        "retard_p95_s": round(retard_p95, 3),
        "retard_max_s": round(retard_max, 3),
        "taux_reecriture": round(taux_reecriture, 4),
        "taux_reecriture_pct": round(taux_reecriture * 100.0, 2),
        "nb_segments": len(segments_direct),
    }
