"""Gestion du tableau unique resultats/resultats.csv et CLI d'évaluation.

Colonnes du fichier CSV :
`date,systeme,version_modele,jeu,mesure,valeur,intervalle,commit_code,configuration`
Chaque ligne est produite par un script, jamais saisie à la main.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Union

DOSSIER_MESURES = Path(__file__).resolve().parent
if str(DOSSIER_MESURES) not in sys.path:
    sys.path.insert(0, str(DOSSIER_MESURES))

from reechantillonnage import comparer_systemes_bootstrap  # noqa: E402
from scores import (  # noqa: E402
    calculer_chrf_bleu,
    calculer_comet,
    calculer_rtf,
    calculer_wer_cer,
)

RACINE_PROJET = Path(__file__).resolve().parents[2]
CHEMIN_RESULTATS_DEFAUT = RACINE_PROJET / "resultats" / "resultats.csv"

COLONNES_RESULTATS = [
    "date",
    "systeme",
    "version_modele",
    "jeu",
    "mesure",
    "valeur",
    "intervalle",
    "commit_code",
    "configuration",
]


_COMMIT_GIT_CACHE: Optional[str] = None


def obtenir_commit_git() -> str:
    """Récupère le hash court du commit Git courant (mis en cache pour éviter les appels répétés)."""
    global _COMMIT_GIT_CACHE
    if _COMMIT_GIT_CACHE is not None:
        return _COMMIT_GIT_CACHE
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(RACINE_PROJET),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        _COMMIT_GIT_CACHE = res.stdout.strip() or "inconnu"
    except Exception:
        _COMMIT_GIT_CACHE = "inconnu"
    return _COMMIT_GIT_CACHE


def enregistrer_resultat(
    systeme: str,
    version_modele: str,
    jeu: str,
    mesure: str,
    valeur: Union[float, int, str],
    intervalle: str = "non applicable",
    configuration: str = "defaut",
    chemin_csv: Optional[Union[str, Path]] = None,
    date_iso: Optional[str] = None,
    commit_code: Optional[str] = None,
) -> Dict[str, str]:
    """Ajoute une ligne de résultat dans `resultats/resultats.csv`."""
    cible = Path(chemin_csv) if chemin_csv is not None else CHEMIN_RESULTATS_DEFAUT
    cible.parent.mkdir(parents=True, exist_ok=True)

    date_str = date_iso or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    commit_str = commit_code or obtenir_commit_git()

    ligne = {
        "date": str(date_str),
        "systeme": str(systeme),
        "version_modele": str(version_modele),
        "jeu": str(jeu),
        "mesure": str(mesure),
        "valeur": str(valeur),
        "intervalle": str(intervalle),
        "commit_code": str(commit_str),
        "configuration": str(configuration),
    }

    ecrire_entete = not cible.exists() or cible.stat().st_size == 0
    with open(cible, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLONNES_RESULTATS)
        if ecrire_entete:
            writer.writeheader()
        writer.writerow(ligne)

    return ligne


def lire_resultats(chemin_csv: Optional[Union[str, Path]] = None) -> List[Dict[str, str]]:
    """Lit toutes les lignes de `resultats/resultats.csv`."""
    cible = Path(chemin_csv) if chemin_csv is not None else CHEMIN_RESULTATS_DEFAUT
    if not cible.exists():
        return []
    with open(cible, mode="r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def evaluer_sorties_systeme(
    fiches_reference: Sequence[Dict],
    sorties_systeme: Sequence[Dict],
    systeme: str,
    version_modele: str,
    jeu: str,
    configuration: str = "defaut",
    chemin_csv: Optional[Union[str, Path]] = None,
    evaluer_par_type: bool = True,
) -> List[Dict[str, str]]:
    """Évalue les sorties d'un système sur un jeu de fiches et écrit dans resultats.csv."""
    if len(fiches_reference) != len(sorties_systeme):
        raise ValueError(
            f"Nombre de fiches de référence ({len(fiches_reference)}) != sorties ({len(sorties_systeme)})."
        )

    lignes_ecrites: List[Dict[str, str]] = []

    def _evaluer_sous_ensemble(refs_sub: Sequence[Dict], hyps_sub: Sequence[Dict], nom_jeu: str):
        ar_refs = [r["ar"] for r in refs_sub if "ar" in r and r["ar"]]
        ar_hyps = [h.get("ar_ecoute", h.get("ar", "")) for h in hyps_sub]
        if len(ar_refs) == len(refs_sub) and all(ar_hyps):
            res_asr = calculer_wer_cer(ar_refs, ar_hyps)
            lignes_ecrites.append(
                enregistrer_resultat(
                    systeme=systeme,
                    version_modele=version_modele,
                    jeu=nom_jeu,
                    mesure="WER (%)",
                    valeur=res_asr["wer_pct"],
                    intervalle=f"outil:{res_asr['outil']}",
                    configuration=configuration,
                    chemin_csv=chemin_csv,
                )
            )
            lignes_ecrites.append(
                enregistrer_resultat(
                    systeme=systeme,
                    version_modele=version_modele,
                    jeu=nom_jeu,
                    mesure="CER (%)",
                    valeur=res_asr["cer_pct"],
                    intervalle=f"outil:{res_asr['outil']}",
                    configuration=configuration,
                    chemin_csv=chemin_csv,
                )
            )

        fr_refs = [r["fr"] for r in refs_sub if "fr" in r and r["fr"]]
        fr_hyps = [h.get("fr_traduit", h.get("fr", "")) for h in hyps_sub]
        if len(fr_refs) == len(refs_sub) and any(fr_hyps):
            fr2_refs = [r.get("fr2") for r in refs_sub]
            res_mt = calculer_chrf_bleu(fr_refs, fr_hyps, references_secondaires=fr2_refs)
            lignes_ecrites.append(
                enregistrer_resultat(
                    systeme=systeme,
                    version_modele=version_modele,
                    jeu=nom_jeu,
                    mesure="chrF++",
                    valeur=res_mt["chrf_pp"],
                    intervalle=res_mt["signature_chrf_pp"],
                    configuration=configuration,
                    chemin_csv=chemin_csv,
                )
            )
            lignes_ecrites.append(
                enregistrer_resultat(
                    systeme=systeme,
                    version_modele=version_modele,
                    jeu=nom_jeu,
                    mesure="BLEU",
                    valeur=res_mt["bleu"],
                    intervalle=res_mt["signature_bleu"],
                    configuration=configuration,
                    chemin_csv=chemin_csv,
                )
            )
            if len(ar_refs) == len(refs_sub):
                res_comet = calculer_comet(ar_refs, fr_hyps, fr_refs)
                lignes_ecrites.append(
                    enregistrer_resultat(
                        systeme=systeme,
                        version_modele=version_modele,
                        jeu=nom_jeu,
                        mesure="COMET",
                        valeur=res_comet["comet"],
                        intervalle=res_comet.get("motif", res_comet.get("modele", "")),
                        configuration=configuration,
                        chemin_csv=chemin_csv,
                    )
                )

        durees = [float(r.get("duree_s", 0.0)) for r in refs_sub]
        temps_calc = [float(h.get("temps_calcul_s", -1.0)) for h in hyps_sub]
        if all(d > 0 for d in durees) and all(t >= 0 for t in temps_calc):
            rtf_global = calculer_rtf(sum(temps_calc), sum(durees))
            lignes_ecrites.append(
                enregistrer_resultat(
                    systeme=systeme,
                    version_modele=version_modele,
                    jeu=nom_jeu,
                    mesure="RTF",
                    valeur=rtf_global,
                    intervalle="temps_calcul/duree_son",
                    configuration=configuration,
                    chemin_csv=chemin_csv,
                )
            )

    _evaluer_sous_ensemble(fiches_reference, sorties_systeme, jeu)

    if evaluer_par_type:
        for type_extrait in ("plateau", "reportage", "debat"):
            paires = [
                (r, h)
                for r, h in zip(fiches_reference, sorties_systeme)
                if r.get("type") == type_extrait
            ]
            if paires:
                r_sub, h_sub = zip(*paires)
                _evaluer_sous_ensemble(list(r_sub), list(h_sub), f"{jeu}:{type_extrait}")

    return lignes_ecrites


def executer_mode_essai(chemin_csv: Optional[Union[str, Path]] = None) -> Dict[str, Union[int, float, str]]:
    """Exécute un essai complet hors-ligne sur un jeu factice contrôlé."""
    refs = [
        {
            "id": "essai_01",
            "ar": "أعلن الرئيس الفرنسي اليوم عن خطة اقتصادية جديدة.",
            "fr": "Le président français a annoncé aujourd'hui un nouveau plan économique.",
            "duree_s": 5.0,
            "type": "plateau",
        },
        {
            "id": "essai_02",
            "ar": "تجري المفاوضات في مدينة جنيف بحضور الوفود الدولية.",
            "fr": "Les négociations se déroulent à Genève en présence des délégations internationales.",
            "duree_s": 5.0,
            "type": "reportage",
        },
    ]
    hyps = [
        {
            "id": "essai_01",
            "ar_ecoute": "اعلن الرئيس الفرنسي اليوم عن خطه اقتصاديه جديده",
            "fr_traduit": "Le président français a annoncé aujourd'hui un nouveau plan économique.",
            "temps_calcul_s": 1.25,
        },
        {
            "id": "essai_02",
            "ar_ecoute": "تجري المفاوضات في مدينه جنيف بحضور الوفود الدوليه",
            "fr_traduit": "Les négociations ont lieu à Genève avec les délégations internationales.",
            "temps_calcul_s": 1.25,
        },
    ]
    cible = Path(chemin_csv) if chemin_csv else (RACINE_PROJET / "resultats" / "resultats_essai.csv")
    lignes = evaluer_sorties_systeme(
        fiches_reference=refs,
        sorties_systeme=hyps,
        systeme="ESSAI_FACTICE",
        version_modele="factice-v1",
        jeu="essai_factice",
        configuration="mode_essai",
        chemin_csv=cible,
        evaluer_par_type=False,
    )
    boot = comparer_systemes_bootstrap(
        references=[r["fr"] for r in refs],
        hypotheses_a=[h["fr_traduit"] for h in hyps],
        hypotheses_b=[h["fr_traduit"] for h in hyps],
        metrique="chrf++",
        n_tirages=100,
    )
    return {
        "nb_lignes_ecrites": len(lignes),
        "chemin_csv": str(cible),
        "bootstrap_auto_contient_zero": boot["contient_zero"],
    }


def construire_parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Évalue les sorties d'un système (WER, CER, chrF++, BLEU, COMET, RTF) et met à jour resultats/resultats.csv."
    )
    parser.add_argument(
        "--essai",
        action="store_true",
        help="Lance une vérification rapide sur un mini-jeu factice sans toucher au vrai tableau.",
    )
    parser.add_argument("--references", type=str, help="Chemin du fichier JSONL de référence (R, T ou FLEURS).")
    parser.add_argument("--sorties", type=str, help="Chemin du fichier JSONL des sorties du système.")
    parser.add_argument("--systeme", type=str, default="S0", help="Code du système (S0, S1, S2, S3, R1, R2).")
    parser.add_argument("--version-modele", type=str, default="v1", help="Identifiant ou version du modèle.")
    parser.add_argument("--jeu", type=str, default="R", help="Nom du jeu évalué (R, T, fleurs_ar_fr).")
    parser.add_argument("--config", type=str, default="defaut", help="Fichier ou nom de configuration.")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(CHEMIN_RESULTATS_DEFAUT),
        help="Chemin du fichier CSV de résultats (défaut: resultats/resultats.csv).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = construire_parseur()
    args = parser.parse_args(argv)

    if args.essai:
        res = executer_mode_essai()
        print(f"[ESSAI MESURES OK] {res['nb_lignes_ecrites']} lignes écrites dans {res['chemin_csv']}")
        return 0

    if not args.references or not args.sorties:
        parser.error("Les arguments --references et --sorties sont requis hors mode --essai.")

    with open(args.references, "r", encoding="utf-8") as f_ref:
        refs = [json.loads(line) for line in f_ref if line.strip()]
    with open(args.sorties, "r", encoding="utf-8") as f_hyp:
        hyps = [json.loads(line) for line in f_hyp if line.strip()]

    lignes = evaluer_sorties_systeme(
        fiches_reference=refs,
        sorties_systeme=hyps,
        systeme=args.systeme,
        version_modele=args.version_modele,
        jeu=args.jeu,
        configuration=args.config,
        chemin_csv=args.csv,
    )
    print(f"[MESURES OK] {len(lignes)} lignes ajoutées dans {args.csv}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
