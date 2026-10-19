"""Évaluation comparative des systèmes, ventilation par type, grille humaine à l'aveugle et analyse d'erreurs."""

import argparse
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

DOSSIER_MESURES = Path(__file__).resolve().parent
RACINE_PROJET = DOSSIER_MESURES.parents[1]
DOSSIER_DONNEES = RACINE_PROJET / "code" / "donnees"

for p in (DOSSIER_MESURES, DOSSIER_DONNEES):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from normalisation import normaliser_arabe  # noqa: E402
from reechantillonnage import comparer_systemes_bootstrap  # noqa: E402
from tableau_resultats import enregistrer_resultat, evaluer_sorties_systeme  # noqa: E402

CATEGORIES_ERREURS = (
    "nom_propre",
    "chiffre",
    "expression_figee",
    "dialecte",
    "erreur_ecoute_propagee",
    "omission",
    "ajout",
)


def preparer_notation_humaine_aveugle(
    fiches_reference: Sequence[Dict[str, Any]],
    sorties_s2: Sequence[Dict[str, Any]],
    sorties_s3: Sequence[Dict[str, Any]],
    dossier_sortie: Union[str, Path],
    nb_extraits: int = 30,
    graine: int = 42,
) -> Dict[str, Any]:
    """Prépare les 30 extraits anonymisés en aveugle (S2 vs S3) pour la notation humaine de 1 à 5.

    Génère :
    - `grille_notation_humaine_aveugle.csv` : grille publique sans mention de S2 ni S3 (colonnes A et B
      mélangées aléatoirement par extrait, critères : fidélité 1-5, fluidité 1-5, noms propres 1-5).
    - `cle_secrete_notation_humaine.json` : correspondance `{id_extrait: {"A": "S2"|"S3", "B": "S3"|"S2"}}`.
    """
    if len(fiches_reference) != len(sorties_s2) or len(fiches_reference) != len(sorties_s3):
        raise ValueError("Les listes fiches_reference, sorties_s2 et sorties_s3 doivent avoir la même taille.")

    cible_dir = Path(dossier_sortie)
    cible_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(graine)
    indices = list(range(len(fiches_reference)))
    if len(indices) > nb_extraits:
        indices = sorted(rng.sample(indices, k=nb_extraits))

    chemin_grille = cible_dir / "grille_notation_humaine_aveugle.csv"
    chemin_cle = cible_dir / "cle_secrete_notation_humaine.json"

    colonnes = [
        "id_extrait",
        "type_extrait",
        "transcription_arabe_reference",
        "traduction_francaise_reference",
        "sortie_systeme_A",
        "sortie_systeme_B",
        "note_A_fidelite_1_5",
        "note_A_fluidite_1_5",
        "note_A_noms_propres_1_5",
        "note_B_fidelite_1_5",
        "note_B_fluidite_1_5",
        "note_B_noms_propres_1_5",
    ]

    cle_secrete: Dict[str, Dict[str, str]] = {}
    with open(chemin_grille, mode="w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=colonnes)
        writer.writeheader()

        for i in indices:
            ref = fiches_reference[i]
            s2 = sorties_s2[i]
            s3 = sorties_s3[i]
            extrait_id = str(ref.get("id", f"extrait_{i+1:02d}"))
            txt_s2 = str(s2.get("fr_traduit", s2.get("fr", ""))).strip()
            txt_s3 = str(s3.get("fr_traduit", s3.get("fr", ""))).strip()

            if rng.random() < 0.5:
                cand_a, cand_b = ("S2", txt_s2), ("S3", txt_s3)
            else:
                cand_a, cand_b = ("S3", txt_s3), ("S2", txt_s2)

            cle_secrete[extrait_id] = {"A": cand_a[0], "B": cand_b[0]}
            writer.writerow(
                {
                    "id_extrait": extrait_id,
                    "type_extrait": ref.get("type", "plateau"),
                    "transcription_arabe_reference": ref.get("ar", ""),
                    "traduction_francaise_reference": ref.get("fr", ""),
                    "sortie_systeme_A": cand_a[1],
                    "sortie_systeme_B": cand_b[1],
                    "note_A_fidelite_1_5": "",
                    "note_A_fluidite_1_5": "",
                    "note_A_noms_propres_1_5": "",
                    "note_B_fidelite_1_5": "",
                    "note_B_fluidite_1_5": "",
                    "note_B_noms_propres_1_5": "",
                }
            )

    with open(chemin_cle, mode="w", encoding="utf-8") as f_json:
        json.dump(cle_secrete, f_json, ensure_ascii=False, indent=2)

    return {
        "nb_extraits_anonymises": len(indices),
        "chemin_grille_csv": str(chemin_grille),
        "chemin_cle_json": str(chemin_cle),
    }


def diagnostiquer_bloc_fautif_s2(
    ar_reference: str,
    ar_ecoute_s2: str,
    fr_reference: str,
    fr_sortie_s2: str,
) -> str:
    """Détermine pour la cascade S2 si la faute provient du bloc d'écoute (ASR) ou du traducteur (LLM)."""
    ar_ref_norm = normaliser_arabe(ar_reference)
    ar_hyp_norm = normaliser_arabe(ar_ecoute_s2)
    mots_ref = set(ar_ref_norm.split())
    mots_hyp = set(ar_hyp_norm.split())
    recouvrement_ar = len(mots_ref & mots_hyp) / float(max(1, len(mots_ref)))

    if recouvrement_ar < 0.80:
        return "bloc_ecoute_asr"
    if fr_sortie_s2.strip().lower() != fr_reference.strip().lower():
        return "bloc_traducteur_llm"
    return "aucun"


def classer_erreurs_systeme(
    exemples_annotes: Sequence[Dict[str, Any]],
    chemin_csv_sortie: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Classe au moins 20 erreurs selon les 7 catégories et identifie le bloc fautif pour S2."""
    lignes: List[Dict[str, Any]] = []
    compteurs = {cat: 0 for cat in CATEGORIES_ERREURS}
    blocs_s2 = {"bloc_ecoute_asr": 0, "bloc_traducteur_llm": 0}

    for idx, ex in enumerate(exemples_annotes, start=1):
        cat = str(ex.get("categorie", "omission"))
        if cat not in CATEGORIES_ERREURS:
            raise ValueError(f"Catégorie d'erreur inconnue : {cat}")
        compteurs[cat] += 1

        bloc = ex.get("bloc_fautif_s2") or diagnostiquer_bloc_fautif_s2(
            ar_reference=str(ex.get("ar_ref", "")),
            ar_ecoute_s2=str(ex.get("ar_ecoute_s2", "")),
            fr_reference=str(ex.get("fr_ref", "")),
            fr_sortie_s2=str(ex.get("fr_s2", "")),
        )
        if bloc in blocs_s2:
            blocs_s2[bloc] += 1

        lignes.append(
            {
                "id_erreur": ex.get("id", f"err_{idx:02d}"),
                "categorie": cat,
                "bloc_fautif_s2": bloc,
                "ar_ref": ex.get("ar_ref", ""),
                "ar_ecoute_s2": ex.get("ar_ecoute_s2", ""),
                "fr_ref": ex.get("fr_ref", ""),
                "fr_s2": ex.get("fr_s2", ""),
                "fr_s3": ex.get("fr_s3", ""),
                "commentaire": ex.get("commentaire", ""),
            }
        )

    if chemin_csv_sortie is not None:
        cible = Path(chemin_csv_sortie)
        cible.parent.mkdir(parents=True, exist_ok=True)
        with open(cible, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(lignes[0].keys()) if lignes else ["id_erreur", "categorie"])
            writer.writeheader()
            writer.writerows(lignes)

    return {
        "nb_erreurs_classees": len(lignes),
        "repartition_categories": compteurs,
        "repartition_blocs_s2": blocs_s2,
    }


def comparer_tous_les_systemes(
    fiches_reference: Sequence[Dict[str, Any]],
    sorties_par_systeme: Dict[str, Sequence[Dict[str, Any]]],
    jeu: str,
    dossier_sortie: Union[str, Path],
    autoriser_jeu_t: bool = False,
    n_tirages_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """Évalue tous les systèmes (globalement et par type plateau/reportage/débat) et calcule les intervalles bootstrap."""
    if jeu.upper() == "T" and not autoriser_jeu_t:
        raise PermissionError(
            "Le lancement sur le jeu T ne se fait qu'une seule fois après accord explicite (--autoriser-jeu-t)."
        )

    cible_dir = Path(dossier_sortie)
    cible_dir.mkdir(parents=True, exist_ok=True)
    chemin_csv = cible_dir / "resultats_comparaison.csv"

    for code_sys, hyps in sorties_par_systeme.items():
        # Pour R2, évaluation sur les 10 premiers extraits seulement
        if code_sys == "R2":
            refs_eval = fiches_reference[:10]
            hyps_eval = hyps[:10]
            nom_jeu = f"{jeu}_10_extraits"
        else:
            refs_eval = fiches_reference
            hyps_eval = hyps
            nom_jeu = jeu

        evaluer_sorties_systeme(
            fiches_reference=refs_eval,
            sorties_systeme=hyps_eval,
            systeme=code_sys,
            version_modele=f"{code_sys.lower()}-v1",
            jeu=nom_jeu,
            configuration=f"eval_{jeu.lower()}",
            chemin_csv=chemin_csv,
            evaluer_par_type=(code_sys != "R2"),
        )

    refs_fr = [str(r["fr"]) for r in fiches_reference]
    comparaisons_paires: Dict[str, Any] = {}
    paires_a_tester = [("S2", "S1"), ("S3", "S2"), ("S2", "R1")]

    for sys_a, sys_b in paires_a_tester:
        if sys_a in sorties_par_systeme and sys_b in sorties_par_systeme:
            hyps_a = [str(h.get("fr_traduit", h.get("fr", ""))) for h in sorties_par_systeme[sys_a]]
            hyps_b = [str(h.get("fr_traduit", h.get("fr", ""))) for h in sorties_par_systeme[sys_b]]
            boot = comparer_systemes_bootstrap(
                references=refs_fr,
                hypotheses_a=hyps_a,
                hypotheses_b=hyps_b,
                metrique="chrf++",
                n_tirages=n_tirages_bootstrap,
                graine=42,
            )
            cle = f"{sys_a}_vs_{sys_b}"
            comparaisons_paires[cle] = boot
            enregistrer_resultat(
                systeme=f"COMP:{cle}",
                version_modele=f"{sys_a}-{sys_b}",
                jeu=jeu,
                mesure="ecart_chrF++",
                valeur=boot["ecart_observe"],
                intervalle=f"{boot['intervalle_texte']} ({boot['conclusion']})",
                configuration="bootstrap_1000",
                chemin_csv=chemin_csv,
            )

    return {
        "jeu": jeu,
        "systemes_evalues": list(sorties_par_systeme.keys()),
        "comparaisons_bootstrap": comparaisons_paires,
        "chemin_csv": str(chemin_csv),
    }


def executer_essai_evaluation_finale(dossier_sortie: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Vérifie le pipeline d'évaluation finale, la grille de 30 extraits en aveugle et les 21 erreurs classées."""
    cible_dir = Path(dossier_sortie) if dossier_sortie else (RACINE_PROJET / "resultats" / "essai_eval_finale")
    cible_dir.mkdir(parents=True, exist_ok=True)

    types_cycle = ("plateau", "reportage", "debat")
    fiches = []
    s0, s1, s2, s3, r1, r2 = [], [], [], [], [], []

    for idx in range(30):
        t_ext = types_cycle[idx % 3]
        ref_fr = f"Le conseil économique numéro {idx + 1} a validé l'accord international à Genève."
        ref_ar = f"صادق المجلس الاقتصادي رقم {idx + 1} على الاتفاق الدولي في جنيف."
        fiches.append({"id": f"ext_{idx+1:02d}", "type": t_ext, "ar": ref_ar, "fr": ref_fr, "duree_s": 5.0})
        s0.append({"id": f"ext_{idx+1:02d}", "ar_ecoute": ref_ar, "fr_traduit": f"Conseil {idx + 1} valide accord.", "temps_calcul_s": 0.8})
        s1.append({"id": f"ext_{idx+1:02d}", "ar_ecoute": ref_ar, "fr_traduit": f"Le conseil numéro {idx + 1} a validé l'accord.", "temps_calcul_s": 1.1})
        s2.append({"id": f"ext_{idx+1:02d}", "ar_ecoute": ref_ar, "fr_traduit": ref_fr, "temps_calcul_s": 1.2})
        s3.append({"id": f"ext_{idx+1:02d}", "fr_traduit": f"Le conseil économique numéro {idx + 1} a approuvé l'accord à Genève.", "temps_calcul_s": 1.0})
        r1.append({"id": f"ext_{idx+1:02d}", "fr_traduit": f"Le conseil économique {idx + 1} a validé l'accord à Genève.", "temps_calcul_s": 1.4})
        if idx < 10:
            r2.append({"id": f"ext_{idx+1:02d}", "fr_traduit": f"Le conseil {idx + 1} valide l'accord à Genève.", "temps_calcul_s": 1.5})

    comp = comparer_tous_les_systemes(
        fiches_reference=fiches,
        sorties_par_systeme={"S0": s0, "S1": s1, "S2": s2, "S3": s3, "R1": r1, "R2": r2},
        jeu="essai_R",
        dossier_sortie=cible_dir,
        n_tirages_bootstrap=100,
    )
    grille = preparer_notation_humaine_aveugle(
        fiches_reference=fiches,
        sorties_s2=s2,
        sorties_s3=s3,
        dossier_sortie=cible_dir,
        nb_extraits=30,
    )

    erreurs_exemples = []
    for idx in range(21):
        cat = CATEGORIES_ERREURS[idx % len(CATEGORIES_ERREURS)]
        ar_ref = "أعلن البنك المركزي عن خطة جديدة في الرباط."
        ar_hyp = "أعلن البنك عن خطة" if cat == "erreur_ecoute_propagee" else ar_ref
        erreurs_exemples.append(
            {
                "id": f"err_{idx+1:02d}",
                "categorie": cat,
                "ar_ref": ar_ref,
                "ar_ecoute_s2": ar_hyp,
                "fr_ref": "La banque centrale a annoncé un nouveau plan à Rabat.",
                "fr_s2": "La banque a annoncé un plan.",
                "fr_s3": "La banque centrale annonce un plan.",
            }
        )
    taxonomie = classer_erreurs_systeme(erreurs_exemples, chemin_csv_sortie=cible_dir / "analyse_21_erreurs.csv")

    return {
        "comparaison": comp,
        "grille_aveugle": grille,
        "taxonomie_erreurs": taxonomie,
    }


def construire_parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Évaluation comparative finale (S0..S3, R1, R2), grille humaine en aveugle et analyse d'erreurs."
    )
    parser.add_argument("--essai", action="store_true", help="Exécute la vérification complète sur 30 extraits d'essai.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = construire_parseur()
    args = parser.parse_args(argv)
    if args.essai:
        res = executer_essai_evaluation_finale()
        boot_s2_s1 = res["comparaison"]["comparaisons_bootstrap"]["S2_vs_S1"]
        print(
            f"[ESSAI EVAL FINALE OK] S2 vs S1 : {boot_s2_s1['ecart_observe']:+.2f} pts "
            f"{boot_s2_s1['intervalle_texte']} ({boot_s2_s1['conclusion']}) | "
            f"Grille aveugle : {res['grille_aveugle']['nb_extraits_anonymises']} extraits | "
            f"Erreurs classées : {res['taxonomie_erreurs']['nb_erreurs_classees']}"
        )
        return 0
    parser.error("Utilisez --essai.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
