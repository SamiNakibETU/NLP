#!/usr/bin/env python3
"""Entraînement S3, temps 1 : projecteur seul (encodeur et LLM gelés), transcription arabe.

Étapes, dans l'ordre :

1. ``separer_validation`` : un tirage fixe de segments du train sert à la
   validation (perte et WER pendant l'entraînement) ; il n'est jamais appris.
   Le test (épisodes jamais vus) ne sert qu'à l'évaluation finale.
2. ``lots_par_longueur`` : lots de segments de longueurs voisines (moins de
   rembourrage, donc moins de calcul), dans un ordre aléatoire à chaque époque.
3. ``entrainer`` : AdamW, montée linéaire puis décroissance en cosinus, gradients
   écrêtés. Toutes les ``eval_toutes_les_etapes`` : perte et WER de validation,
   point de reprise, meilleur projecteur (plus petit WER) gardé à part.
   Arrêt si le WER ne baisse plus ``patience_arret`` fois de suite.
4. ``evaluer`` : WER et CER (``code/mesures/scores.py``) sur le test.

Sorties dans ``sortie.dossier`` (ignoré par git, car les hypothèses contiennent du texte MGB-2) :
``metriques.jsonl`` (une ligne par mesure), ``projecteur_meilleur.pt``,
``reprise.pt``, ``evaluation_test.json`` et ``hypotheses_test.jsonl``.
"""

import argparse
import json
import math
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import yaml

RACINE_PROJET = Path(__file__).resolve().parents[2]
for _dossier in (RACINE_PROJET / "code" / "projecteur", RACINE_PROJET / "code" / "mesures",
                 RACINE_PROJET / "code" / "donnees"):
    if str(_dossier) not in sys.path:
        sys.path.insert(0, str(_dossier))

from donnees_s3 import DatasetS3  # noqa: E402
from etapes_s3 import preparer_temps  # noqa: E402
from modele_s3 import SpeechLLMS3, limite_tokens, retirer_boucles  # noqa: E402
from scores import calculer_wer_cer  # noqa: E402

CHEMIN_CONFIG = RACINE_PROJET / "configs" / "s3.yaml"
PRECISIONS = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
CLES_SECTION = ("dossier_vecteurs", "learning_rate", "warmup_steps", "taille_lot", "micro_lot", "nb_epoques",
                "max_norm_gradient", "eval_toutes_les_etapes", "nb_segments_validation",
                "nb_segments_wer_validation", "patience_arret", "precision_llm", "taille_lot_generation",
                "max_nouveaux_tokens", "repetition_penalty", "frequence_vecteurs_hz", "tokens_par_seconde_max",
                "tokens_marge", "repetitions_boucle", "sortie")


def chemin_projet(valeur: str) -> Path:
    chemin = Path(valeur).expanduser()
    return chemin if chemin.is_absolute() else RACINE_PROJET / chemin


def charger_config(chemin: Optional[Path] = None, section: str = "temps_1_pilote") -> Dict[str, Any]:
    chemin = Path(chemin or CHEMIN_CONFIG)
    if not chemin.is_file():
        raise FileNotFoundError(f"configuration introuvable : {chemin}")
    config = yaml.safe_load(chemin.read_text(encoding="utf-8"))
    if section not in config:
        raise KeyError(f"{chemin} : section '{section}' absente")
    for cle in CLES_SECTION:
        if cle not in config[section]:
            raise KeyError(f"{chemin} : clé '{section}.{cle}' absente")
    if config[section]["precision_llm"] not in PRECISIONS:
        raise ValueError(f"precision_llm inconnue : {config[section]['precision_llm']}")
    return config


def choisir_appareil() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchroniser(appareil: torch.device) -> None:
    if appareil.type == "cuda":
        torch.cuda.synchronize()
    elif appareil.type == "mps":
        torch.mps.synchronize()


# ----------------------------------------------------------------------------
# Données
# ----------------------------------------------------------------------------

def separer_validation(nb: int, nb_validation: int, graine: int):
    """(indices d'entraînement, indices de validation), tirage fixe par la graine."""
    indices = list(range(nb))
    random.Random(f"{graine}-validation").shuffle(indices)
    return sorted(indices[nb_validation:]), sorted(indices[:nb_validation])


def lots_par_longueur(indices: Sequence[int], longueurs: Sequence[int], taille_lot: int,
                      generateur: random.Random, facteur_seau: int = 50) -> List[List[int]]:
    """Mélange, trie par longueur dans des seaux de ``facteur_seau`` lots, puis mélange les lots."""
    ordre = list(indices)
    generateur.shuffle(ordre)
    lots: List[List[int]] = []
    pas = taille_lot * facteur_seau
    for debut in range(0, len(ordre), pas):
        seau = sorted(ordre[debut: debut + pas], key=lambda i: longueurs[i])
        lots.extend(seau[i: i + taille_lot] for i in range(0, len(seau), taille_lot))
    generateur.shuffle(lots)
    return lots


def lire_lot(jeu: DatasetS3, lot: Sequence[int]):
    exemples = [jeu[i] for i in lot]
    return [v for v, _ in exemples], [t for _, t in exemples]


def taux_apprentissage(etape: int, total: int, warmup: int) -> float:
    """Facteur multiplicatif : montée linéaire sur ``warmup`` étapes, puis cosinus jusqu'à 0."""
    if etape < warmup:
        return (etape + 1) / warmup
    progression = (etape - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progression)))


# ----------------------------------------------------------------------------
# Mesures
# ----------------------------------------------------------------------------

@torch.no_grad()
def perte_moyenne(modele: SpeechLLMS3, jeu: DatasetS3, indices: Sequence[int], taille_lot: int) -> float:
    modele.eval()
    total, nb_tokens = 0.0, 0
    ordre = sorted(indices, key=lambda i: jeu.index[i]["nb_vecteurs"])
    for debut in range(0, len(ordre), taille_lot):
        vecteurs, textes = lire_lot(jeu, ordre[debut: debut + taille_lot])
        sortie = modele(vecteurs, "transcription", textes)
        n = sortie.logits.shape[0]  # une ligne de logits par token prédit
        total += float(sortie.loss) * n
        nb_tokens += n
    modele.train()
    return total / max(1, nb_tokens)


@torch.no_grad()
def transcrire(modele: SpeechLLMS3, jeu: DatasetS3, indices: Sequence[int], section: Dict[str, Any]) -> List[str]:
    modele.eval()
    hypotheses: Dict[int, str] = {}
    ordre = sorted(indices, key=lambda i: jeu.index[i]["nb_vecteurs"])
    for debut in range(0, len(ordre), section["taille_lot_generation"]):
        lot = ordre[debut: debut + section["taille_lot_generation"]]
        vecteurs, _ = lire_lot(jeu, lot)
        limites = [limite_tokens(v.shape[0], section["frequence_vecteurs_hz"], section["tokens_par_seconde_max"],
                                 section["tokens_marge"]) for v in vecteurs]
        textes = modele.generer(vecteurs, "transcription", max_nouveaux_tokens=section["max_nouveaux_tokens"],
                                limites_tokens=limites, repetition_penalty=section["repetition_penalty"])
        hypotheses.update(zip(lot, textes))
    modele.train()
    return [retirer_boucles(hypotheses[i].strip(), section["repetitions_boucle"]) for i in indices]


def wer_cer(jeu: DatasetS3, indices: Sequence[int], hypotheses: Sequence[str]) -> Dict[str, Any]:
    return calculer_wer_cer([jeu.index[i]["texte_ar"] for i in indices], list(hypotheses))


# ----------------------------------------------------------------------------
# Entraînement
# ----------------------------------------------------------------------------

def construire_modele(config: Dict[str, Any], section: Dict[str, Any], d_enc: int, appareil: torch.device,
                      llm=None, tokenizer=None) -> SpeechLLMS3:
    if llm is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        llm_id = config["briques"]["modele_llm_id"]
        tokenizer = AutoTokenizer.from_pretrained(llm_id)
        llm = AutoModelForCausalLM.from_pretrained(llm_id, dtype=PRECISIONS[section["precision_llm"]])
    modele = SpeechLLMS3.depuis_config(config, llm, tokenizer, d_enc)
    return modele.to(appareil)


def entrainer(config: Dict[str, Any], section_nom: str = "temps_1_pilote", modele: Optional[SpeechLLMS3] = None,
              journal=print) -> Dict[str, Any]:
    """Boucle du temps 1. ``taille_lot`` est le lot effectif, calculé en micro-lots de ``micro_lot``
    segments (accumulation de gradient). Critère du meilleur projecteur : le WER de validation,
    ou la perte de validation si ``nb_segments_wer_validation`` vaut 0. ``etapes_max`` (facultatif)
    coupe l'entraînement plus tôt (balayage du taux d'apprentissage)."""
    section = config[section_nom]
    graine = config["graine"]
    torch.manual_seed(graine)
    dossier = chemin_projet(section["sortie"])
    dossier.mkdir(parents=True, exist_ok=True)
    jeu = DatasetS3(chemin_projet(section["dossier_vecteurs"]) / "train")
    appareil = choisir_appareil() if modele is None else modele.device
    if modele is None:
        modele = construire_modele(config, section, jeu.d_enc, appareil)
    parametres = preparer_temps(modele, 1, config["temps_2"]["lora"])
    optimiseur = torch.optim.AdamW(parametres, lr=section["learning_rate"])
    longueurs = [ligne["nb_vecteurs"] for ligne in jeu.index]
    indices_train, indices_val = separer_validation(len(jeu), section["nb_segments_validation"], graine)
    indices_wer = indices_val[: section["nb_segments_wer_validation"]]
    micro_lot = section["micro_lot"]
    lots_par_epoque = math.ceil(len(indices_train) / section["taille_lot"])
    total = min(lots_par_epoque * section["nb_epoques"], section.get("etapes_max") or float("inf"))
    planificateur = torch.optim.lr_scheduler.LambdaLR(
        optimiseur, lambda e: taux_apprentissage(e, total, section["warmup_steps"]))

    etape, meilleur, sans_progres = 0, float("inf"), 0
    chemin_reprise = dossier / "reprise.pt"
    if chemin_reprise.is_file():
        etat = torch.load(chemin_reprise, map_location=appareil)
        modele.projecteur.load_state_dict(etat["projecteur"])
        optimiseur.load_state_dict(etat["optimiseur"])
        planificateur.load_state_dict(etat["planificateur"])
        etape, meilleur, sans_progres = etat["etape"], etat["meilleur"], etat["sans_progres"]
        journal(f"reprise à l'étape {etape}")
    elif (dossier / "metriques.jsonl").is_file():
        (dossier / "metriques.jsonl").unlink()

    def noter(ligne: Dict[str, Any]) -> None:
        with open(dossier / "metriques.jsonl", "a", encoding="utf-8") as fichier:
            fichier.write(json.dumps(ligne, ensure_ascii=False) + "\n")

    def mesurer() -> bool:
        """Mesure de validation ; renvoie True s'il faut s'arrêter."""
        nonlocal meilleur, sans_progres
        depart = time.perf_counter()
        perte_val = perte_moyenne(modele, jeu, indices_val, micro_lot)
        ligne = {"type": "validation", "etape": etape, "epoque": round(etape / lots_par_epoque, 3),
                 "perte": round(perte_val, 4)}
        critere = perte_val
        if indices_wer:
            scores = wer_cer(jeu, indices_wer, transcrire(modele, jeu, indices_wer, section))
            ligne.update(wer_pct=scores["wer_pct"], cer_pct=scores["cer_pct"])
            critere = scores["wer"]
        noter(ligne)
        if critere < meilleur:
            meilleur, sans_progres = critere, 0
            torch.save(modele.projecteur.state_dict(), dossier / "projecteur_meilleur.pt")
        else:
            sans_progres += 1
        torch.save({"projecteur": modele.projecteur.state_dict(), "optimiseur": optimiseur.state_dict(),
                    "planificateur": planificateur.state_dict(), "etape": etape, "meilleur": meilleur,
                    "sans_progres": sans_progres}, chemin_reprise)
        texte_wer = f", WER {ligne['wer_pct']:.1f} %, CER {ligne['cer_pct']:.1f} %" if indices_wer else ""
        journal(f"étape {etape} : perte val {perte_val:.3f}{texte_wer} ({time.perf_counter() - depart:.0f} s)")
        return sans_progres >= section["patience_arret"]

    if etape == 0:
        mesurer()
    modele.train()
    generateur = random.Random(f"{graine}-lots")
    arret, depart, segments_vus = etape >= total, time.perf_counter(), 0
    for epoque in range(section["nb_epoques"]):
        if arret:
            break
        lots = lots_par_longueur(indices_train, longueurs, section["taille_lot"], generateur)
        for numero, lot in enumerate(lots):
            if epoque * lots_par_epoque + numero < etape:
                continue  # déjà fait avant la reprise (même ordre : même graine)
            perte_lot = 0.0
            for debut in range(0, len(lot), micro_lot):
                vecteurs, textes = lire_lot(jeu, lot[debut: debut + micro_lot])
                sortie = modele(vecteurs, "transcription", textes)
                part = len(vecteurs) / len(lot)
                (sortie.loss * part).backward()
                perte_lot += float(sortie.loss.detach()) * part
            norme = torch.nn.utils.clip_grad_norm_(parametres, section["max_norm_gradient"])
            optimiseur.step()
            planificateur.step()
            optimiseur.zero_grad(set_to_none=True)
            etape += 1
            segments_vus += len(lot)
            if etape % 10 == 0:
                noter({"type": "train", "etape": etape, "epoque": round(etape / lots_par_epoque, 3),
                       "perte": round(perte_lot, 4), "norme_gradient": round(float(norme), 3),
                       "lr": planificateur.get_last_lr()[0],
                       "segments_par_s": round(segments_vus / (time.perf_counter() - depart), 2)})
            if etape % section["eval_toutes_les_etapes"] == 0 or etape == total:
                if mesurer():
                    journal(f"arrêt : pas de progrès depuis {section['patience_arret']} mesures")
                    arret = True
            if etape >= total:
                arret = True
            if arret:
                break
    return {"etapes": etape, "meilleur_critere": round(meilleur, 4), "critere": "wer" if indices_wer else "perte",
            "dossier": str(dossier), "indices_validation": indices_val}


def balayer_taux(config: Dict[str, Any], valeurs: Sequence[float], etapes: int,
                 section_nom: str = "temps_1_pilote", modele: Optional[SpeechLLMS3] = None,
                 journal=print) -> List[Dict[str, Any]]:
    """Courts entraînements (``etapes`` pas, perte de validation seule) pour chaque taux d'apprentissage.

    Le LLM est chargé une fois ; le projecteur repart de la même initialisation à chaque essai.
    Sorties dans ``<sortie>/balayage/lr_<valeur>/``.
    """
    from projecteur_s3 import ProjecteurS3

    section = config[section_nom]
    if modele is None:
        d_enc = DatasetS3(chemin_projet(section["dossier_vecteurs"]) / "train").d_enc
        modele = construire_modele(config, section, d_enc, choisir_appareil())
    resultats = []
    for valeur in valeurs:
        essai = json.loads(json.dumps(config))
        essai[section_nom].update(learning_rate=valeur, etapes_max=etapes, nb_segments_wer_validation=0,
                                  eval_toutes_les_etapes=etapes, patience_arret=10 ** 6,
                                  warmup_steps=min(section["warmup_steps"], max(1, etapes // 5)),
                                  sortie=str(chemin_projet(section["sortie"]) / "balayage" / f"lr_{valeur:g}"))
        torch.manual_seed(config["graine"])
        ancien = modele.projecteur
        modele.projecteur = ProjecteurS3(ancien.norm.normalized_shape[0], ancien.mlp[-1].out_features,
                                         ancien.nb_reductions).to(modele.device)
        journal(f"taux {valeur:g}")
        resultat = entrainer(essai, section_nom, modele, journal)
        mesures = [l for l in lire_metriques(resultat["dossier"]) if l["type"] == "validation"]
        resultats.append({"learning_rate": valeur, "perte_validation": mesures[-1]["perte"],
                          "dossier": resultat["dossier"]})
    return resultats


def evaluer(config: Dict[str, Any], section_nom: str = "temps_1_pilote", modele: Optional[SpeechLLMS3] = None,
            partie: str = "test", journal=print) -> Dict[str, Any]:
    """WER et CER du meilleur projecteur sur ``partie`` (test : épisodes jamais vus)."""
    section = config[section_nom]
    dossier = chemin_projet(section["sortie"])
    jeu = DatasetS3(chemin_projet(section["dossier_vecteurs"]) / partie)
    chemin = dossier / "projecteur_meilleur.pt"
    if not chemin.is_file():
        raise FileNotFoundError(f"projecteur introuvable : {chemin} (lancer l'entraînement)")
    if modele is None:
        modele = construire_modele(config, section, jeu.d_enc, choisir_appareil())
    modele.projecteur.load_state_dict(torch.load(chemin, map_location=modele.device))
    indices = list(range(len(jeu)))
    depart = time.perf_counter()
    hypotheses = transcrire(modele, jeu, indices, section)
    synchroniser(modele.device)
    duree_calcul = time.perf_counter() - depart
    scores = wer_cer(jeu, indices, hypotheses)
    duree_audio = sum(ligne["duree_s"] for ligne in jeu.index)
    resultat = dict(scores, partie=partie, heures_audio=round(duree_audio / 3600, 3),
                    secondes_calcul=round(duree_calcul, 1), rtf=round(duree_calcul / duree_audio, 4))
    with open(dossier / f"evaluation_{partie}.json", "w", encoding="utf-8") as fichier:
        json.dump(resultat, fichier, ensure_ascii=False, indent=2)
    with open(dossier / f"hypotheses_{partie}.jsonl", "w", encoding="utf-8") as fichier:
        for i, hypothese in zip(indices, hypotheses):
            fichier.write(json.dumps({"id": jeu.index[i]["id"], "reference": jeu.index[i]["texte_ar"],
                                      "hypothese": hypothese}, ensure_ascii=False) + "\n")
    journal(f"{partie} : WER {scores['wer_pct']} %, CER {scores['cer_pct']} % sur {len(indices)} segments")
    return resultat


def lire_metriques(dossier: Path) -> List[Dict[str, Any]]:
    chemin = Path(dossier) / "metriques.jsonl"
    if not chemin.is_file():
        return []
    with open(chemin, "r", encoding="utf-8") as fichier:
        return [json.loads(ligne) for ligne in fichier if ligne.strip()]


# ----------------------------------------------------------------------------
# Mode essai et point d'entrée
# ----------------------------------------------------------------------------

def preparer_essai(config: Dict[str, Any], section_nom: str, dossier: Path):
    """Index factices (train et test) et petits modèles, sans réseau ni GPU."""
    import numpy as np

    from factices_s3 import D_ENC, TokenizerFactice, petit_llm, vecteurs_factices

    for partie, nb in (("train", 40), ("test", 6)):
        sous_dossier = dossier / "vecteurs" / partie
        sous_dossier.mkdir(parents=True, exist_ok=True)
        longueurs = [10 + (7 * k) % 30 for k in range(nb)]
        vecteurs = vecteurs_factices(longueurs, graine=nb)
        np.save(sous_dossier / "vecteurs_00000.npy", np.concatenate([v.numpy() for v in vecteurs]).astype(np.float16))
        debut = 0
        with open(sous_dossier / "index.jsonl", "w", encoding="utf-8") as fichier:
            for k, n in enumerate(longueurs):
                fichier.write(json.dumps({"id": f"{partie}_{k}", "texte_ar": "نص قصير", "duree_s": n / 12.5,
                                          "fichier_npy": "vecteurs_00000.npy", "ligne_debut": debut, "nb_vecteurs": n,
                                          "d_enc": D_ENC, "encodeur": "essai"}, ensure_ascii=False) + "\n")
                debut += n
    config[section_nom].update(dossier_vecteurs=str(dossier / "vecteurs"), sortie=str(dossier / "sortie"),
                               taille_lot=8, micro_lot=4, nb_epoques=2, warmup_steps=2, eval_toutes_les_etapes=4,
                               nb_segments_validation=8, nb_segments_wer_validation=4, taille_lot_generation=4,
                               max_nouveaux_tokens=4, patience_arret=10)
    torch.manual_seed(0)
    return SpeechLLMS3.depuis_config(config, petit_llm(), TokenizerFactice(), D_ENC)


def construire_parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S3 temps 1 : entraînement du projecteur (transcription arabe).")
    parser.add_argument("--config", type=str, default=str(CHEMIN_CONFIG), help="Config YAML (s3.yaml).")
    parser.add_argument("--section", type=str, default="temps_1_pilote", help="Section des réglages dans la config.")
    parser.add_argument("--essai", action="store_true", help="Petits modèles et vecteurs factices, sans réseau.")
    parser.add_argument("--sans-evaluation", action="store_true", help="Ne pas évaluer sur le test à la fin.")
    return parser


def main(arguments: Optional[List[str]] = None) -> Dict[str, Any]:
    args = construire_parseur().parse_args(arguments)
    config = charger_config(args.config, args.section)
    modele = None
    if args.essai:
        modele = preparer_essai(config, args.section, Path(tempfile.mkdtemp(prefix="entrainer_temps1_essai_")))
    resultat = entrainer(config, args.section, modele)
    if not args.sans_evaluation:
        resultat["test"] = evaluer(config, args.section, modele)
    resultat.pop("indices_validation")
    print(json.dumps(resultat, ensure_ascii=False, indent=2))
    return resultat


if __name__ == "__main__":
    main()
