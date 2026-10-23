#!/usr/bin/env python3
"""Manifestes S3 : ``train.jsonl`` et ``test.jsonl`` tirés du train MGB-2.

Étapes, dans l'ordre :

1. ``lire_corpus`` : lit tous les XML de ``racine_mgb2/sous_dossier_xml``
   avec ``lire_xml_mgb2`` (code/traduction/benchmark_traduction.py).
2. ``filtrer_segments`` : seuils de ``code/donnees/seuils.yaml`` (durée, mots,
   WMER, segments max par épisode). Un rejet compte pour son premier motif.
3. Découpage par épisode avec ``determiner_partie`` (hachage MD5, pipeline_mgb2.py) :
   une émission va entièrement en train ou entièrement en test.
4. ``tirer_segments`` : tirage emboîté. Avec la même graine, les N premiers
   segments d'un tirage de M > N sont exactement le tirage de N.
5. Écriture des manifestes (une ligne JSON par segment, chemins wav relatifs
   à ``racine_mgb2``) et de ``resume_manifeste.json``.

Le texte arabe reste brut. Le champ ``texte_fr`` sera ajouté pour le temps 2.
"""

import argparse
import json
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

RACINE_PROJET = Path(__file__).resolve().parents[2]
for _dossier in (RACINE_PROJET / "code" / "donnees", RACINE_PROJET / "code" / "traduction"):
    if str(_dossier) not in sys.path:
        sys.path.insert(0, str(_dossier))

from benchmark_traduction import lire_xml_mgb2  # noqa: E402
from pipeline_mgb2 import determiner_partie  # noqa: E402
from seuils import charger_seuils  # noqa: E402

CHEMIN_CONFIG = RACINE_PROJET / "configs" / "encodage_s3.yaml"
PARTIES = ("train", "test")
# Ordre d'application des filtres : un segment rejeté compte pour le premier motif.
MOTIFS_REJET = ("duree", "mots", "wmer", "max_par_episode")
# determiner_partie renvoie "entrainement" ou "validation" ; ici la validation sert de test.
NOM_PARTIE = {"entrainement": "train", "validation": "test"}


class ErreurManifeste(Exception):
    """Configuration ou données sources absentes ou incohérentes."""


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

def chemin_projet(valeur: str) -> Path:
    """Chemin absolu tel quel, chemin relatif résolu depuis la racine du dépôt."""
    chemin = Path(valeur).expanduser()
    return chemin if chemin.is_absolute() else RACINE_PROJET / chemin


def charger_config(chemin: Optional[Path] = None) -> Dict[str, Any]:
    chemin = Path(chemin or CHEMIN_CONFIG)
    if not chemin.is_file():
        raise ErreurManifeste(f"configuration introuvable : {chemin}")
    with open(chemin, "r", encoding="utf-8") as fichier:
        config = yaml.safe_load(fichier)
    attendues = {
        "sources": ("racine_mgb2", "sous_dossier_xml", "sous_dossier_wav"),
        "sortie": ("dossier",),
        "selection": ("graine", "pct_episodes_test", "nb_segments_train", "nb_segments_test"),
    }
    for section, cles in attendues.items():
        bloc = (config or {}).get(section)
        if not isinstance(bloc, dict):
            raise ErreurManifeste(f"{chemin} : section '{section}' absente")
        for cle in cles:
            if cle not in bloc:
                raise ErreurManifeste(f"{chemin} : clé '{section}.{cle}' absente")
    return config


def lire_bornes(seuils: Dict[str, Any]) -> Dict[str, Any]:
    """Les seuils du plan utilisés ici, lus dans seuils.yaml sans les modifier."""
    constantes = seuils["constantes"]
    return {
        "duree_min_s": constantes["duree_segment_min_s"],
        "duree_max_s": constantes["duree_segment_max_s"],
        "mots_min": constantes["mots_min_segment"],
        "wmer_max": seuils["proposes"]["ecart_ecoute_max_pct"],
        "max_par_episode": constantes["segments_max_par_episode"],
    }


# ----------------------------------------------------------------------------
# Lecture, filtres, découpage et tirage
# ----------------------------------------------------------------------------

def lire_corpus(dossier_xml: Path) -> List[Dict[str, Any]]:
    fichiers = sorted(Path(dossier_xml).glob("*.xml"))
    if not fichiers:
        raise ErreurManifeste(f"aucun fichier XML MGB-2 dans {dossier_xml}")
    segments: List[Dict[str, Any]] = []
    for fichier in fichiers:
        segments.extend(lire_xml_mgb2(fichier))
    return segments


def motif_rejet(segment: Dict[str, Any], bornes: Dict[str, Any]) -> Optional[str]:
    """Premier filtre non satisfait, ou None si le segment passe (bornes incluses)."""
    if not bornes["duree_min_s"] <= segment["duree_s"] <= bornes["duree_max_s"]:
        return "duree"
    if segment["nb_mots"] < bornes["mots_min"]:
        return "mots"
    if segment["wmer"] > bornes["wmer_max"]:
        return "wmer"
    return None


def filtrer_segments(
    segments: Iterable[Dict[str, Any]], bornes: Dict[str, Any], graine: int
) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], str]]]:
    """Renvoie (gardés, [(rejeté, motif)]).

    Au-delà de ``max_par_episode`` segments retenus dans un épisode, on en garde
    un tirage aléatoire (graine propre à l'épisode) plutôt que les premiers, pour
    ne pas privilégier le début des émissions.
    """
    par_episode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rejets: List[Tuple[Dict[str, Any], str]] = []
    for segment in segments:
        motif = motif_rejet(segment, bornes)
        if motif:
            rejets.append((segment, motif))
        else:
            par_episode[segment["episode"]].append(segment)

    gardes: List[Dict[str, Any]] = []
    for episode in sorted(par_episode):
        candidats = sorted(par_episode[episode], key=lambda s: (s["debut"], s["id"]))
        if len(candidats) > bornes["max_par_episode"]:
            elus = set(id(s) for s in random.Random(f"{graine}-{episode}").sample(candidats, bornes["max_par_episode"]))
            rejets.extend((s, "max_par_episode") for s in candidats if id(s) not in elus)
            candidats = [s for s in candidats if id(s) in elus]
        gardes.extend(candidats)
    return gardes, rejets


def partie_episode(episode: str, pct_test: int) -> str:
    return NOM_PARTIE[determiner_partie(episode, pct_test)]


def tirer_segments(segments: List[Dict[str, Any]], nb: int, graine: int, partie: str) -> List[Dict[str, Any]]:
    """Tirage emboîté : on mélange une liste triée par id, puis on garde le début.

    Le mélange ne dépend que de la graine et de l'ensemble de départ, pas de
    ``nb`` : le tirage de 5 000 est donc le début de celui de 50 000.
    """
    ordre = sorted(segments, key=lambda s: s["id"])
    random.Random(f"{graine}-{partie}").shuffle(ordre)
    return ordre[:nb]


def ligne_manifeste(segment: Dict[str, Any], sous_dossier_wav: str, partie: str) -> Dict[str, Any]:
    return {
        "id": segment["id"],
        "episode": segment["episode"],
        "wav": f"{sous_dossier_wav}/{segment['episode']}.wav",
        "debut": segment["debut"],
        "fin": segment["fin"],
        "duree_s": segment["duree_s"],
        "texte_ar": segment["ar"],
        "nb_mots": segment["nb_mots"],
        "wmer": segment["wmer"],
        "pmer": segment["pmer"],
        "awd": segment["debit"],
        "locuteur": segment["locuteur"],
        "date": segment["date"],
        "titre": segment["titre"],
        "partie": partie,
    }


def bilan(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "segments": len(segments),
        "heures": round(sum(s["duree_s"] for s in segments) / 3600, 2),
        "episodes": len({s["episode"] for s in segments}),
    }


# ----------------------------------------------------------------------------
# Construction complète
# ----------------------------------------------------------------------------

def construire_manifestes(config: Dict[str, Any], seuils: Dict[str, Any], dossier_sortie: Path) -> Dict[str, Any]:
    sources, selection = config["sources"], config["selection"]
    racine = chemin_projet(sources["racine_mgb2"])
    dossier_xml = racine / sources["sous_dossier_xml"]
    if not dossier_xml.is_dir():
        raise ErreurManifeste(f"dossier XML introuvable : {dossier_xml} (vérifier sources.racine_mgb2)")

    bornes = lire_bornes(seuils)
    graine, pct_test = selection["graine"], selection["pct_episodes_test"]
    segments = lire_corpus(dossier_xml)
    gardes, rejets = filtrer_segments(segments, bornes, graine)

    lus: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    eligibles: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rejets_partie: Dict[str, Counter] = {partie: Counter() for partie in PARTIES}
    for segment in segments:
        lus[partie_episode(segment["episode"], pct_test)].append(segment)
    for segment in gardes:
        eligibles[partie_episode(segment["episode"], pct_test)].append(segment)
    for segment, motif in rejets:
        rejets_partie[partie_episode(segment["episode"], pct_test)][motif] += 1

    episodes_communs = {s["episode"] for s in eligibles["train"]} & {s["episode"] for s in eligibles["test"]}
    if episodes_communs:
        raise ErreurManifeste(f"{len(episodes_communs)} épisode(s) à la fois en train et en test")

    nb_voulus = {"train": selection["nb_segments_train"], "test": selection["nb_segments_test"]}
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    resume: Dict[str, Any] = {
        "racine_mgb2": str(racine),
        "graine": graine,
        "pct_episodes_test": pct_test,
        "bornes": bornes,
        "parties": {},
    }
    for partie in PARTIES:
        if nb_voulus[partie] > len(eligibles[partie]):
            raise ErreurManifeste(
                f"{partie} : {nb_voulus[partie]} segments demandés, {len(eligibles[partie])} disponibles"
            )
        tirage = tirer_segments(eligibles[partie], nb_voulus[partie], graine, partie)
        wav_absents = sorted({
            s["episode"] for s in tirage if not (racine / sources["sous_dossier_wav"] / f"{s['episode']}.wav").is_file()
        })
        if wav_absents:
            raise ErreurManifeste(f"{partie} : {len(wav_absents)} wav introuvable(s), dont {wav_absents[0]}.wav")

        chemin = dossier_sortie / f"{partie}.jsonl"
        temporaire = chemin.with_suffix(".jsonl.tmp")
        with open(temporaire, "w", encoding="utf-8") as fichier:
            for segment in tirage:
                ligne = ligne_manifeste(segment, sources["sous_dossier_wav"], partie)
                fichier.write(json.dumps(ligne, ensure_ascii=False) + "\n")
        temporaire.replace(chemin)

        resume["parties"][partie] = {
            "lus": bilan(lus[partie]),
            "rejets": {motif: rejets_partie[partie][motif] for motif in MOTIFS_REJET},
            "selection_complete": bilan(eligibles[partie]),
            "tirage": bilan(tirage),
        }

    with open(dossier_sortie / "resume_manifeste.json", "w", encoding="utf-8") as fichier:
        json.dump(resume, fichier, ensure_ascii=False, indent=2)
    return resume


# ----------------------------------------------------------------------------
# Mode essai : petit corpus factice, sans réseau
# ----------------------------------------------------------------------------

def ecrire_corpus_factice(racine: Path, nb_episodes: int = 40, segments_par_episode: int = 12) -> None:
    """XML au format MGB-2 (utf8) et wav vides : seule leur présence est vérifiée ici."""
    (racine / "xml" / "utf8").mkdir(parents=True, exist_ok=True)
    (racine / "wav").mkdir(parents=True, exist_ok=True)
    generateur = random.Random(0)
    for numero in range(nb_episodes):
        episode = f"{numero:08X}-0000-0000-0000-000000000000"
        prefixe = episode.replace("-", "_")
        balises, debut = [], 0.0
        for indice in range(1, segments_par_episode + 1):
            duree = round(generateur.uniform(2.0, 30.0), 2)
            nb_mots = generateur.randint(3, 20)
            wmer = round(generateur.choice([0.0, 5.0, 12.0, 15.0, 20.0, 40.0]), 1)
            mots = "".join(
                f'<element id="{prefixe}_w{indice}_{k}_align" type="word">كلمة{k}</element>' for k in range(nb_mots)
            )
            balises.append(
                f'<segment AWD="1.5" PMER="{wmer}" WMER="{wmer}" endtime="{debut + duree:.2f}" '
                f'id="{prefixe}_utt_{indice}_align" starttime="{debut:.2f}" '
                f'who="{prefixe}_speaker1_align">{mots}</segment>'
            )
            debut = round(debut + duree + 0.5, 2)
        contenu = (
            '<?xml version="1.0" encoding="UTF-8"?><transcript><head>'
            f'<recording date="2013/01/17" filename="{episode}" title="essai"/></head>'
            f'<body><segments>{"".join(balises)}</segments></body></transcript>'
        )
        (racine / "xml" / "utf8" / f"{episode}.xml").write_text(contenu, encoding="utf-8")
        (racine / "wav" / f"{episode}.wav").touch()


def construire_parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manifestes train/test S3 tirés du train MGB-2.")
    parser.add_argument("--config", type=str, default=str(CHEMIN_CONFIG), help="Config YAML (encodage_s3.yaml).")
    parser.add_argument("--essai", action="store_true", help="Petit corpus factice, sortie dans un dossier temporaire.")
    return parser


def main(arguments: Optional[List[str]] = None) -> Dict[str, Any]:
    args = construire_parseur().parse_args(arguments)
    config = charger_config(args.config)
    seuils = charger_seuils()
    if args.essai:
        temporaire = Path(tempfile.mkdtemp(prefix="manifeste_s3_essai_"))
        ecrire_corpus_factice(temporaire / "mgb2")
        config["sources"]["racine_mgb2"] = str(temporaire / "mgb2")
        config["selection"]["nb_segments_train"] = 50
        config["selection"]["nb_segments_test"] = 2
        config["selection"]["pct_episodes_test"] = 10
        dossier_sortie = temporaire / "sortie"
    else:
        dossier_sortie = chemin_projet(config["sortie"]["dossier"])
    resume = construire_manifestes(config, seuils, dossier_sortie)
    print(json.dumps(resume["parties"], ensure_ascii=False, indent=2))
    print(f"Manifestes écrits dans {dossier_sortie}")
    return resume


if __name__ == "__main__":
    main()
