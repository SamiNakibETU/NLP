#!/usr/bin/env python3
"""Benchmark de trois traducteurs arabe -> français sur les transcriptions XML MGB-2.

Le gagnant traduira ensuite le reste du corpus pour produire les sous-titres
français d'entraînement. Étapes, dans l'ordre :

1. ``lire_dossier_xml`` puis ``selectionner_segments`` : jeu gelé de 60 segments
   (``segments_benchmark.jsonl``), relu tel quel aux lancements suivants.
2. ``traduire_candidat`` pour chaque modèle (GPU), avec reprise si un fichier
   ``traductions_<nom>.jsonl`` existe déjà.
3. ``controles_automatiques`` : échecs durs (vide, arabe restant, méta-texte,
   pas du français) et signaux (longueur, chiffres, noms propres, débit de lecture).
4. ``ecrire_grille_aveugle`` : grille A/B/C mélangée par segment, clé à part.
5. ``noter_grilles`` puis ``choisir_modele`` : la note de sens humaine décide,
   les mesures automatiques départagent les égalités.
"""

import csv
import json
import random
import re
import statistics
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import yaml

RACINE_PROJET = Path(__file__).resolve().parents[2]
for _dossier in (RACINE_PROJET / "code" / "mesures", RACINE_PROJET / "code" / "donnees"):
    if str(_dossier) not in sys.path:
        sys.path.insert(0, str(_dossier))

from scores import calculer_chrf_bleu, calculer_comet  # noqa: E402
from seuils import charger_seuils  # noqa: E402

CHEMIN_CONFIG = RACINE_PROJET / "configs" / "benchmark_traduction.yaml"
CHEMIN_NOMS = RACINE_PROJET / "donnees" / "noms" / "noms_v1.csv"
LETTRES = "ABCDEFGH"
PONCTUATION_SEULE = {"،", ".", "؟", "!", ":", "؛", ",", "?", "-"}
MOTIF_ARABE = re.compile(r"[\u0600-\u06FF]")
MOTIF_CHIFFRES = re.compile(r"[0-9\u0660-\u0669\u06F0-\u06F9]+")
META = ("voici la traduction", "traduction :", "traduction:", "here is", "translation:", "note :")
ECHECS_DURS = ("vide", "arabe_restant", "meta_texte", "pas_francais")


def charger_config(chemin: Optional[Path] = None) -> Dict[str, Any]:
    with open(chemin or CHEMIN_CONFIG, "r", encoding="utf-8") as fichier:
        return yaml.safe_load(fichier)


# ----------------------------------------------------------------------------
# Lecture XML MGB-2
# ----------------------------------------------------------------------------

def lire_xml_mgb2(chemin: Path) -> List[Dict[str, Any]]:
    """Un segment par balise <segment>, dans l'ordre du fichier.

    ``wmer`` : écart (%) entre le texte Al Jazeera et le décodage de l'audio par
    MGB-2 ; 0 veut dire que l'audio confirme le texte. ``debit`` est l'attribut
    AWD, proche du nombre de mots par seconde.
    """
    racine = ET.parse(str(chemin)).getroot()
    enregistrement = racine.find("./head/recording")
    episode = enregistrement.get("filename") if enregistrement is not None else Path(chemin).stem
    infos = dict(enregistrement.attrib) if enregistrement is not None else {}
    segments = []
    for balise in racine.iter("segment"):
        mots = [element.text.strip() for element in balise.iter("element") if element.text and element.text.strip()]
        debut = float(balise.get("starttime", 0.0))
        fin = float(balise.get("endtime", 0.0))
        segments.append({
            "id": balise.get("id"),
            "episode": episode,
            "titre": infos.get("title", ""),
            "date": infos.get("date", ""),
            "locuteur": (balise.get("who") or "").replace(f"{episode.replace('-', '_')}_", ""),
            "debut": debut,
            "fin": fin,
            "duree_s": round(fin - debut, 3),
            "ar": " ".join(mots),
            "nb_mots": sum(1 for mot in mots if mot not in PONCTUATION_SEULE),
            "wmer": float(balise.get("WMER", 0.0)),
            "pmer": float(balise.get("PMER", 0.0)),
            "debit": float(balise.get("AWD", 0.0)),
        })
    return segments


def lire_dossier_xml(dossier: Path) -> List[Dict[str, Any]]:
    fichiers = sorted(Path(dossier).glob("*.xml"))
    if not fichiers:
        raise FileNotFoundError(f"Aucun fichier XML MGB-2 dans {dossier}")
    segments: List[Dict[str, Any]] = []
    for fichier in fichiers:
        segments.extend(lire_xml_mgb2(fichier))
    return segments


# ----------------------------------------------------------------------------
# Lecture des fichiers « marqués » (un bloc <<id | début-fin>> puis le texte)
# ----------------------------------------------------------------------------

MOTIF_BLOC = re.compile(r"^<<\s*(\S+)\s*\|\s*([\d:.,]+)\s*-\s*([\d:.,]+)\s*>>\s*$")


def _secondes(horodatage: str) -> float:
    total = 0.0
    for morceau in horodatage.replace(",", ".").split(":"):
        total = total * 60 + float(morceau)
    return total


def lire_marque(chemin: Path) -> List[Dict[str, Any]]:
    """Blocs ``<<id | HH:MM:SS-HH:MM:SS>>`` suivis de leur texte, dans l'ordre du fichier."""
    blocs: List[Dict[str, Any]] = []
    for ligne in Path(chemin).read_text(encoding="utf-8-sig").splitlines():
        entete = MOTIF_BLOC.match(ligne.strip())
        if entete:
            identifiant, debut, fin = entete.groups()
            blocs.append({"id": identifiant, "debut": _secondes(debut), "fin": _secondes(fin), "texte": ""})
        elif blocs and ligne.strip():
            blocs[-1]["texte"] = (blocs[-1]["texte"] + " " + ligne.strip()).strip()
    if not blocs:
        raise ValueError(f"Aucun bloc <<id | début-fin>> dans {chemin}")
    return blocs


def _episode_du_bloc(identifiant: str) -> str:
    return identifiant.rsplit("_utt_", 1)[0]


def trouver_marque_francais(chemin_ar: Path, dossiers: Sequence[Path] = ()) -> Optional[Path]:
    """Le fichier de traduction de la même émission : mêmes blocs, texte sans arabe."""
    episode = _episode_du_bloc(lire_marque(chemin_ar)[0]["id"])
    for dossier in [Path(chemin_ar).parent, *dossiers]:
        for candidat in sorted(Path(dossier).glob(f"{episode}*.txt")):
            if candidat.resolve() == Path(chemin_ar).resolve():
                continue
            try:
                blocs = lire_marque(candidat)
            except ValueError:
                continue
            textes = " ".join(b["texte"] for b in blocs[:20])
            if textes and not MOTIF_ARABE.search(textes):
                return candidat
    return None


def segments_marque(
    chemin_ar: Path,
    chemin_fr: Optional[Path] = None,
    nb_contexte: int = 2,
) -> List[Dict[str, Any]]:
    """Segments au format de ``lire_xml_mgb2``, avec ``fr_reference`` quand la traduction existe.

    Les horodatages sont à la seconde près : ``duree_s`` est donc approximative.
    """
    blocs = lire_marque(chemin_ar)
    references = {b["id"]: b["texte"] for b in lire_marque(chemin_fr)} if chemin_fr else {}
    segments = []
    for position, bloc in enumerate(blocs):
        mots = bloc["texte"].split()
        segments.append({
            "id": bloc["id"],
            "episode": _episode_du_bloc(bloc["id"]),
            "locuteur": "",
            "debut": bloc["debut"],
            "fin": bloc["fin"],
            "duree_s": round(bloc["fin"] - bloc["debut"], 3),
            "ar": bloc["texte"],
            "nb_mots": sum(1 for mot in mots if mot not in PONCTUATION_SEULE),
            "wmer": 0.0,
            "contexte_ar": [b["texte"] for b in blocs[max(0, position - nb_contexte):position]],
            "fr_reference": references.get(bloc["id"], ""),
        })
    return segments


def decrire_corpus(segments: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    durees = sorted(s["duree_s"] for s in segments)
    wmer = [s["wmer"] for s in segments]

    def quantile(valeurs: Sequence[float], q: float) -> float:
        return round(valeurs[min(len(valeurs) - 1, int(q * len(valeurs)))], 2)

    return {
        "episodes": len({s["episode"] for s in segments}),
        "segments": len(segments),
        "heures": round(sum(durees) / 3600.0, 2),
        "mots": sum(s["nb_mots"] for s in segments),
        "duree_mediane_s": quantile(durees, 0.5),
        "duree_p90_s": quantile(durees, 0.9),
        "duree_max_s": round(durees[-1], 2),
        "segments_wmer_0": sum(1 for x in wmer if x == 0),
        "segments_plus_de_25s": sum(1 for d in durees if d > 25),
        "locuteurs": len({(s["episode"], s["locuteur"]) for s in segments}),
    }


# ----------------------------------------------------------------------------
# Sélection gelée
# ----------------------------------------------------------------------------

def selectionner_segments(
    segments: Sequence[Dict[str, Any]],
    nb: int = 60,
    graine: int = 42,
    duree_min_s: float = 5.0,
    duree_max_s: float = 25.0,
    mots_min: int = 6,
    wmer_max: float = 15.0,
    nb_contexte: int = 2,
) -> List[Dict[str, Any]]:
    """Tire ``nb`` segments éligibles, à tour de rôle par émission puis par locuteur.

    Chaque segment garde les ``nb_contexte`` segments arabes qui le précèdent dans
    l'émission : les phrases sont souvent coupées au milieu.
    """
    par_episode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        par_episode[segment["episode"]].append(segment)
    for liste in par_episode.values():
        liste.sort(key=lambda s: s["debut"])

    files: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for episode, liste in par_episode.items():
        for position, segment in enumerate(liste):
            if not (duree_min_s <= segment["duree_s"] <= duree_max_s):
                continue
            if segment["nb_mots"] < mots_min or segment["wmer"] > wmer_max:
                continue
            fiche = dict(segment)
            fiche["contexte_ar"] = [s["ar"] for s in liste[max(0, position - nb_contexte):position]]
            files[(episode, segment["locuteur"])].append(fiche)

    generateur = random.Random(graine)
    for cle in sorted(files):
        generateur.shuffle(files[cle])
    episodes = sorted({cle[0] for cle in files})
    choisis: List[Dict[str, Any]] = []
    while len(choisis) < nb and any(files.values()):
        for episode in episodes:
            locuteurs = sorted((cle for cle in files if cle[0] == episode and files[cle]), key=lambda c: -len(files[c]))
            if not locuteurs:
                continue
            for cle in locuteurs:
                if len(choisis) >= nb:
                    break
                choisis.append(files[cle].pop())
            if len(choisis) >= nb:
                break
    choisis.sort(key=lambda s: (s["episode"], s["debut"]))
    return choisis


def geler_selection(segments: Sequence[Dict[str, Any]], chemin: Path) -> List[Dict[str, Any]]:
    """Écrit la sélection la première fois ; ensuite la relit sans la changer."""
    chemin = Path(chemin)
    if chemin.is_file():
        return lire_jsonl(chemin)
    ecrire_jsonl(chemin, segments)
    return list(segments)


def charger_jeu(
    config: Dict[str, Any],
    dossier_sortie: Path,
    racine: Path = RACINE_PROJET,
) -> Tuple[List[Dict[str, Any]], Optional[Path]]:
    """Jeu commun à tous les notebooks, et le fichier de référence française s'il existe.

    Avec ``donnees.marque_ar`` : tous les blocs du fichier marqué. Sinon : tirage gelé dans les XML.
    """
    donnees = config["donnees"]
    sel = config["selection"]
    if donnees.get("marque_ar"):
        chemin_ar = Path(racine) / donnees["marque_ar"]
        if donnees.get("marque_fr"):
            chemin_fr: Optional[Path] = Path(racine) / donnees["marque_fr"]
        else:
            chemin_fr = trouver_marque_francais(chemin_ar, [Path(racine), Path(racine) / "donnees"])
        return segments_marque(chemin_ar, chemin_fr, sel["nb_phrases_contexte"]), chemin_fr
    tirage = selectionner_segments(
        lire_dossier_xml(Path(racine) / donnees["dossier_xml"]),
        nb=sel["nb_segments"], graine=sel["graine"],
        duree_min_s=sel["duree_min_s"], duree_max_s=sel["duree_max_s"],
        mots_min=sel["mots_min"], wmer_max=sel["wmer_max"],
        nb_contexte=sel["nb_phrases_contexte"],
    )
    return geler_selection(tirage, Path(dossier_sortie) / "segments_benchmark.jsonl"), None


def charger_traductions_supplementaires(
    dossier: Path,
    segments: Sequence[Dict[str, Any]],
    deja: Sequence[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Relit les ``traductions_<nom>.jsonl`` d'autres notebooks (Seamless) qui couvrent les mêmes ids."""
    ids = [s["id"] for s in segments]
    trouves: Dict[str, List[Dict[str, Any]]] = {}
    for chemin in sorted(Path(dossier).glob("traductions_*.jsonl")):
        nom = chemin.stem[len("traductions_"):]
        if nom in deja:
            continue
        lignes = lire_jsonl(chemin)
        if [ligne["id"] for ligne in lignes] == ids:
            trouves[nom] = lignes
    return trouves


def lire_jsonl(chemin: Path) -> List[Dict[str, Any]]:
    with open(chemin, "r", encoding="utf-8") as fichier:
        return [json.loads(ligne) for ligne in fichier if ligne.strip()]


def ecrire_jsonl(chemin: Path, lignes: Sequence[Dict[str, Any]]) -> None:
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as fichier:
        for ligne in lignes:
            fichier.write(json.dumps(ligne, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
# Traducteurs
# ----------------------------------------------------------------------------

def nettoyer_sortie(texte: str) -> str:
    """Retire le préfixe "Traduction :" et les guillemets englobants ; garde le premier paragraphe."""
    texte = (texte or "").strip()
    texte = re.sub(r"^(traduction|translation|français|french)\s*:\s*", "", texte, flags=re.IGNORECASE)
    texte = texte.split("\n\n")[0].strip()
    if len(texte) >= 2 and texte[0] in "\"«“" and texte[-1] in "\"»”":
        texte = texte[1:-1].strip()
    return texte


def construire_message_llm(segment: Dict[str, Any]) -> str:
    contexte = [phrase for phrase in segment.get("contexte_ar") or [] if phrase]
    if contexte:
        lignes = "\n".join(contexte)
        return f"Contexte (ne pas traduire) :\n{lignes}\n\nSegment à traduire :\n{segment['ar']}"
    return f"Segment à traduire :\n{segment['ar']}"


def _appareil():
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


class TraducteurNLLB:
    def __init__(self, modele_id: str, taille_lot: int = 8, max_new_tokens: int = 256):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.appareil, dtype = _appareil()
        self.tokenizer = AutoTokenizer.from_pretrained(modele_id, src_lang="arb_Arab")
        self.modele = AutoModelForSeq2SeqLM.from_pretrained(modele_id, torch_dtype=dtype).to(self.appareil).eval()
        self.cible = self.tokenizer.convert_tokens_to_ids("fra_Latn")
        self.taille_lot = taille_lot
        self.max_new_tokens = max_new_tokens

    def traduire(self, segments: Sequence[Dict[str, Any]]) -> List[Tuple[str, float]]:
        import torch

        sorties: List[Tuple[str, float]] = []
        for debut in range(0, len(segments), self.taille_lot):
            lot = segments[debut:debut + self.taille_lot]
            depart = time.perf_counter()
            entrees = self.tokenizer([s["ar"] for s in lot], return_tensors="pt", padding=True, truncation=True, max_length=512)
            entrees = {cle: valeur.to(self.appareil) for cle, valeur in entrees.items()}
            with torch.inference_mode():
                ids = self.modele.generate(**entrees, forced_bos_token_id=self.cible, num_beams=4, max_new_tokens=self.max_new_tokens)
            textes = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
            par_segment = (time.perf_counter() - depart) / len(lot)
            sorties.extend((nettoyer_sortie(t), par_segment) for t in textes)
        return sorties

    def fermer(self) -> None:
        _liberer(self)


class TraducteurLLM:
    def __init__(self, modele_id: str, consigne: str, max_new_tokens: int = 256, quantification_4bits: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.appareil, dtype = _appareil()
        self.tokenizer = AutoTokenizer.from_pretrained(modele_id)
        arguments: Dict[str, Any] = {"torch_dtype": dtype}
        if self.appareil == "cuda" and quantification_4bits:
            from transformers import BitsAndBytesConfig

            arguments["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
            arguments["device_map"] = "auto"
        self.modele = AutoModelForCausalLM.from_pretrained(modele_id, **arguments)
        if "device_map" not in arguments:
            self.modele.to(self.appareil)
        self.modele.eval()
        self.consigne = consigne.strip()
        self.max_new_tokens = max_new_tokens

    def _prompt(self, segment: Dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": self.consigne},
            {"role": "user", "content": construire_message_llm(segment)},
        ]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def traduire(self, segments: Sequence[Dict[str, Any]]) -> List[Tuple[str, float]]:
        import torch

        sorties: List[Tuple[str, float]] = []
        for segment in segments:
            depart = time.perf_counter()
            entrees = self.tokenizer(self._prompt(segment), return_tensors="pt", add_special_tokens=False).to(self.modele.device)
            with torch.inference_mode():
                ids = self.modele.generate(
                    **entrees,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            texte = self.tokenizer.decode(ids[0, entrees["input_ids"].shape[1]:], skip_special_tokens=True)
            sorties.append((nettoyer_sortie(texte), time.perf_counter() - depart))
        return sorties

    def fermer(self) -> None:
        _liberer(self)


def _liberer(traducteur: Any) -> None:
    import gc

    traducteur.modele = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def fabriquer_traducteur(candidat: Dict[str, Any], config: Dict[str, Any]):
    generation = config.get("generation", {})
    if candidat["type"] == "nllb":
        return TraducteurNLLB(
            candidat["modele_id"],
            taille_lot=int(generation.get("taille_lot_nllb", 8)),
            max_new_tokens=int(generation.get("max_new_tokens", 256)),
        )
    if candidat["type"] == "llm":
        return TraducteurLLM(
            candidat["modele_id"],
            consigne=config["consigne_llm"],
            max_new_tokens=int(generation.get("max_new_tokens", 256)),
            quantification_4bits=bool(generation.get("quantification_4bits", True)),
        )
    raise ValueError(f"Type de candidat inconnu : {candidat['type']}")


def traduire_candidat(
    candidat: Dict[str, Any],
    segments: Sequence[Dict[str, Any]],
    dossier_sortie: Path,
    fabrique: Callable[[Dict[str, Any]], Any],
) -> List[Dict[str, Any]]:
    """Traduit tout le jeu avec un candidat. Relit le fichier s'il couvre déjà les mêmes ids."""
    chemin = Path(dossier_sortie) / f"traductions_{candidat['nom']}.jsonl"
    ids = [s["id"] for s in segments]
    if chemin.is_file():
        deja = lire_jsonl(chemin)
        if [ligne["id"] for ligne in deja] == ids:
            return deja
    traducteur = fabrique(candidat)
    try:
        resultats = traducteur.traduire(segments)
    finally:
        traducteur.fermer()
    lignes = [
        {"id": s["id"], "modele": candidat["nom"], "fr": fr, "temps_s": round(temps, 4)}
        for s, (fr, temps) in zip(segments, resultats)
    ]
    ecrire_jsonl(chemin, lignes)
    return lignes


# ----------------------------------------------------------------------------
# Contrôles automatiques (sans référence)
# ----------------------------------------------------------------------------

def _sans_accents(texte: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", texte.casefold()) if not unicodedata.combining(c))


def _chiffres_latins(texte: str) -> List[str]:
    table = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
    return [m.translate(table) for m in MOTIF_CHIFFRES.findall(texte)]


def _motif_nom_arabe(forme: str) -> "re.Pattern[str]":
    # mot entier, avec les préfixes collés و ب ل ف ك
    return re.compile(r"(?<![\u0600-\u06FF])[وبلفك]?" + re.escape(forme) + r"(?![\u0600-\u06FF])")


def _racine_nom_francais(nom: str) -> str:
    plat = re.sub(r"^(le|la|les|l')\s*", "", _sans_accents(nom))
    return plat[:max(4, len(plat) - 2)]


def charger_lexique_noms(chemin: Path = CHEMIN_NOMS) -> List[Tuple[List["re.Pattern[str]"], str]]:
    """(motifs arabes, racine française). Les variantes de plus de trois mots sont des titres : ignorées."""
    if not Path(chemin).is_file():
        return []
    entrees = []
    with open(chemin, "r", encoding="utf-8") as fichier:
        for ligne in csv.DictReader(fichier):
            formes = [ligne.get("forme_arabe", "").strip()]
            formes += [v.strip() for v in (ligne.get("variantes") or "").split("|") if v.strip()]
            formes = [f for f in formes if MOTIF_ARABE.search(f) and len(f.split()) <= 3]
            francais = (ligne.get("forme_francaise") or "").strip()
            if formes and francais:
                entrees.append(([_motif_nom_arabe(f) for f in formes], _racine_nom_francais(francais)))
    return entrees


def _semble_francais(texte: str) -> bool:
    try:
        import py3langid

        return py3langid.classify(texte)[0] == "fr"
    except ImportError:
        mots = set(_sans_accents(texte).split())
        return bool(mots & {"le", "la", "les", "de", "des", "et", "est", "un", "une", "a", "en", "du", "que", "qui", "pour"})


def controler_traduction(
    segment: Dict[str, Any],
    francais: str,
    ratio_min: float,
    ratio_max: float,
    lexique: Sequence[Tuple[List[str], str]] = (),
) -> Dict[str, Any]:
    signaux: List[str] = []
    texte = (francais or "").strip()
    if not texte:
        signaux.append("vide")
    else:
        if MOTIF_ARABE.search(texte):
            signaux.append("arabe_restant")
        if any(marque in texte.casefold() for marque in META):
            signaux.append("meta_texte")
        if not _semble_francais(texte):
            signaux.append("pas_francais")
        ratio = len(texte) / max(1, len(segment["ar"]))
        if not (ratio_min <= ratio <= ratio_max):
            signaux.append("longueur_suspecte")
        attendus = _chiffres_latins(segment["ar"])
        if attendus and any(c not in _chiffres_latins(texte) for c in attendus):
            signaux.append("chiffre_manquant")
        texte_plat = _sans_accents(texte)
        for motifs, racine_fr in lexique:
            if any(m.search(segment["ar"]) for m in motifs) and racine_fr not in texte_plat:
                signaux.append("nom_propre_different")
                break
    duree = float(segment.get("duree_s") or 0.0)
    return {
        "signaux": signaux,
        "echec_dur": any(s in ECHECS_DURS for s in signaux),
        "caracteres_par_s": round(len(texte) / duree, 2) if duree > 0 else None,
    }


def controles_automatiques(
    segments: Sequence[Dict[str, Any]],
    traductions: Dict[str, Sequence[Dict[str, Any]]],
    lexique: Optional[Sequence[Tuple[List[str], str]]] = None,
) -> Dict[str, Dict[str, Any]]:
    proposes = charger_seuils()["proposes"]
    lexique = charger_lexique_noms() if lexique is None else lexique
    resume: Dict[str, Dict[str, Any]] = {}
    for nom, lignes in traductions.items():
        par_id = {ligne["id"]: ligne for ligne in lignes}
        compte: Dict[str, int] = defaultdict(int)
        echecs = 0
        cps: List[float] = []
        temps: List[float] = []
        for segment in segments:
            ligne = par_id.get(segment["id"], {"fr": "", "temps_s": 0.0})
            controle = controler_traduction(
                segment, ligne["fr"], proposes["rapport_longueur_min"], proposes["rapport_longueur_max"], lexique
            )
            ligne["signaux"] = controle["signaux"]
            for signal in controle["signaux"]:
                compte[signal] += 1
            echecs += int(controle["echec_dur"])
            if controle["caracteres_par_s"] is not None:
                cps.append(controle["caracteres_par_s"])
            temps.append(float(ligne.get("temps_s") or 0.0))
        n = max(1, len(segments))
        resume[nom] = {
            "echec_dur_pct": round(100.0 * echecs / n, 1),
            "signaux_pct": {cle: round(100.0 * valeur / n, 1) for cle, valeur in sorted(compte.items())},
            "caracteres_par_s_median": round(statistics.median(cps), 1) if cps else None,
            "temps_moyen_s": round(sum(temps) / n, 3),
        }
    return resume


def scores_cometkiwi(segments: Sequence[Dict[str, Any]], traductions: Dict[str, Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    """Qualité estimée sans référence (Unbabel/wmt22-cometkiwi-da, accès Hugging Face à demander)."""
    try:
        from comet import download_model, load_from_checkpoint

        modele = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))
    except Exception as exc:
        return {nom: "non mesuré" for nom in traductions} | {"_motif": str(exc)[:200]}
    import torch

    gpus = 1 if torch.cuda.is_available() else 0
    sorties: Dict[str, Any] = {}
    for nom, lignes in traductions.items():
        par_id = {ligne["id"]: ligne["fr"] for ligne in lignes}
        donnees = [{"src": s["ar"], "mt": par_id.get(s["id"], "")} for s in segments]
        sorties[nom] = round(float(modele.predict(donnees, batch_size=16, gpus=gpus, progress_bar=False).system_score), 4)
    return sorties


# ----------------------------------------------------------------------------
# Évaluation humaine à l'aveugle
# ----------------------------------------------------------------------------

def ecrire_grille_aveugle(
    segments: Sequence[Dict[str, Any]],
    traductions: Dict[str, Sequence[Dict[str, Any]]],
    dossier: Path,
    graine: int = 42,
) -> Tuple[Path, Path]:
    """Grille à remplir par chaque annotateur (copie ``grille_<initiales>.csv``) et clé secrète.

    Notes : ``sens`` 1 (contresens) à 5 (fidèle), ``fluidite`` 1 à 5,
    ``noms_chiffres`` ok / ko / na, ``prefere`` lettre ou ``egal``.
    ``fr_reference`` : traduction humaine facultative, qui active chrF++ et COMET.
    """
    noms = sorted(traductions)
    dossier = Path(dossier)
    chemin_grille = dossier / "grille_aveugle_vierge.csv"
    chemin_cle = dossier / "cle_secrete.json"
    deja_notees = [p for p in dossier.glob("grille_*.csv") if p.name != chemin_grille.name]
    if chemin_cle.is_file() and deja_notees:
        ancienne = json.loads(chemin_cle.read_text(encoding="utf-8"))
        anciens_noms = sorted({nom for attribution in ancienne.values() for nom in attribution.values()})
        if anciens_noms == noms and sorted(ancienne) == sorted(s["id"] for s in segments):
            return chemin_grille, chemin_cle
        raise RuntimeError(
            f"Des grilles sont déjà notées ({', '.join(p.name for p in deja_notees)}) avec les modèles "
            f"{anciens_noms}. Les modèles sont maintenant {noms} : réécrire la clé rendrait ces notes fausses. "
            "Déplace les grilles notées ailleurs avant de régénérer, ou retire le modèle ajouté."
        )
    lettres = list(LETTRES[:len(noms)])
    par_modele = {nom: {ligne["id"]: ligne["fr"] for ligne in lignes} for nom, lignes in traductions.items()}
    generateur = random.Random(graine)
    cle: Dict[str, Dict[str, str]] = {}
    colonnes = ["id", "contexte_ar", "ar"]
    colonnes += [f"traduction_{l}" for l in lettres]
    colonnes += [f"{critere}_{l}" for critere in ("sens", "fluidite", "noms_chiffres") for l in lettres]
    colonnes += ["prefere", "fr_reference", "commentaire"]
    lignes_grille = []
    for segment in segments:
        ordre = list(noms)
        generateur.shuffle(ordre)
        cle[segment["id"]] = dict(zip(lettres, ordre))
        ligne = {colonne: "" for colonne in colonnes}
        ligne.update({
            "id": segment["id"],
            "contexte_ar": " / ".join(segment.get("contexte_ar") or []),
            "ar": segment["ar"],
        })
        for lettre, nom in zip(lettres, ordre):
            ligne[f"traduction_{lettre}"] = par_modele[nom].get(segment["id"], "")
        lignes_grille.append(ligne)
    dossier.mkdir(parents=True, exist_ok=True)
    with open(chemin_grille, "w", encoding="utf-8-sig", newline="") as fichier:
        ecrivain = csv.DictWriter(fichier, fieldnames=colonnes)
        ecrivain.writeheader()
        ecrivain.writerows(lignes_grille)
    chemin_cle.write_text(json.dumps(cle, ensure_ascii=False, indent=2), encoding="utf-8")
    return chemin_grille, chemin_cle


def _note(valeur: str) -> Optional[float]:
    valeur = (valeur or "").strip().replace(",", ".")
    try:
        note = float(valeur)
    except ValueError:
        return None
    return note if 1 <= note <= 5 else None


def _intervalle_bootstrap(ecarts: Sequence[float], tirages: int, graine: int = 42) -> Tuple[float, float]:
    generateur = random.Random(graine)
    n = len(ecarts)
    moyennes = sorted(sum(ecarts[generateur.randrange(n)] for _ in range(n)) / n for _ in range(tirages))
    return round(moyennes[int(0.025 * tirages)], 3), round(moyennes[min(tirages - 1, int(0.975 * tirages))], 3)


def scores_avec_reference(
    segments: Sequence[Dict[str, Any]],
    traductions: Dict[str, Sequence[Dict[str, Any]]],
    references: Optional[Dict[str, str]] = None,
    avec_comet: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """chrF++, BLEU et COMET contre la traduction humaine.

    Sans ``references``, prend le champ ``fr_reference`` des segments.
    """
    if references is None:
        references = {s["id"]: s["fr_reference"] for s in segments if (s.get("fr_reference") or "").strip()}
    ids = [s["id"] for s in segments if s["id"] in references]
    if not ids:
        return {}
    sources = {s["id"]: s["ar"] for s in segments}
    refs = [references[i] for i in ids]
    gpus = 0
    if avec_comet:
        try:
            import torch

            gpus = 1 if torch.cuda.is_available() else 0
        except ImportError:
            pass
    resultats: Dict[str, Dict[str, Any]] = {}
    for nom in sorted(traductions):
        par_id = {ligne["id"]: ligne["fr"] for ligne in traductions[nom]}
        hypotheses = [par_id.get(i, "") for i in ids]
        mesure = calculer_chrf_bleu(refs, hypotheses)
        comet = calculer_comet([sources[i] for i in ids], hypotheses, refs, gpus=gpus) if avec_comet else {"comet": "non mesuré"}
        resultats[nom] = {"nb": len(ids), "chrf_pp": mesure["chrf_pp"], "bleu": mesure["bleu"], "comet": comet["comet"]}
    return resultats


def noter_grilles(
    chemins_grilles: Sequence[Path],
    chemin_cle: Path,
    segments: Sequence[Dict[str, Any]],
    traductions: Dict[str, Sequence[Dict[str, Any]]],
    nb_tirages: int = 1000,
    avec_comet: bool = True,
) -> Dict[str, Any]:
    """Dévoile la clé et agrège les notes de tous les annotateurs."""
    cle = json.loads(Path(chemin_cle).read_text(encoding="utf-8"))
    sens: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    fluidite: Dict[str, List[float]] = defaultdict(list)
    noms_ko: Dict[str, List[int]] = defaultdict(list)
    preferences: Dict[str, int] = defaultdict(int)
    nb_preferences = 0
    references: Dict[str, str] = {}
    choix_par_annotateur: List[Dict[str, str]] = []

    for chemin in chemins_grilles:
        choix: Dict[str, str] = {}
        with open(chemin, "r", encoding="utf-8-sig", newline="") as fichier:
            for ligne in csv.DictReader(fichier):
                sid = ligne["id"]
                if sid not in cle:
                    continue
                for lettre, nom in cle[sid].items():
                    note_sens = _note(ligne.get(f"sens_{lettre}", ""))
                    if note_sens is not None:
                        sens[nom][sid].append(note_sens)
                    note_fluidite = _note(ligne.get(f"fluidite_{lettre}", ""))
                    if note_fluidite is not None:
                        fluidite[nom].append(note_fluidite)
                    marque = (ligne.get(f"noms_chiffres_{lettre}", "") or "").strip().lower()
                    if marque in ("ok", "ko"):
                        noms_ko[nom].append(int(marque == "ko"))
                prefere = (ligne.get("prefere", "") or "").strip().upper()
                if prefere in cle[sid]:
                    preferences[cle[sid][prefere]] += 1
                    nb_preferences += 1
                    choix[sid] = cle[sid][prefere]
                elif prefere == "EGAL":
                    nb_preferences += 1
                    choix[sid] = "egal"
                reference = (ligne.get("fr_reference", "") or "").strip()
                if reference and sid not in references:
                    references[sid] = reference
        choix_par_annotateur.append(choix)

    noms = sorted(traductions)
    humain: Dict[str, Dict[str, Any]] = {}
    for nom in noms:
        moyennes = [statistics.mean(v) for v in sens[nom].values()]
        humain[nom] = {
            "sens_moyen": round(statistics.mean(moyennes), 3) if moyennes else None,
            "nb_segments_notes": len(moyennes),
            "fluidite_moyenne": round(statistics.mean(fluidite[nom]), 3) if fluidite[nom] else None,
            "noms_chiffres_ko_pct": round(100.0 * statistics.mean(noms_ko[nom]), 1) if noms_ko[nom] else None,
            "prefere_pct": round(100.0 * preferences[nom] / nb_preferences, 1) if nb_preferences else None,
        }

    notes_valides = [n for n in noms if humain[n]["sens_moyen"] is not None]
    if notes_valides:
        meilleur = max(notes_valides, key=lambda n: humain[n]["sens_moyen"])
        for nom in notes_valides:
            if nom == meilleur:
                continue
            communs = sorted(set(sens[meilleur]) & set(sens[nom]))
            ecarts = [statistics.mean(sens[meilleur][s]) - statistics.mean(sens[nom][s]) for s in communs]
            if ecarts:
                bas, haut = _intervalle_bootstrap(ecarts, nb_tirages)
                humain[nom]["ecart_sens_vs_meilleur"] = {"ecart": round(statistics.mean(ecarts), 3), "ic95": [bas, haut], "significatif": bas > 0}

    accord = None
    if len(choix_par_annotateur) >= 2:
        premier, second = choix_par_annotateur[0], choix_par_annotateur[1]
        communs = set(premier) & set(second)
        if communs:
            accord = round(100.0 * sum(premier[s] == second[s] for s in communs) / len(communs), 1)

    avec_reference = scores_avec_reference(segments, traductions, references, avec_comet) if references else {}

    return {
        "humain": humain,
        "accord_preference_pct": accord,
        "nb_annotateurs": len(chemins_grilles),
        "avec_reference": avec_reference,
    }


# ----------------------------------------------------------------------------
# Décision
# ----------------------------------------------------------------------------

def _valeur(x: Any) -> float:
    return float(x) if isinstance(x, (int, float)) else float("-inf")


def choisir_modele(
    automatique: Dict[str, Dict[str, Any]],
    humain: Optional[Dict[str, Any]] = None,
    cometkiwi: Optional[Dict[str, Any]] = None,
    echec_dur_max_pct: float = 5.0,
    nb_segments_cible: int = 50000,
) -> Dict[str, Any]:
    """Règle, dans l'ordre :

    1. écarter un modèle dont plus de ``echec_dur_max_pct`` % des sorties sont inutilisables ;
    2. garder le meilleur sens moyen humain, et ceux dont l'écart avec lui n'est pas significatif ;
    3. départager par COMET avec référence, puis chrF++, puis COMETKiwi, puis préférence, puis vitesse.

    Sans grille notée, le choix est marqué provisoire.
    """
    humain_par_modele = (humain or {}).get("humain", {})
    reference = (humain or {}).get("avec_reference", {})
    cometkiwi = cometkiwi or {}
    restants = [n for n, r in automatique.items() if r["echec_dur_pct"] <= echec_dur_max_pct]
    ecartes = {n: f"{r['echec_dur_pct']} % de sorties inutilisables" for n, r in automatique.items() if n not in restants}
    if not restants:
        return {"choix": None, "provisoire": True, "ecartes": ecartes, "raison": "aucun modèle sous le seuil d'échec"}

    notes = {n: humain_par_modele.get(n, {}).get("sens_moyen") for n in restants}
    provisoire = all(v is None for v in notes.values())
    if provisoire:
        egalite = list(restants)
    else:
        meilleur = max((n for n in restants if notes[n] is not None), key=lambda n: notes[n])
        egalite = [meilleur] + [
            n for n in restants
            if n != meilleur and notes[n] is not None
            and not humain_par_modele[n].get("ecart_sens_vs_meilleur", {}).get("significatif", True)
        ]

    def cle_tri(nom: str):
        return (
            _valeur(reference.get(nom, {}).get("comet")),
            _valeur(reference.get(nom, {}).get("chrf_pp")),
            _valeur(cometkiwi.get(nom)),
            _valeur(humain_par_modele.get(nom, {}).get("prefere_pct")),
            -automatique[nom]["temps_moyen_s"],
        )

    choix = max(egalite, key=cle_tri)
    temps = automatique[choix]["temps_moyen_s"]
    return {
        "choix": choix,
        "provisoire": provisoire,
        "a_egalite_avec": [n for n in egalite if n != choix],
        "ecartes": ecartes,
        "heures_projetees_corpus": round(temps * nb_segments_cible / 3600.0, 1),
        "raison": (
            ("aucune note humaine : classement par COMET puis chrF++ avec référence, puis COMETKiwi, puis vitesse"
             if reference else "aucune note humaine : classement par COMETKiwi puis vitesse") if provisoire
            else "meilleur sens humain" + (" (égalité départagée par les mesures automatiques)" if len(egalite) > 1 else "")
        ),
    }


def enregistrer_dans_resultats(
    automatique: Dict[str, Dict[str, Any]],
    humain: Optional[Dict[str, Any]],
    cometkiwi: Optional[Dict[str, Any]],
    candidats: Sequence[Dict[str, Any]],
    chemin_csv: Optional[Path] = None,
    jeu: str = "bench_trad_xml",
) -> None:
    from tableau_resultats import enregistrer_resultat

    identifiants = {c["nom"]: c["modele_id"] for c in candidats}
    for nom, auto in automatique.items():
        lignes = [("echec_dur (%)", auto["echec_dur_pct"]), ("temps_par_segment (s)", auto["temps_moyen_s"])]
        if cometkiwi and isinstance(cometkiwi.get(nom), (int, float)):
            lignes.append(("COMETKiwi", cometkiwi[nom]))
        h = (humain or {}).get("humain", {}).get(nom, {})
        if h.get("sens_moyen") is not None:
            lignes.append(("sens_humain (1-5)", h["sens_moyen"]))
        r = (humain or {}).get("avec_reference", {}).get(nom, {})
        for cle, mesure in (("chrf_pp", "chrF++"), ("bleu", "BLEU"), ("comet", "COMET")):
            if cle in r:
                lignes.append((mesure, r[cle]))
        for mesure, valeur in lignes:
            enregistrer_resultat(
                systeme=f"TRAD:{nom}",
                version_modele=identifiants.get(nom, nom),
                jeu=jeu,
                mesure=mesure,
                valeur=valeur,
                configuration="configs/benchmark_traduction.yaml",
                chemin_csv=chemin_csv,
            )
