#!/usr/bin/env python3
"""Encodage S3 : vecteurs de l'encodeur Cohere Transcribe Arabic pour chaque segment du manifeste.

Étapes, dans l'ordre :

1. ``EncodeurS3.charger`` : modèle complet, puis on garde son encodeur Conformer
   (``get_encoder()``), gelé et en ``eval`` ; le décodeur est libéré.
2. Le manifeste (``code/donnees/manifeste_s3.py``) est découpé en fichiers de
   ``segments_par_fichier_npy`` segments, dans l'ordre du manifeste. Le tirage
   étant emboîté, les fichiers du pilote sont aussi les premiers du manifeste à 50 000.
3. Dans chaque fichier, ``former_lots`` regroupe des segments de durées voisines
   (complétion seulement jusqu'au plus long du lot, avec masque). On ne garde que
   les vecteurs utiles de chaque segment.
4. ``vecteurs_<numéro>.npy`` (float16, segments bout à bout) est écrit dans un
   fichier temporaire puis renommé ; ses lignes d'``index.jsonl`` suivent.
   Un fichier présent avec toutes ses lignes d'index est considéré comme fait.
5. ``verifier_encodage`` écrit ``rapport_encodage.json``.

Lecture : ``np.load(fichier, mmap_mode="r")[ligne_debut : ligne_debut + nb_vecteurs]``.
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml

RACINE_PROJET = Path(__file__).resolve().parents[2]
for _dossier in (RACINE_PROJET / "code" / "ecoute", RACINE_PROJET / "code" / "projecteur"):
    if str(_dossier) not in sys.path:
        sys.path.insert(0, str(_dossier))

from audio_s3 import FREQUENCE_ATTENDUE, charger_extrait  # noqa: E402
from encodeur import lire_dimension_encodeur  # noqa: E402

CHEMIN_CONFIG = RACINE_PROJET / "configs" / "encodage_s3.yaml"
PRECISIONS = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


class ErreurEncodage(Exception):
    """Configuration, modèle ou fichiers absents ou incohérents."""


# ----------------------------------------------------------------------------
# Configuration et appareil
# ----------------------------------------------------------------------------

def chemin_projet(valeur: str) -> Path:
    chemin = Path(valeur).expanduser()
    return chemin if chemin.is_absolute() else RACINE_PROJET / chemin


def charger_config(chemin: Optional[Path] = None) -> Dict[str, Any]:
    chemin = Path(chemin or CHEMIN_CONFIG)
    if not chemin.is_file():
        raise ErreurEncodage(f"configuration introuvable : {chemin}")
    with open(chemin, "r", encoding="utf-8") as fichier:
        config = yaml.safe_load(fichier)
    attendues = {
        "sources": ("racine_mgb2",),
        "sortie": ("dossier",),
        "encodage": ("modele_id", "appareil", "precision", "secondes_audio_par_lot", "segments_par_fichier_npy"),
        "controles": ("marge_disque_go", "nb_segments_reencodes", "tolerance_absolue", "tolerance_relative",
                      "tolerance_fin_wav_s"),
    }
    for section, cles in attendues.items():
        bloc = (config or {}).get(section)
        if not isinstance(bloc, dict):
            raise ErreurEncodage(f"{chemin} : section '{section}' absente")
        for cle in cles:
            if cle not in bloc:
                raise ErreurEncodage(f"{chemin} : clé '{section}.{cle}' absente")
    if config["encodage"]["precision"] not in PRECISIONS:
        raise ErreurEncodage(f"precision inconnue : {config['encodage']['precision']} (connues : {sorted(PRECISIONS)})")
    return config


def choisir_appareil(appareil: str) -> torch.device:
    """``auto`` : cuda, puis mps, puis cpu. Un appareil demandé mais absent est une erreur."""
    if appareil == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if appareil == "cuda" and not torch.cuda.is_available():
        raise ErreurEncodage("appareil 'cuda' demandé mais indisponible")
    if appareil == "mps" and not torch.backends.mps.is_available():
        raise ErreurEncodage("appareil 'mps' demandé mais indisponible")
    return torch.device(appareil)


# ----------------------------------------------------------------------------
# Encodeur
# ----------------------------------------------------------------------------

def nb_vecteurs_theoriques(nb_echantillons: int, preparateur, config_encodeur) -> int:
    """Nombre de vecteurs utiles pour ``nb_echantillons``, d'après le code de transformers.

    Trames Mel : (n + n_fft//2*2 - n_fft) // hop, soit n // 160. Puis, pour chaque
    convolution de sous-échantillonnage (noyau 3, pas 2, padding 1) :
    L -> floor((L - 1) / 2) + 1. Avec 3 convolutions (facteur 8) : ceil(ceil(ceil(L/2)/2)/2).
    """
    longueur = (nb_echantillons + preparateur.n_fft // 2 * 2 - preparateur.n_fft) // preparateur.hop_length
    noyau = config_encodeur.subsampling_conv_kernel_size
    pas = config_encodeur.subsampling_conv_stride
    complement = (noyau - 1) // 2 * 2 - noyau
    for _ in range(int(math.log2(config_encodeur.subsampling_factor))):
        longueur = math.floor((longueur + complement) / pas + 1)
    return int(longueur)


class EncodeurS3:
    """Encodeur gelé d'un modèle Cohere Transcribe et son préparateur de features."""

    def __init__(self, modele_asr, preparateur, appareil: torch.device, precision: torch.dtype, modele_id: str):
        self.encodeur = modele_asr.get_encoder()
        del modele_asr  # le décodeur n'est plus référencé : libéré
        self.encodeur.to(appareil, precision).eval()
        for p in self.encodeur.parameters():
            p.requires_grad = False
        self.preparateur = preparateur
        self.appareil = appareil
        self.precision = precision
        self.modele_id = modele_id
        self.d_enc = lire_dimension_encodeur(self.encodeur.config)

    @classmethod
    def charger(cls, modele_id: str, appareil: torch.device, precision: torch.dtype) -> "EncodeurS3":
        """Modèle réel (accès restreint : HF_TOKEN dans .env). Aucun repli vers un modèle factice."""
        from dotenv import load_dotenv
        from transformers import CohereAsrFeatureExtractor, CohereAsrForConditionalGeneration

        load_dotenv(RACINE_PROJET / ".env")
        jeton = os.environ.get("HF_TOKEN")
        modele = CohereAsrForConditionalGeneration.from_pretrained(modele_id, token=jeton, dtype=precision)
        preparateur = CohereAsrFeatureExtractor.from_pretrained(modele_id, token=jeton)
        return cls(modele, preparateur, appareil, precision, modele_id)

    @torch.no_grad()
    def encoder_lot(self, extraits: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Liste d'extraits 16 kHz -> liste de matrices float16 (T_i, d_enc), sans complétion."""
        entree = self.preparateur(list(extraits), sampling_rate=FREQUENCE_ATTENDUE, return_attention_mask=True,
                                  padding="longest", return_tensors="pt")
        if any(morceau is not None for _, morceau in entree["audio_chunk_index"]):
            raise ErreurEncodage("le préparateur a découpé un extrait trop long (max_audio_clip_s)")
        features = entree["input_features"].to(self.appareil, self.precision)
        masque = entree["attention_mask"].to(self.appareil)
        sortie = self.encodeur(features, attention_mask=masque)
        longueurs = sortie.attention_mask.sum(-1).tolist()
        etats = sortie.last_hidden_state.float().cpu().numpy()
        return [etats[i, : int(n)].astype(np.float16) for i, n in enumerate(longueurs)]

    def nb_vecteurs_attendus(self, nb_echantillons: int) -> int:
        return nb_vecteurs_theoriques(nb_echantillons, self.preparateur, self.encodeur.config)


def former_lots(durees: Sequence[float], secondes_par_lot: float) -> List[List[int]]:
    """Indices regroupés par durées voisines, chaque lot totalisant au plus ``secondes_par_lot``."""
    lots: List[List[int]] = []
    courant: List[int] = []
    total = 0.0
    for indice in sorted(range(len(durees)), key=lambda i: durees[i]):
        if courant and total + durees[indice] > secondes_par_lot:
            lots.append(courant)
            courant, total = [], 0.0
        courant.append(indice)
        total += durees[indice]
    if courant:
        lots.append(courant)
    return lots


# ----------------------------------------------------------------------------
# Stockage et reprise
# ----------------------------------------------------------------------------

def nom_fichier(numero: int) -> str:
    return f"vecteurs_{numero:05d}.npy"


def lire_jsonl(chemin: Path) -> List[Dict[str, Any]]:
    if not chemin.is_file():
        return []
    with open(chemin, "r", encoding="utf-8") as fichier:
        return [json.loads(ligne) for ligne in fichier if ligne.strip()]


def ecrire_jsonl_atomique(chemin: Path, lignes: Sequence[Dict[str, Any]]) -> None:
    temporaire = chemin.with_suffix(chemin.suffix + ".tmp")
    with open(temporaire, "w", encoding="utf-8") as fichier:
        for ligne in lignes:
            fichier.write(json.dumps(ligne, ensure_ascii=False) + "\n")
    temporaire.replace(chemin)


def fichiers_faits(dossier: Path, paquets: List[List[Dict[str, Any]]], modele_id: str) -> List[Dict[str, Any]]:
    """Lignes d'index valides : fichiers présents, avec toutes leurs lignes, dans l'ordre du manifeste.

    Les lignes d'un fichier incomplet sont écartées ; ``index.jsonl`` est réécrit sans elles.
    """
    index = lire_jsonl(dossier / "index.jsonl")
    for ligne in index:
        if ligne["encodeur"] != modele_id:
            raise ErreurEncodage(f"{dossier} contient des vecteurs de {ligne['encodeur']}, pas de {modele_id}")
    par_fichier: Dict[str, List[Dict[str, Any]]] = {}
    for ligne in index:
        par_fichier.setdefault(ligne["fichier_npy"], []).append(ligne)
    valides: List[Dict[str, Any]] = []
    for numero, paquet in enumerate(paquets):
        nom = nom_fichier(numero)
        lignes = par_fichier.get(nom, [])
        chemin = dossier / nom
        if not chemin.is_file() or [l["id"] for l in lignes] != [e["id"] for e in paquet]:
            break
        if np.load(chemin, mmap_mode="r").shape[0] != sum(l["nb_vecteurs"] for l in lignes):
            break
        valides.extend(lignes)
    if len(valides) != len(index):
        ecrire_jsonl_atomique(dossier / "index.jsonl", valides)
    return valides


def encoder_paquet(encodeur: EncodeurS3, paquet: List[Dict[str, Any]], racine: Path,
                   secondes_par_lot: float, tolerance_fin_s: float = 0.0):
    """Renvoie (vecteurs de chaque segment, nombre d'échantillons réellement lus)."""
    vecteurs: List[Optional[np.ndarray]] = [None] * len(paquet)
    echantillons: List[int] = [0] * len(paquet)
    for lot in former_lots([e["duree_s"] for e in paquet], secondes_par_lot):
        extraits = [charger_extrait(paquet[i], racine, tolerance_fin_s=tolerance_fin_s) for i in lot]
        for i, x, v in zip(lot, extraits, encodeur.encoder_lot(extraits)):
            vecteurs[i], echantillons[i] = v, len(x)
    return vecteurs, echantillons


def encoder_partie(config: Dict[str, Any], encodeur: EncodeurS3, manifeste: List[Dict[str, Any]],
                   racine: Path, dossier: Path, journal=print) -> Dict[str, Any]:
    """Encode tous les fichiers manquants. Renvoie les mesures de vitesse de cette exécution."""
    reglages, controles = config["encodage"], config["controles"]
    taille = reglages["segments_par_fichier_npy"]
    paquets = [manifeste[i: i + taille] for i in range(0, len(manifeste), taille)]
    dossier.mkdir(parents=True, exist_ok=True)
    index = fichiers_faits(dossier, paquets, encodeur.modele_id)
    deja = len({l["fichier_npy"] for l in index})
    journal(f"{deja}/{len(paquets)} fichier(s) déjà faits")

    secondes_audio, secondes_calcul = 0.0, 0.0
    duree_restante = sum(e["duree_s"] for e in manifeste[len(index):])
    octets_par_seconde = None
    for numero in range(deja, len(paquets)):
        paquet = paquets[numero]
        depart = time.perf_counter()
        vecteurs, echantillons = encoder_paquet(encodeur, paquet, racine, reglages["secondes_audio_par_lot"],
                                                controles["tolerance_fin_wav_s"])
        ecoule = time.perf_counter() - depart
        duree_paquet = sum(e["duree_s"] for e in paquet)
        secondes_audio += duree_paquet
        secondes_calcul += ecoule

        matrice = np.concatenate(vecteurs, axis=0)
        octets_par_seconde = matrice.nbytes / duree_paquet
        libre = shutil.disk_usage(dossier).free
        marge = controles["marge_disque_go"] * 1e9
        estimation = octets_par_seconde * duree_restante
        if libre - estimation < marge:
            raise ErreurEncodage(
                f"espace disque insuffisant : {estimation / 1e9:.1f} Go encore attendus, {libre / 1e9:.1f} Go libres, "
                f"marge {controles['marge_disque_go']} Go"
            )
        nom = nom_fichier(numero)
        temporaire = dossier / (nom + ".tmp")
        with open(temporaire, "wb") as fichier:
            np.save(fichier, matrice)
        temporaire.replace(dossier / nom)

        ligne_debut = 0
        with open(dossier / "index.jsonl", "a", encoding="utf-8") as fichier:
            for entree, v, n in zip(paquet, vecteurs, echantillons):
                ligne = dict(entree, fichier_npy=nom, ligne_debut=ligne_debut, nb_vecteurs=int(v.shape[0]),
                             nb_echantillons=n,
                             encodeur=encodeur.modele_id, d_enc=encodeur.d_enc)
                fichier.write(json.dumps(ligne, ensure_ascii=False) + "\n")
                ligne_debut += int(v.shape[0])
            fichier.flush()
            os.fsync(fichier.fileno())
        duree_restante = max(0.0, duree_restante - duree_paquet)
        journal(f"{nom} : {len(paquet)} segments, {duree_paquet / 3600:.2f} h d'audio en {ecoule:.0f} s "
                f"({duree_paquet / ecoule:.0f} s d'audio/s), {matrice.nbytes / 1e9:.3f} Go ; "
                f"reste estimé {octets_par_seconde * duree_restante / 1e9:.1f} Go")
    return {
        "secondes_audio_encodees": round(secondes_audio, 1),
        "secondes_calcul": round(secondes_calcul, 1),
        "vitesse_s_audio_par_s": round(secondes_audio / secondes_calcul, 1) if secondes_calcul else None,
        "octets_par_heure_audio": round(octets_par_seconde * 3600) if octets_par_seconde else None,
    }


def lire_vecteurs(dossier: Path, ligne: Dict[str, Any]) -> np.ndarray:
    matrice = np.load(dossier / ligne["fichier_npy"], mmap_mode="r")
    return matrice[ligne["ligne_debut"]: ligne["ligne_debut"] + ligne["nb_vecteurs"]]


# ----------------------------------------------------------------------------
# Vérifications finales
# ----------------------------------------------------------------------------

def verifier_encodage(config: Dict[str, Any], encodeur: EncodeurS3, dossier: Path, racine: Path,
                      mesures: Dict[str, Any], graine: int) -> Dict[str, Any]:
    controles = config["controles"]
    index = lire_jsonl(dossier / "index.jsonl")
    fichiers = sorted({l["fichier_npy"] for l in index})
    lignes_fichiers, non_finis = 0, 0
    for nom in fichiers:
        matrice = np.load(dossier / nom, mmap_mode="r")
        lignes_fichiers += matrice.shape[0]
        for debut in range(0, matrice.shape[0], 100_000):
            non_finis += int((~np.isfinite(matrice[debut: debut + 100_000])).sum())
    somme_index = sum(l["nb_vecteurs"] for l in index)
    ecarts_formule = sum(
        1 for l in index
        if l["nb_vecteurs"] != encodeur.nb_vecteurs_attendus(l.get(
            "nb_echantillons", round(l["fin"] * FREQUENCE_ATTENDUE) - round(l["debut"] * FREQUENCE_ATTENDUE)))
    )

    reencodes = []
    for ligne in random.Random(graine).sample(index, min(controles["nb_segments_reencodes"], len(index))):
        extrait = charger_extrait(ligne, racine, tolerance_fin_s=controles["tolerance_fin_wav_s"])
        seul = encodeur.encoder_lot([extrait])[0].astype(np.float32)
        stocke = lire_vecteurs(dossier, ligne).astype(np.float32)
        meme_forme = seul.shape == stocke.shape
        ecart = float(np.abs(seul - stocke).max()) if meme_forme else None
        reencodes.append({
            "id": ligne["id"],
            "nb_vecteurs": ligne["nb_vecteurs"],
            "meme_forme": meme_forme,
            "ecart_max": ecart,
            "conforme": bool(meme_forme and np.allclose(seul, stocke, atol=controles["tolerance_absolue"],
                                                         rtol=controles["tolerance_relative"])),
        })

    duree_totale = sum(l["duree_s"] for l in index)
    taille_disque = sum((dossier / nom).stat().st_size for nom in fichiers)
    rapport = {
        "encodeur": encodeur.modele_id,
        "d_enc": encodeur.d_enc,
        "appareil": str(encodeur.appareil),
        "precision": str(encodeur.precision).replace("torch.", ""),
        "segments": len(index),
        "heures_audio": round(duree_totale / 3600, 3),
        "fichiers_npy": len(fichiers),
        "valeurs_non_finies": non_finis,
        "somme_nb_vecteurs_index": somme_index,
        "somme_lignes_fichiers": lignes_fichiers,
        "segments_hors_formule": ecarts_formule,
        "frequence_vecteurs_par_s": round(somme_index / duree_totale, 4) if duree_totale else None,
        "taille_disque_octets": taille_disque,
        "go_par_heure_audio": round(taille_disque / 1e9 / (duree_totale / 3600), 4) if duree_totale else None,
        "reencodage_seul": reencodes,
        "execution": mesures,
    }
    rapport["conforme"] = bool(
        non_finis == 0 and somme_index == lignes_fichiers and ecarts_formule == 0
        and all(r["conforme"] for r in reencodes)
    )
    with open(dossier / "rapport_encodage.json", "w", encoding="utf-8") as fichier:
        json.dump(rapport, fichier, ensure_ascii=False, indent=2)
    return rapport


# ----------------------------------------------------------------------------
# Mode essai et point d'entrée
# ----------------------------------------------------------------------------

def ecrire_audio_factice(racine: Path, nb_segments: int = 12, graine: int = 0) -> List[Dict[str, Any]]:
    """Un wav de bruit 16 kHz et un manifeste factice de segments de 1 à 6 s."""
    import soundfile as sf

    generateur = np.random.default_rng(graine)
    (racine / "wav").mkdir(parents=True, exist_ok=True)
    durees = [round(float(d), 2) for d in generateur.uniform(1.0, 6.0, nb_segments)]
    signal = (0.1 * generateur.standard_normal(int((sum(durees) + nb_segments) * FREQUENCE_ATTENDUE))).astype(np.float32)
    sf.write(str(racine / "wav" / "essai.wav"), signal, FREQUENCE_ATTENDUE, subtype="PCM_16")
    manifeste, debut = [], 0.5
    for numero, duree in enumerate(durees):
        manifeste.append({"id": f"essai_utt_{numero}", "episode": "essai", "wav": "wav/essai.wav",
                          "debut": round(debut, 2), "fin": round(debut + duree, 2), "duree_s": duree,
                          "texte_ar": "نص تجريبي", "partie": "train"})
        debut += duree + 1.0
    return manifeste


def executer(config: Dict[str, Any], encodeur: EncodeurS3, manifeste: List[Dict[str, Any]], racine: Path,
             dossier: Path, graine: int, journal=print) -> Dict[str, Any]:
    mesures = encoder_partie(config, encodeur, manifeste, racine, dossier, journal)
    return verifier_encodage(config, encodeur, dossier, racine, mesures, graine)


def construire_parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encodage S3 des segments MGB-2 (Cohere Transcribe Arabic).")
    parser.add_argument("--config", type=str, default=str(CHEMIN_CONFIG), help="Config YAML (encodage_s3.yaml).")
    parser.add_argument("--partie", choices=["train", "test"], default="train", help="Manifeste à encoder.")
    parser.add_argument("--essai", action="store_true", help="Petit encodeur aléatoire et audio factice, sans réseau.")
    return parser


def main(arguments: Optional[List[str]] = None) -> Dict[str, Any]:
    args = construire_parseur().parse_args(arguments)
    config = charger_config(args.config)
    graine = config.get("selection", {}).get("graine", 42)
    if args.essai:
        from factices_s3 import petit_modele_asr, petit_preparateur

        temporaire = Path(tempfile.mkdtemp(prefix="encoder_s3_essai_"))
        racine = temporaire / "mgb2"
        manifeste = ecrire_audio_factice(racine)
        config["encodage"].update(segments_par_fichier_npy=5, secondes_audio_par_lot=10)
        encodeur = EncodeurS3(petit_modele_asr(), petit_preparateur(), torch.device("cpu"), torch.float32, "essai")
        dossier = temporaire / "sortie" / args.partie
    else:
        racine = chemin_projet(config["sources"]["racine_mgb2"])
        dossier_sortie = chemin_projet(config["sortie"]["dossier"])
        chemin_manifeste = dossier_sortie / f"{args.partie}.jsonl"
        if not chemin_manifeste.is_file():
            raise ErreurEncodage(f"manifeste introuvable : {chemin_manifeste} (lancer code/donnees/manifeste_s3.py)")
        manifeste = lire_jsonl(chemin_manifeste)
        reglages = config["encodage"]
        encodeur = EncodeurS3.charger(reglages["modele_id"], choisir_appareil(reglages["appareil"]),
                                      PRECISIONS[reglages["precision"]])
        dossier = dossier_sortie / args.partie
    rapport = executer(config, encodeur, manifeste, racine, dossier, graine)
    print(json.dumps({k: v for k, v in rapport.items() if k != "reencodage_seul"}, ensure_ascii=False, indent=2))
    print(f"Rapport : {dossier / 'rapport_encodage.json'}")
    return rapport


if __name__ == "__main__":
    main()
