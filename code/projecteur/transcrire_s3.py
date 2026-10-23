#!/usr/bin/env python3
"""Transcription S3 d'un fichier audio quelconque (temps 1 : arabe).

Chaîne : audio (tout format lu par librosa, ramené en mono 16 kHz) -> découpage en
morceaux de ``duree_max_morceau_s`` au plus, coupés au point le plus calme -> encodeur
Cohere Transcribe Arabic (gelé) -> projecteur entraîné -> Qwen2.5-1.5B-Instruct (gelé).

Le modèle a appris sur des segments de 5 à 25 s : le découpage garde chaque morceau
dans cet ordre de grandeur. Les réglages sont dans la section ``transcription`` de
configs/s3.yaml ; le projecteur est le meilleur de la section d'entraînement choisie.

    python code/projecteur/transcrire_s3.py --audio extrait.wav [--srt extrait.srt]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

RACINE_PROJET = Path(__file__).resolve().parents[2]
for _dossier in (RACINE_PROJET / "code" / "projecteur", RACINE_PROJET / "code" / "ecoute",
                 RACINE_PROJET / "code" / "demo"):
    if str(_dossier) not in sys.path:
        sys.path.insert(0, str(_dossier))

from audio_s3 import FREQUENCE_ATTENDUE  # noqa: E402
from encoder_s3 import EncodeurS3, former_lots  # noqa: E402
from entrainer_temps1 import PRECISIONS, chemin_projet, choisir_appareil, construire_modele  # noqa: E402
from modele_s3 import (  # noqa: E402
    LimiteParSegment, SpeechLLMS3, encoder_texte, limite_tokens, prefixe_chatml, retirer_boucles, suffixe_chatml,
)

CHEMIN_CONFIG_S3 = RACINE_PROJET / "configs" / "s3.yaml"
CHEMIN_CONFIG_ENCODAGE = RACINE_PROJET / "configs" / "encodage_s3.yaml"


def charger_audio(chemin) -> np.ndarray:
    """Fichier audio -> signal float32 mono à 16 kHz."""
    import librosa

    chemin = Path(chemin)
    if not chemin.is_file():
        raise FileNotFoundError(f"fichier audio introuvable : {chemin}")
    signal, _ = librosa.load(str(chemin), sr=FREQUENCE_ATTENDUE, mono=True)
    if signal.size == 0:
        raise ValueError(f"fichier audio vide : {chemin}")
    return signal.astype(np.float32)


def point_le_plus_calme(zone: np.ndarray, fenetre: int) -> int:
    """Position (en échantillons depuis le début de ``zone``) du milieu de la fenêtre la moins énergétique."""
    nb = len(zone) // fenetre
    energies = (zone[: nb * fenetre].reshape(nb, fenetre) ** 2).mean(axis=1)
    return int(np.argmin(energies)) * fenetre + fenetre // 2


def decouper(signal: np.ndarray, reglages: Dict[str, Any]) -> List[tuple]:
    """Bornes (début, fin) en échantillons de morceaux d'au plus ``duree_max_morceau_s``.

    Chaque coupure tombe sur la fenêtre la moins énergétique des ``recherche_coupure_s``
    dernières secondes du morceau ; un reste plus court que ``duree_min_morceau_s`` est
    rattaché au morceau précédent.
    """
    sr = FREQUENCE_ATTENDUE
    longueur_max = int(reglages["duree_max_morceau_s"] * sr)
    recherche = int(reglages["recherche_coupure_s"] * sr)
    fenetre = max(1, int(reglages["fenetre_energie_s"] * sr))
    longueur_min = int(reglages["duree_min_morceau_s"] * sr)
    bornes, debut = [], 0
    while len(signal) - debut > longueur_max:
        zone_debut = debut + longueur_max - recherche
        coupure = zone_debut + point_le_plus_calme(signal[zone_debut: debut + longueur_max], fenetre)
        bornes.append((debut, coupure))
        debut = coupure
    if bornes and len(signal) - debut < longueur_min:
        bornes[-1] = (bornes[-1][0], len(signal))
    else:
        bornes.append((debut, len(signal)))
    return bornes


class TranscripteurS3:
    """Encodeur Cohere gelé + SpeechLLMS3 avec le meilleur projecteur entraîné."""

    def __init__(self, encodeur: EncodeurS3, modele: SpeechLLMS3, reglages_generation: Dict[str, Any],
                 reglages_decoupage: Dict[str, Any], secondes_par_lot: float,
                 reglages_traduction: Optional[Dict[str, Any]] = None):
        self.encodeur = encodeur
        self.modele = modele.eval()
        self.generation = reglages_generation
        self.decoupage = reglages_decoupage
        self.secondes_par_lot = secondes_par_lot
        self.traduction = reglages_traduction

    @classmethod
    def charger(cls, section: str = "temps_1_50k", chemin_projecteur: Optional[Path] = None) -> "TranscripteurS3":
        """Vrais modèles (HF_TOKEN dans .env pour l'encodeur) ; aucun repli vers des modèles factices."""
        config_s3 = yaml.safe_load(CHEMIN_CONFIG_S3.read_text(encoding="utf-8"))
        config_encodage = yaml.safe_load(CHEMIN_CONFIG_ENCODAGE.read_text(encoding="utf-8"))
        reglages = dict(config_s3[section])
        chemin = Path(chemin_projecteur or chemin_projet(reglages["sortie"]) / "projecteur_meilleur.pt")
        if not chemin.is_file():
            raise FileNotFoundError(f"projecteur introuvable : {chemin} (lancer entrainer_temps1.py)")
        appareil = choisir_appareil()
        # Entraîné en bf16 sur Mac ; un T4 (avant Ampere) n'a pas de bf16 natif, on passe en fp16.
        if (appareil.type == "cuda" and reglages["precision_llm"] == "bfloat16"
                and torch.cuda.get_device_capability(appareil)[0] < 8):
            reglages["precision_llm"] = "float16"
        encodage = config_encodage["encodage"]
        encodeur = EncodeurS3.charger(encodage["modele_id"], appareil, PRECISIONS[encodage["precision"]])
        modele = construire_modele(config_s3, reglages, encodeur.d_enc, appareil)
        modele.projecteur.load_state_dict(torch.load(chemin, map_location=appareil))
        return cls(encodeur, modele, reglages, config_s3["transcription"], encodage["secondes_audio_par_lot"],
                   config_s3["traduction_texte"])

    def transcrire_signal(self, signal: np.ndarray) -> List[Dict[str, Any]]:
        """Signal 16 kHz -> [{"debut_s", "fin_s", "texte"}] dans l'ordre du temps."""
        return self._generer_signal(signal, "transcription")

    def traduire_signal(self, signal: np.ndarray) -> List[Dict[str, Any]]:
        """Traduction directe : audio projeté + « Traduis en français. », sans passer par le texte arabe.

        Le projecteur du temps 1 n'a appris que la transcription : c'est un essai sans
        réentraînement (le temps 2 apprendra cette tâche). Une traduction française prend
        environ ``ratio_tokens`` fois plus de tokens que la transcription arabe, d'où une
        limite par morceau plus large. Même nettoyage de sortie que la cascade.
        """
        if self.traduction is None:
            raise ValueError("réglages de traduction absents (section traduction_texte de s3.yaml)")
        segments = self._generer_signal(signal, "traduction", self.traduction["ratio_tokens"],
                                        self.traduction["max_nouveaux_tokens"])
        return [dict(s, texte=nettoyer_traduction(s["texte"])) for s in segments]

    @torch.no_grad()
    def _generer_signal(self, signal: np.ndarray, tache: str, facteur_limite: float = 1.0,
                        max_nouveaux_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        bornes = decouper(signal, self.decoupage)
        morceaux = [signal[a:b] for a, b in bornes]
        textes: List[Optional[str]] = [None] * len(morceaux)
        g = self.generation
        for lot in former_lots([len(m) / FREQUENCE_ATTENDUE for m in morceaux], self.secondes_par_lot):
            vecteurs = [torch.from_numpy(v.astype(np.float32)) for v in self.encodeur.encoder_lot([morceaux[i] for i in lot])]
            for debut in range(0, len(lot), g["taille_lot_generation"]):
                sous_lot = list(range(debut, min(debut + g["taille_lot_generation"], len(lot))))
                limites = [limite_tokens(vecteurs[j].shape[0], g["frequence_vecteurs_hz"],
                                         facteur_limite * g["tokens_par_seconde_max"], g["tokens_marge"])
                           for j in sous_lot]
                sorties = self.modele.generer([vecteurs[j] for j in sous_lot], tache,
                                              max_nouveaux_tokens=max_nouveaux_tokens or g["max_nouveaux_tokens"],
                                              limites_tokens=limites, repetition_penalty=g["repetition_penalty"])
                for j, texte in zip(sous_lot, sorties):
                    textes[lot[j]] = retirer_boucles(texte.strip(), g["repetitions_boucle"])
        return [{"debut_s": round(a / FREQUENCE_ATTENDUE, 2), "fin_s": round(b / FREQUENCE_ATTENDUE, 2), "texte": t}
                for (a, b), t in zip(bornes, textes)]

    def transcrire_fichier(self, chemin) -> List[Dict[str, Any]]:
        return self.transcrire_signal(charger_audio(chemin))

    @torch.no_grad()
    def traduire(self, textes: List[str]) -> List[str]:
        """Traduction texte arabe -> français par le même LLM, sans audio (cascade).

        Le LLM est celui du temps 1 : ses poids d'origine n'ont pas changé (seul le
        projecteur a appris), c'est donc Qwen2.5-1.5B-Instruct tel quel. Chaque texte est
        traduit seul ; un texte vide donne une traduction vide. Si de l'arabe reste dans la
        sortie, le texte est retraduit avec ``gabarit_secours``.
        """
        if self.traduction is None:
            raise ValueError("réglages de traduction absents (section traduction_texte de s3.yaml)")
        t = self.traduction
        a_traduire = [i for i, texte in enumerate(textes) if texte.strip()]
        traductions = [""] * len(textes)
        for i, fr in zip(a_traduire, self._generer_traductions([textes[i] for i in a_traduire], t["gabarit"])):
            traductions[i] = fr
        a_reprendre = [i for i in a_traduire if MOTIF_ARABE.search(traductions[i])]
        for i, fr in zip(a_reprendre, self._generer_traductions([textes[i] for i in a_reprendre], t["gabarit_secours"])):
            traductions[i] = fr
        return traductions

    @torch.no_grad()
    def traduire_avec_contexte(self, texte: str, contexte: List[str]) -> str:
        """Traduit ``texte`` en donnant au LLM les phrases arabes précédentes (``gabarit_contexte``).

        Sert au mode direct : les pronoms et les noms propres se lèvent mieux avec les deux phrases d'avant.
        Sans contexte, ou sans ``gabarit_contexte`` dans la config, c'est ``traduire`` ordinaire. Si de
        l'arabe reste dans la sortie, repli sur ``gabarit_secours`` sans contexte.
        """
        if self.traduction is None:
            raise ValueError("réglages de traduction absents (section traduction_texte de s3.yaml)")
        if not texte.strip():
            return ""
        t = self.traduction
        n = int(t.get("nb_phrases_contexte", 2))
        contexte = [c.strip() for c in contexte if c and c.strip()][-n:] if n > 0 else []   # [-0:] rendrait tout
        if not contexte or "gabarit_contexte" not in t:
            return self.traduire([texte])[0]
        # Les accolades du contexte ne doivent pas être prises pour des champs par str.format.
        bloc = "\n".join(contexte).replace("{", "{{").replace("}", "}}")
        gabarit = t["gabarit_contexte"].replace("{contexte}", bloc)
        fr = self._generer_traductions([texte], gabarit)[0]
        if MOTIF_ARABE.search(fr):
            fr = self._generer_traductions([texte], t["gabarit_secours"])[0]
        return fr

    def _generer_traductions(self, textes: List[str], gabarit: str) -> List[str]:
        from transformers import LogitsProcessorList

        t, modele = self.traduction, self.modele
        embeddings = modele.llm.get_input_embeddings()
        sorties: List[str] = []
        for debut in range(0, len(textes), t["taille_lot"]):
            lot = textes[debut: debut + t["taille_lot"]]
            sequences, limites = [], []
            for texte in lot:
                invite = prefixe_chatml(t["systeme"]) + suffixe_chatml(gabarit.format(texte=texte.strip()))
                sequences.append(embeddings(torch.tensor(encoder_texte(modele.tokenizer, invite), device=modele.device)))
                limites.append(int(t["ratio_tokens"] * len(encoder_texte(modele.tokenizer, texte))) + t["tokens_marge"])
            etiquettes = [torch.full((len(e),), -100, dtype=torch.long, device=modele.device) for e in sequences]
            entree, masque, _ = modele.rembourrer(sequences, etiquettes, "gauche")
            generes = modele.llm.generate(
                inputs_embeds=entree, attention_mask=masque, do_sample=False,
                max_new_tokens=min(t["max_nouveaux_tokens"], max(limites)), repetition_penalty=t["repetition_penalty"],
                logits_processor=LogitsProcessorList([LimiteParSegment(limites, modele.tokenizer.eos_token_id)]))
            for ids in generes:
                texte = modele.tokenizer.decode(ids, skip_special_tokens=True)
                sorties.append(nettoyer_traduction(retirer_boucles(texte.strip(), t["repetitions_boucle"])))
        return sorties


MOTIF_ARABE = re.compile(r"[\u0600-\u06FF]")
MOTIF_INTRODUCTION = re.compile(r"^(voici|pour traduire)[^:\n]{0,80}:\s*", re.IGNORECASE)


def nettoyer_traduction(texte: str) -> str:
    """Retire une introduction du type « Voici la traduction en français : » et les guillemets autour."""
    texte = MOTIF_INTRODUCTION.sub("", texte.strip()).strip()
    if len(texte) >= 2 and texte[0] in "\"«“" and texte[-1] in "\"»”":
        texte = texte[1:-1].strip()
    return texte


def transcripteur_factice(section: str = "temps_1_50k") -> TranscripteurS3:
    """Petits modèles aléatoires (tests et --essai) : la sortie n'a pas de sens, seule la chaîne est vérifiée."""
    from factices_s3 import D_ENC, TokenizerFactice, petit_llm, petit_modele_asr, petit_preparateur

    config_s3 = yaml.safe_load(CHEMIN_CONFIG_S3.read_text(encoding="utf-8"))
    reglages = dict(config_s3[section], max_nouveaux_tokens=4, taille_lot_generation=4)
    encodeur = EncodeurS3(petit_modele_asr(), petit_preparateur(), torch.device("cpu"), torch.float32, "essai")
    torch.manual_seed(0)
    modele = SpeechLLMS3.depuis_config(config_s3, petit_llm(), TokenizerFactice(), D_ENC)
    traduction = dict(config_s3["traduction_texte"], max_nouveaux_tokens=4)
    return TranscripteurS3(encodeur, modele, reglages, config_s3["transcription"], 120, traduction)


def construire_parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcription arabe S3 d'un fichier audio.")
    parser.add_argument("--audio", type=str, help="Fichier audio (wav, flac, mp3...).")
    parser.add_argument("--section", type=str, default="temps_1_50k", help="Section d'entraînement de s3.yaml.")
    parser.add_argument("--projecteur", type=str, help="Projecteur à utiliser (défaut : le meilleur de la section).")
    parser.add_argument("--srt", type=str, help="Écrire aussi les sous-titres dans ce fichier .srt.")
    parser.add_argument("--essai", action="store_true", help="Petits modèles aléatoires et bruit de 30 s.")
    return parser


def main(arguments: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    args = construire_parseur().parse_args(arguments)
    if args.essai:
        transcripteur = transcripteur_factice(args.section)
        signal = (0.1 * np.random.default_rng(0).standard_normal(30 * FREQUENCE_ATTENDUE)).astype(np.float32)
    else:
        if not args.audio:
            raise SystemExit("--audio est obligatoire hors mode --essai")
        transcripteur = TranscripteurS3.charger(args.section, args.projecteur)
        signal = charger_audio(args.audio)
    depart = time.perf_counter()
    segments = transcripteur.transcrire_signal(signal)
    duree = len(signal) / FREQUENCE_ATTENDUE
    for s in segments:
        print(f"[{s['debut_s']:7.2f} - {s['fin_s']:7.2f}] {s['texte']}")
    print(f"{duree:.1f} s d'audio transcrites en {time.perf_counter() - depart:.1f} s")
    if args.srt:
        from srt import sauvegarder_srt

        print(f"sous-titres : {sauvegarder_srt(segments, args.srt)}")
    return segments


if __name__ == "__main__":
    main()
