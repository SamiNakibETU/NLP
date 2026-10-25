"""Chaîne S3 : vecteurs d'encodeur précalculés -> projecteur -> LLM décodeur.

Repris de ``projecteur_v1/speech_llm/model.py`` (dépôt MLP). L'entrée n'est plus
un lot de features Whisper mais la liste des vecteurs utiles de chaque segment,
``(T_i, d_enc)``, calculés une fois pour toutes par ``code/ecoute/encoder_s3.py``.
L'encodeur n'est donc ni chargé ni appelé ici.

Séquence donnée au LLM :
``[préfixe système + début user][audio][consigne + début assistant][cible + EOS]``,
labels à -100 partout sauf sur la cible et l'EOS.
"""

import math
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from projecteur_s3 import ProjecteurS3, rembourrer_vecteurs

# Tâche -> clé de la consigne dans la section `consignes` de s3.yaml.
CLES_CONSIGNES = {"transcription": "temps_1_transcription", "traduction": "temps_2_traduction"}


def prefixe_chatml(systeme: str) -> str:
    return f"<|im_start|>system\n{systeme}<|im_end|>\n<|im_start|>user\n"


def suffixe_chatml(consigne: str) -> str:
    return f"{consigne}<|im_end|>\n<|im_start|>assistant\n"


def encoder_texte(tokenizer, texte: str) -> List[int]:
    return list(tokenizer(texte, add_special_tokens=False).input_ids)


class SpeechLLMS3(nn.Module):
    """``consignes`` : {tâche: consigne}, par exemple {"transcription": "Transcris en arabe."}."""

    def __init__(self, llm: nn.Module, tokenizer, d_enc: int, nb_reductions: int,
                 systeme: str, consignes: Dict[str, str], max_tokens_cible: int = 256):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.systeme = systeme
        self.consignes = dict(consignes)
        self.max_tokens_cible = max_tokens_cible
        self.projecteur = ProjecteurS3(d_enc, llm.get_input_embeddings().embedding_dim, nb_reductions)

    @classmethod
    def depuis_config(cls, config_s3: Dict, llm: nn.Module, tokenizer, d_enc: int) -> "SpeechLLMS3":
        """Construit le modèle à partir de configs/s3.yaml ; ``d_enc`` vient de l'index des vecteurs."""
        consignes = config_s3["consignes"]
        return cls(llm, tokenizer, d_enc,
                   nb_reductions=config_s3["briques"]["nb_reductions"],
                   systeme=consignes["systeme"],
                   consignes={tache: consignes[cle] for tache, cle in CLES_CONSIGNES.items()},
                   max_tokens_cible=config_s3["briques"]["max_tokens_cible"])

    @property
    def device(self):
        return self.projecteur.norm.weight.device

    def embeddings_audio(self, vecteurs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Embeddings audio utiles de chaque segment (rembourrage jeté) : liste de (n_i, d_llm)."""
        lot, longueurs = rembourrer_vecteurs([v.to(self.device, torch.float32) for v in vecteurs])
        projetes, longueurs_sortie = self.projecteur(lot, longueurs)
        return [projetes[i, : int(n)] for i, n in enumerate(longueurs_sortie)]

    def construire_sequences(self, audio: List[torch.Tensor], tache: str,
                             cibles: Optional[List[str]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if tache not in self.consignes:
            raise KeyError(f"tâche inconnue : {tache} (connues : {sorted(self.consignes)})")
        embeddings = self.llm.get_input_embeddings()
        appareil = audio[0].device
        prefixe = torch.tensor(encoder_texte(self.tokenizer, prefixe_chatml(self.systeme)), device=appareil)
        suffixe = torch.tensor(encoder_texte(self.tokenizer, suffixe_chatml(self.consignes[tache])), device=appareil)
        sequences, labels = [], []
        for i, a in enumerate(audio):
            morceaux = [embeddings(prefixe), a.to(embeddings.weight.dtype), embeddings(suffixe)]
            etiquettes = [-100] * (len(prefixe) + len(a) + len(suffixe))
            if cibles is not None:
                cible = encoder_texte(self.tokenizer, cibles[i])[: self.max_tokens_cible - 1]
                cible = cible + [self.tokenizer.eos_token_id]
                morceaux.append(embeddings(torch.tensor(cible, device=appareil)))
                etiquettes += cible
            sequences.append(torch.cat(morceaux))
            labels.append(torch.tensor(etiquettes, device=appareil))
        return sequences, labels

    @staticmethod
    def rembourrer(sequences, labels, cote: str):
        """Rembourre le lot : cote="droite" pour l'entraînement, "gauche" pour la génération."""
        if cote not in ("droite", "gauche"):
            raise ValueError(f"cote doit valoir 'droite' ou 'gauche' (reçu : {cote})")
        b, longueur, largeur = len(sequences), max(len(s) for s in sequences), sequences[0].shape[-1]
        embeddings = sequences[0].new_zeros(b, longueur, largeur)
        masque = torch.zeros(b, longueur, dtype=torch.long, device=embeddings.device)
        etiquettes = torch.full((b, longueur), -100, dtype=torch.long, device=embeddings.device)
        for i, (s, l) in enumerate(zip(sequences, labels)):
            n = len(s)
            tranche = slice(0, n) if cote == "droite" else slice(longueur - n, longueur)
            embeddings[i, tranche], masque[i, tranche], etiquettes[i, tranche] = s, 1, l
        return embeddings, masque, etiquettes

    def forward(self, vecteurs: List[torch.Tensor], tache: str, cibles: List[str]):
        """Perte d'entropie croisée sur la cible et l'EOS seulement.

        La tête de sortie (vocabulaire de 151 936 pour Qwen2.5) n'est appliquée qu'aux
        positions qui prédisent un token de la cible, pas à toute la séquence : même
        perte que ``llm(labels=...)``, pour une fraction de la mémoire et du calcul.
        """
        audio = self.embeddings_audio(vecteurs)
        sequences, labels = self.construire_sequences(audio, tache, cibles)
        embeddings, masque, etiquettes = self.rembourrer(sequences, labels, "droite")
        base = self.llm.get_base_model() if hasattr(self.llm, "get_base_model") else self.llm
        etats = base.model(inputs_embeds=embeddings, attention_mask=masque).last_hidden_state
        a_predire = etiquettes[:, 1:] != -100  # la position t prédit le token t + 1
        logits = base.lm_head(etats[:, :-1][a_predire]).float()
        perte = nn.functional.cross_entropy(logits, etiquettes[:, 1:][a_predire])
        return SimpleNamespace(loss=perte, logits=logits)

    @torch.no_grad()
    def generer(self, vecteurs: List[torch.Tensor], tache: str, max_nouveaux_tokens: int = 128,
                limites_tokens: Optional[List[int]] = None, **kw) -> List[str]:
        """Décodage glouton. ``limites_tokens`` : longueur maximale propre à chaque segment
        (au-delà, l'EOS est imposé), en plus de ``max_nouveaux_tokens`` pour tout le lot."""
        audio = self.embeddings_audio(vecteurs)
        sequences, labels = self.construire_sequences(audio, tache, None)
        embeddings, masque, _ = self.rembourrer(sequences, labels, "gauche")
        if limites_tokens is not None:
            from transformers import LogitsProcessorList

            kw["logits_processor"] = LogitsProcessorList([LimiteParSegment(limites_tokens, self.tokenizer.eos_token_id)])
            max_nouveaux_tokens = min(max_nouveaux_tokens, max(limites_tokens))
        sortie = self.llm.generate(inputs_embeds=embeddings, attention_mask=masque,
                                   max_new_tokens=max_nouveaux_tokens, do_sample=False, **kw)
        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in sortie]


class LimiteParSegment:
    """Impose l'EOS au segment i dès qu'il a produit ``limites[i]`` tokens.

    Avec ``inputs_embeds``, les ``input_ids`` vus pendant la génération ne contiennent que
    les tokens produits : leur longueur est le nombre de tokens déjà générés.
    """

    def __init__(self, limites: List[int], eos_token_id: int):
        self.limites = torch.tensor(limites)
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        atteint = (input_ids.shape[1] >= self.limites.to(scores.device))
        if atteint.any():
            scores[atteint] = torch.finfo(scores.dtype).min
            scores[atteint, self.eos_token_id] = 0.0
        return scores


def limite_tokens(nb_vecteurs: int, frequence_hz: float, tokens_par_seconde: float, marge: int) -> int:
    """Longueur maximale de transcription pour un audio de ``nb_vecteurs`` vecteurs."""
    return int(math.ceil(tokens_par_seconde * nb_vecteurs / frequence_hz)) + marge


def retirer_boucles(texte: str, repetitions_min: int = 4, n_max: int = 8) -> str:
    """Ramène à une seule occurrence toute suite de 1 à ``n_max`` mots répétée au moins
    ``repetitions_min`` fois de suite (boucles de génération, par exemple « بسم الله » répété)."""
    if repetitions_min <= 0:
        return texte
    mots = texte.split()
    i, sortie = 0, []
    while i < len(mots):
        for n in range(1, min(n_max, len(mots) - i) + 1):
            motif = mots[i: i + n]
            k = 1
            while mots[i + k * n: i + (k + 1) * n] == motif:
                k += 1
            if k >= repetitions_min:
                sortie.extend(motif)
                i += k * n
                break
        else:
            sortie.append(mots[i])
            i += 1
    return " ".join(sortie)
