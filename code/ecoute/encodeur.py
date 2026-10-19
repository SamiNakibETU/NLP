"""Appel des modèles d'écoute et extraction de l'encodeur pour S3 (sections 3.1 et 3.3).

Principes :
- Aucune dimension n'est écrite en dur : `d_enc` est lu depuis la configuration de l'encodeur
  (`d_model` ou `hidden_size`), et `d_llm` depuis `config.json` du LLM (`hidden_size`).
- Pour Whisper : entrée de 30 secondes complétée par du silence -> sortie de 1 500 vecteurs
  (soit 50 vecteurs par seconde). Seuls les vecteurs correspondant à la durée réelle du son
  sont gardés avant le regroupement (le silence de complétion est retiré).
"""

import math
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


def lire_dimension_encodeur(config_ou_modele: Any) -> int:
    """Lit dynamiquement la dimension `d_enc` dans la configuration du modèle d'écoute."""
    cfg = getattr(config_ou_modele, "config", config_ou_modele)
    if isinstance(cfg, dict):
        for cle in ("d_model", "hidden_size", "encoder_hidden_size", "output_dim"):
            if cle in cfg and isinstance(cfg[cle], int) and cfg[cle] > 0:
                return int(cfg[cle])
        if "audio_config" in cfg and isinstance(cfg["audio_config"], dict):
            return lire_dimension_encodeur(cfg["audio_config"])
        if "thinker_config" in cfg and isinstance(cfg["thinker_config"], dict):
            return lire_dimension_encodeur(cfg["thinker_config"])
    else:
        for attr in ("d_model", "hidden_size", "encoder_hidden_size", "output_dim"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return int(val)
        for sous_attr in ("audio_config", "thinker_config"):
            sous_cfg = getattr(cfg, sous_attr, None)
            if sous_cfg is not None:
                return lire_dimension_encodeur(sous_cfg)
    raise ValueError(f"Impossible de déterminer la dimension d_enc depuis {cfg!r}")


def lire_dimension_llm(config_ou_modele: Any) -> int:
    """Lit dynamiquement la dimension `hidden_size` (`d_llm`) depuis la configuration du LLM."""
    cfg = getattr(config_ou_modele, "config", config_ou_modele)
    if isinstance(cfg, dict):
        if "hidden_size" in cfg and isinstance(cfg["hidden_size"], int) and cfg["hidden_size"] > 0:
            return int(cfg["hidden_size"])
        if "text_config" in cfg and isinstance(cfg["text_config"], dict):
            return lire_dimension_llm(cfg["text_config"])
        if "thinker_config" in cfg and isinstance(cfg["thinker_config"], dict):
            return lire_dimension_llm(cfg["thinker_config"])
    else:
        val = getattr(cfg, "hidden_size", None)
        if isinstance(val, int) and val > 0:
            return int(val)
        for sous_attr in ("text_config", "thinker_config"):
            sous_cfg = getattr(cfg, sous_attr, None)
            if sous_cfg is not None:
                return lire_dimension_llm(sous_cfg)
    raise ValueError(f"Impossible de déterminer hidden_size (d_llm) depuis {cfg!r}")


def calculer_nb_vecteurs_utiles(
    duree_reelle_s: float,
    nb_total_vecteurs: int = 1500,
    duree_fenetre_s: float = 30.0,
) -> int:
    """Calcule le nombre de vecteurs d'encodeur correspondant à la durée réelle du son.

    Pour Whisper : 1 500 vecteurs pour 30,0 s (soit 50 vecteurs/s).
    Un son réel de 6,0 s correspond aux 300 premiers vecteurs ; les 1 200 vecteurs
    suivants (silence de complétion) sont écartés.
    """
    if duree_reelle_s <= 0:
        raise ValueError(f"duree_reelle_s doit être > 0 (reçu : {duree_reelle_s})")
    if nb_total_vecteurs <= 0 or duree_fenetre_s <= 0:
        raise ValueError("nb_total_vecteurs et duree_fenetre_s doivent être > 0")
    proportion = min(1.0, float(duree_reelle_s) / float(duree_fenetre_s))
    nb_utiles = max(1, int(math.ceil(proportion * nb_total_vecteurs)))
    return min(nb_total_vecteurs, nb_utiles)


def extraire_vecteurs_sans_silence(
    vecteurs_encodeur: torch.Tensor,
    duree_reelle_s: float,
    duree_fenetre_s: float = 30.0,
) -> torch.Tensor:
    """Tronque la sortie de l'encodeur `(B, T_enc, d_enc)` ou `(T_enc, d_enc)` à la durée réelle du son."""
    if vecteurs_encodeur.ndim == 2:
        t_total = vecteurs_encodeur.shape[0]
        nb_utiles = calculer_nb_vecteurs_utiles(duree_reelle_s, t_total, duree_fenetre_s)
        return vecteurs_encodeur[:nb_utiles, :]
    if vecteurs_encodeur.ndim == 3:
        t_total = vecteurs_encodeur.shape[1]
        nb_utiles = calculer_nb_vecteurs_utiles(duree_reelle_s, t_total, duree_fenetre_s)
        return vecteurs_encodeur[:, :nb_utiles, :]
    raise ValueError(f"Dimensions inattendues pour vecteurs_encodeur : { tuple(vecteurs_encodeur.shape) }")


class ExtracteurEncodeurAudio:
    """Encapsule l'encodeur d'un modèle d'écoute (ex. Whisper) pour produire les vecteurs sans silence."""

    def __init__(
        self,
        modele_id: str = "openai/whisper-large-v3",
        duree_fenetre_s: float = 30.0,
        mode_factice: bool = False,
        d_enc_factice: int = 1280,
        debit_vecteurs_hz: float = 50.0,
        local_files_only: bool = False,
    ):
        self.modele_id = modele_id
        self.duree_fenetre_s = float(duree_fenetre_s)
        self.mode_factice = mode_factice
        self._debit_vecteurs_hz = float(debit_vecteurs_hz)
        self._processor = None
        self._encoder = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        if mode_factice:
            self.d_enc = int(d_enc_factice)
        else:
            from transformers import WhisperFeatureExtractor, WhisperModel

            self._processor = WhisperFeatureExtractor.from_pretrained(
                modele_id, local_files_only=local_files_only
            )
            modele_complet = WhisperModel.from_pretrained(
                modele_id, local_files_only=local_files_only
            )
            self._encoder = modele_complet.encoder.to(self._device).eval()
            for param in self._encoder.parameters():
                param.requires_grad = False
            self.d_enc = lire_dimension_encodeur(self._encoder.config)

    @torch.no_grad()
    def extraire(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sr: int = 16000,
        retirer_silence_padding: bool = True,
    ) -> Dict[str, Union[torch.Tensor, int, float]]:
        """Transforme un signal audio 16 kHz en suite de vecteurs `(T_utile, d_enc)` sans gradient."""
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy().astype(np.float32)
        else:
            audio_np = np.asarray(audio, dtype=np.float32)

        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=-1)

        duree_reelle_s = max(0.02, len(audio_np) / float(sr))

        if self.mode_factice:
            t_total = int(round(self.duree_fenetre_s * self._debit_vecteurs_hz))
            gen = torch.Generator().manual_seed(int(len(audio_np) % 10000))
            vecteurs_complets = torch.randn((1, t_total, self.d_enc), generator=gen, dtype=torch.float32)
        else:
            inputs = self._processor(
                audio_np,
                sampling_rate=sr,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            input_features = inputs.input_features.to(self._device)
            sortie_enc = self._encoder(input_features)
            vecteurs_complets = sortie_enc.last_hidden_state

        if retirer_silence_padding:
            vecteurs_utiles = extraire_vecteurs_sans_silence(
                vecteurs_complets,
                duree_reelle_s=duree_reelle_s,
                duree_fenetre_s=self.duree_fenetre_s,
            )
        else:
            vecteurs_utiles = vecteurs_complets

        return {
            "vecteurs": vecteurs_utiles.squeeze(0),
            "nb_vecteurs_complets": int(vecteurs_complets.shape[1]),
            "nb_vecteurs_utiles": int(vecteurs_utiles.shape[1]),
            "d_enc": self.d_enc,
            "duree_reelle_s": round(duree_reelle_s, 3),
        }
