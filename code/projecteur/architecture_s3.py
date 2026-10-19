"""Regroupement temporel, projecteur audio-vers-LLM et assemblage de l'entrée pour S3."""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import yaml

RACINE_PROJET = Path(__file__).resolve().parents[2]
DOSSIER_ECOUTE = RACINE_PROJET / "code" / "ecoute"
if str(DOSSIER_ECOUTE) not in sys.path:
    sys.path.insert(0, str(DOSSIER_ECOUTE))

from encodeur import (  # noqa: E402
    extraire_vecteurs_sans_silence,
    lire_dimension_encodeur,
    lire_dimension_llm,
)

CHEMIN_CONFIG_S3 = RACINE_PROJET / "configs" / "s3.yaml"


def charger_config_s3(chemin_config: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Charge la configuration YAML du système S3."""
    cible = Path(chemin_config) if chemin_config else CHEMIN_CONFIG_S3
    if not cible.exists():
        raise FileNotFoundError(f"Configuration S3 introuvable : {cible}")
    with open(cible, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def regrouper_vecteurs_audio(
    vecteurs: torch.Tensor,
    k: int = 4,
) -> torch.Tensor:
    """Colle bout à bout `k` vecteurs voisins de l'encodeur.

    Entrée : `(T_utile, d_enc)` ou `(B, T_utile, d_enc)` (après retrait du silence).
    Si `T_utile` n'est pas un multiple de `k`, complète les derniers pas par des zéros
    pour conserver l'intégralité du signal parlé.
    Sortie : `(ceil(T_utile / k), k * d_enc)` ou `(B, ceil(T_utile / k), k * d_enc)`.
    """
    if k <= 0:
        raise ValueError(f"Le facteur de regroupement k doit être > 0 (reçu : {k})")

    entree_2d = vecteurs.ndim == 2
    if entree_2d:
        vecteurs = vecteurs.unsqueeze(0)
    elif vecteurs.ndim != 3:
        raise ValueError(f"Forme attendue 2D ou 3D, reçu : {tuple(vecteurs.shape)}")

    b, t_utile, d_enc = vecteurs.shape
    reste = t_utile % k
    if reste != 0:
        pad_len = k - reste
        padding = torch.zeros((b, pad_len, d_enc), dtype=vecteurs.dtype, device=vecteurs.device)
        vecteurs = torch.cat([vecteurs, padding], dim=1)
        t_utile = vecteurs.shape[1]

    t_groupe = t_utile // k
    regroupes = vecteurs.contiguous().view(b, t_groupe, k * d_enc)
    return regroupes.squeeze(0) if entree_2d else regroupes


class ProjecteurAudioLLM(nn.Module):
    """Pont MLP entre l'encodeur audio et l'espace d'embeddings du LLM.

    Architecture : Linear(k * d_enc -> d_llm) -> GELU -> Linear(d_llm -> d_llm).
    Les dimensions `d_enc` et `d_llm` sont lues dynamiquement depuis les configurations des modèles.
    """

    def __init__(self, d_enc: int, d_llm: int, k: int = 4):
        super().__init__()
        if d_enc <= 0 or d_llm <= 0 or k <= 0:
            raise ValueError(f"Dimensions invalides : d_enc={d_enc}, d_llm={d_llm}, k={k}")
        self.d_enc = int(d_enc)
        self.d_llm = int(d_llm)
        self.k = int(k)
        self.dim_entree = self.k * self.d_enc

        self.fc1 = nn.Linear(self.dim_entree, self.d_llm)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.d_llm, self.d_llm)

    @classmethod
    def depuis_configs(cls, config_encodeur: Any, config_llm: Any, k: int = 4) -> "ProjecteurAudioLLM":
        """Instancie le projecteur en lisant `d_enc` et `d_llm` dans les configurations fournies."""
        d_enc = lire_dimension_encodeur(config_encodeur)
        d_llm = lire_dimension_llm(config_llm)
        return cls(d_enc=d_enc, d_llm=d_llm, k=k)

    def nb_parametres(self) -> int:
        """Retourne le nombre total de paramètres entraînables du projecteur."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        vecteurs_encodeur: torch.Tensor,
        duree_reelle_s: Optional[float] = None,
        duree_fenetre_s: float = 30.0,
    ) -> torch.Tensor:
        """Retire le silence de complétion si `duree_reelle_s` est donné, regroupe par `k` et projette vers `d_llm`."""
        if duree_reelle_s is not None:
            vecteurs_utiles = extraire_vecteurs_sans_silence(
                vecteurs_encodeur,
                duree_reelle_s=duree_reelle_s,
                duree_fenetre_s=duree_fenetre_s,
            )
        else:
            vecteurs_utiles = vecteurs_encodeur

        regroupes = regrouper_vecteurs_audio(vecteurs_utiles, k=self.k)
        return self.fc2(self.act(self.fc1(regroupes)))


def assembler_entree_s3(
    vecteurs_audio_projetes: torch.Tensor,
    ids_debut_user: torch.Tensor,
    ids_consigne: torch.Tensor,
    ids_fin_user_debut_assistant: torch.Tensor,
    table_embeddings_llm: nn.Module,
    ids_reponse: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Construit la séquence `inputs_embeds` et le vecteur `labels` pour S3.

    Ordre strict :
    1. début du tour utilisateur ;
    2. vecteurs audio sortis du projecteur ;
    3. vecteurs de la consigne (table d'embeddings du LLM) ;
    4. fin du tour utilisateur et début du tour assistant ;
    5. pendant l'entraînement, vecteurs de la réponse attendue.

    Toutes les positions 1 à 4 portent l'étiquette `-100`. Seuls les tokens de la réponse (5)
    portent leurs identifiants cibles pour le calcul de l'erreur.
    """
    if vecteurs_audio_projetes.ndim == 3:
        if vecteurs_audio_projetes.shape[0] != 1:
            raise ValueError("assembler_entree_s3 attend un exemple unitaire (T_audio, d_llm) ou (1, T_audio, d_llm).")
        vecteurs_audio_projetes = vecteurs_audio_projetes.squeeze(0)

    device = vecteurs_audio_projetes.device
    ids_debut_user = ids_debut_user.to(device).view(-1)
    ids_consigne = ids_consigne.to(device).view(-1)
    ids_fin_user_debut_assistant = ids_fin_user_debut_assistant.to(device).view(-1)

    emb_debut_user = table_embeddings_llm(ids_debut_user)
    emb_consigne = table_embeddings_llm(ids_consigne)
    emb_fin_user = table_embeddings_llm(ids_fin_user_debut_assistant)

    blocs_embeds: List[torch.Tensor] = [
        emb_debut_user,
        vecteurs_audio_projetes,
        emb_consigne,
        emb_fin_user,
    ]

    nb_avant_reponse = (
        emb_debut_user.shape[0]
        + vecteurs_audio_projetes.shape[0]
        + emb_consigne.shape[0]
        + emb_fin_user.shape[0]
    )
    labels_list: List[int] = [-100] * nb_avant_reponse

    nb_reponse = 0
    if ids_reponse is not None and ids_reponse.numel() > 0:
        ids_reponse = ids_reponse.to(device).view(-1)
        emb_reponse = table_embeddings_llm(ids_reponse)
        blocs_embeds.append(emb_reponse)
        labels_list.extend(int(x) for x in ids_reponse.tolist())
        nb_reponse = int(ids_reponse.numel())

    inputs_embeds = torch.cat(blocs_embeds, dim=0)
    labels = torch.tensor(labels_list, dtype=torch.long, device=device)
    attention_mask = torch.ones(inputs_embeds.shape[0], dtype=torch.long, device=device)

    plages = {
        "debut_user": (0, int(emb_debut_user.shape[0])),
        "audio": (int(emb_debut_user.shape[0]), int(emb_debut_user.shape[0] + vecteurs_audio_projetes.shape[0])),
        "consigne": (
            int(emb_debut_user.shape[0] + vecteurs_audio_projetes.shape[0]),
            int(nb_avant_reponse - emb_fin_user.shape[0]),
        ),
        "fin_user_debut_assistant": (int(nb_avant_reponse - emb_fin_user.shape[0]), int(nb_avant_reponse)),
        "reponse": (int(nb_avant_reponse), int(nb_avant_reponse + nb_reponse)),
    }

    return {
        "inputs_embeds": inputs_embeds,
        "labels": labels,
        "attention_mask": attention_mask,
        "plages": plages,
        "nb_masques": nb_avant_reponse,
        "nb_supervises": nb_reponse,
    }
