"""Projecteur S3 : vecteurs de l'encodeur audio -> embeddings du LLM.

Repris de ``projecteur_v1/speech_llm/projector.py`` (dépôt MLP). Le nombre de
convolutions de pas 2 devient un paramètre (``nb_reductions`` dans s3.yaml) :
avec Cohere Transcribe à 12,5 vecteurs/s, le LLM reçoit déjà le bon débit et
``nb_reductions`` vaut 0 (LayerNorm puis MLP seulement).
"""

from typing import List, Tuple

import torch
from torch import nn


def longueur_apres_convolution(n: int) -> int:
    """Sortie d'un Conv1d(noyau 3, pas 2, padding 1) : floor((n + 2 - 3) / 2) + 1."""
    return (n + 2 - 3) // 2 + 1


def nb_vecteurs_llm(nb_vecteurs_encodeur: int, nb_reductions: int) -> int:
    """Nombre d'embeddings audio donnés au LLM pour ``nb_vecteurs_encodeur`` vecteurs utiles."""
    n = nb_vecteurs_encodeur
    for _ in range(nb_reductions):
        n = longueur_apres_convolution(n)
    return n


class ProjecteurS3(nn.Module):
    """LayerNorm -> nb_reductions x (Conv1d pas 2 + GELU) -> MLP d_enc -> 2*d_llm -> d_llm."""

    def __init__(self, d_enc: int, d_llm: int, nb_reductions: int = 0):
        super().__init__()
        if nb_reductions < 0:
            raise ValueError(f"nb_reductions doit être >= 0 (reçu : {nb_reductions})")
        self.nb_reductions = nb_reductions
        self.norm = nn.LayerNorm(d_enc)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(d_enc, d_enc, kernel_size=3, stride=2, padding=1) for _ in range(nb_reductions)
        )
        self.activation = nn.GELU()
        self.mlp = nn.Sequential(nn.Linear(d_enc, 2 * d_llm), nn.GELU(), nn.Linear(2 * d_llm, d_llm))

    def forward(self, x: torch.Tensor, longueurs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x : (B, T, d_enc), longueurs : vecteurs utiles par exemple (B,).

        Renvoie (y, longueurs_sortie) : y (B, T', d_llm). Les positions au-delà de
        longueurs_sortie sont du rembourrage : l'appelant les jette.
        """
        h = self.norm(x)
        longueurs_courantes = [int(n) for n in longueurs]
        for convolution in self.convolutions:
            # Le rembourrage est remis à zéro avant chaque convolution (LayerNorm et la
            # convolution précédente l'ont rendu non nul) : un segment donne alors les
            # mêmes vecteurs utiles seul ou dans un lot.
            masque = torch.arange(h.shape[1], device=h.device)[None, :] < torch.tensor(longueurs_courantes, device=h.device)[:, None]
            h = h * masque.unsqueeze(-1).to(h.dtype)
            h = self.activation(convolution(h.transpose(1, 2))).transpose(1, 2)
            longueurs_courantes = [longueur_apres_convolution(n) for n in longueurs_courantes]
        y = self.mlp(h)
        sortie = torch.tensor([nb_vecteurs_llm(int(n), self.nb_reductions) for n in longueurs], device=y.device)
        return y, sortie.clamp(max=y.shape[1])

    def nb_parametres(self) -> int:
        return sum(p.numel() for p in self.parameters())


def rembourrer_vecteurs(vecteurs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Liste de (T_i, d_enc) -> lot (B, max T_i, d_enc) complété par des zéros, et longueurs (B,)."""
    longueurs = torch.tensor([v.shape[0] for v in vecteurs])
    lot = vecteurs[0].new_zeros(len(vecteurs), int(longueurs.max()), vecteurs[0].shape[1])
    for i, v in enumerate(vecteurs):
        lot[i, : v.shape[0]] = v
    return lot, longueurs
