"""Projecteur audio : sortie de l'encodeur Whisper (50 Hz) -> embeddings du LLM (12,5 Hz)."""
from __future__ import annotations

import torch
from torch import nn


def conv_out_len(n: int) -> int:
    """Longueur de sortie d'un Conv1d(noyau 3, pas 2, padding 1) : floor((n + 2 - 3) / 2) + 1."""
    return (n + 2 - 3) // 2 + 1


def audio_token_count(useful_encoder_frames: int) -> int:
    """Nombre de tokens audio donnés au LLM pour un nombre de frames utiles de l'encodeur."""
    return conv_out_len(conv_out_len(useful_encoder_frames))


class AudioProjector(nn.Module):
    """LayerNorm -> 2 x (Conv1d pas 2 + GELU) -> MLP enc_dim -> 2*llm_dim -> llm_dim."""

    def __init__(self, enc_dim: int, llm_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(enc_dim)
        self.conv1 = nn.Conv1d(enc_dim, enc_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(enc_dim, enc_dim, kernel_size=3, stride=2, padding=1)
        self.act = nn.GELU()
        self.mlp = nn.Sequential(nn.Linear(enc_dim, 2 * llm_dim), nn.GELU(), nn.Linear(2 * llm_dim, llm_dim))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x : (B, T, enc_dim), lengths : frames utiles par exemple (B,).

        Retourne (y, out_lengths) : y (B, T', llm_dim), out_lengths (B,) = tokens audio utiles.
        Les positions au-delà de out_lengths sont du rembourrage : l'appelant les jette.
        """
        h = self.norm(x).transpose(1, 2)
        h = self.act(self.conv1(h))
        h = self.act(self.conv2(h))
        y = self.mlp(h.transpose(1, 2))
        out = torch.tensor([audio_token_count(int(n)) for n in lengths], device=y.device)
        return y, out.clamp(max=y.shape[1])
