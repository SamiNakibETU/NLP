"""Modèles aléatoires minuscules pour les tests (aucun téléchargement)."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder

N_MELS = 16
MAX_SOURCE = 50  # -> 100 frames mel d'entrée (au lieu de 3000)
EOS = 1


class FakeTokenizer:
    """Tokenizer au niveau octet : id = octet + 2. 0 = rembourrage, 1 = EOS."""

    eos_token_id = EOS
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[b + 2 for b in text.encode("utf-8")])

    def decode(self, ids, skip_special_tokens=True):
        data = bytes(int(i) - 2 for i in ids if 2 <= int(i) < 258)  # ids hors octet ignorés
        return data.decode("utf-8", errors="replace")


def tiny_encoder() -> WhisperEncoder:
    cfg = WhisperConfig(num_mel_bins=N_MELS, d_model=32, encoder_layers=1, encoder_attention_heads=2,
                        encoder_ffn_dim=64, max_source_positions=MAX_SOURCE)
    return WhisperEncoder(cfg)


def tiny_llm() -> Qwen2ForCausalLM:
    cfg = Qwen2Config(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                      eos_token_id=EOS, pad_token_id=0, initializer_range=0.3)
    return Qwen2ForCausalLM(cfg)


def tiny_batch(batch: int = 2, seed: int = 0):
    """(features, mel_lengths) aléatoires : 100 frames mel, dont 60 et 100 utiles."""
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(batch, N_MELS, 2 * MAX_SOURCE, generator=g)
    lengths = torch.tensor([60, 100][:batch] if batch <= 2 else [100] * batch)
    return feats, lengths
