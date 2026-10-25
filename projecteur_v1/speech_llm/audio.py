"""Chargement audio et features Whisper."""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
MAX_SECONDS = 30.0
HOP = 160  # échantillons par frame mel (10 ms)


def load_segment(entry: dict, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Charge l'audio d'une ligne du manifeste (mono, 16 kHz, 30 s maximum).

    Si `start` et `end` sont présents, seul ce passage est lu (offset + duration).
    """
    import librosa

    start, end = entry.get("start"), entry.get("end")
    if start is not None and end is not None:
        duration = min(float(end) - float(start), MAX_SECONDS)
        wav, _ = librosa.load(entry["audio"], sr=sr, mono=True, offset=float(start), duration=duration)
    else:
        wav, _ = librosa.load(entry["audio"], sr=sr, mono=True, duration=MAX_SECONDS)
    return wav[: int(MAX_SECONDS * sr)]


def useful_mel_frames(n_samples: int, total_frames: int) -> int:
    """Frames mel utiles (hors rembourrage) = échantillons // 160."""
    return min(n_samples // HOP, total_frames)


def useful_encoder_frames(mel_frames: int) -> int:
    """Frames utiles en sortie d'encodeur Whisper (conv de pas 2) = (mel + 1) // 2."""
    return (mel_frames + 1) // 2


def extract_features(feature_extractor, wavs: list[np.ndarray]):
    """Features Whisper rembourrées à 30 s, et frames mel utiles par exemple.

    Retourne (features (B, n_mels, 3000), mel_lengths (B,) en torch).
    """
    import torch

    out = feature_extractor(wavs, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    feats = out.input_features
    lengths = torch.tensor([useful_mel_frames(len(w), feats.shape[-1]) for w in wavs])
    return feats, lengths
