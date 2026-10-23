"""Moteur réel : charge un checkpoint de `train.py` (projecteur ou cross-attention) et traduit de l'audio.

Le code du modèle vient du projet `speech_xattn` (il contient aussi la version projecteur) : donner son dossier à
`load_engine(code_dir=...)`. Les architectures sont relues depuis les arguments sauvegardés dans le checkpoint.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from engines import Engine, Result, Segment

SAMPLE_RATE = 16000


def subtitle_lines(text: str, start: float, end: float, max_chars: int = 42) -> list[tuple[float, float, str]]:
    """Découpe un texte en lignes de sous-titre d'au plus `max_chars` caractères, réparties dans [start, end]
    proportionnellement au nombre de caractères (on ne connaît pas l'alignement mot à mot)."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    total = sum(len(l) for l in lines) or 1
    out, t = [], start
    for l in lines:
        d = (end - start) * len(l) / total
        out.append((t, t + d, l))
        t += d
    return out


def windows(duration: float, size: float = 25.0, min_len: float = 0.4) -> list[tuple[float, float]]:
    """Fenêtres contiguës de `size` secondes (Whisper accepte 30 s au plus) ; la dernière trop courte est jetée."""
    out, t = [], 0.0
    while t < duration:
        end = min(t + size, duration)
        if end - t >= min_len:
            out.append((t, end))
        t += size
    return out


def librosa_loader(path: str) -> np.ndarray:
    import librosa
    return librosa.load(path, sr=SAMPLE_RATE, mono=True)[0]


class SpeechLLMEngine(Engine):
    available = True
    status = "Modèle entraîné"

    def __init__(self, name, label, blurb, model, feature_extractor, *, window=25.0, batch=4, max_new_tokens=128,
                 produces_arabic=False, device="cpu", loader=librosa_loader, extract=None):
        from speech_llm.audio import extract_features  # fourni par le dossier de code ajouté au sys.path
        self.name, self.label, self.blurb = name, label, blurb
        self.model, self.fe, self.window, self.batch = model.eval(), feature_extractor, window, batch
        self.max_new_tokens, self.produces_arabic, self.device, self.loader = max_new_tokens, produces_arabic, device, loader
        self.extract = extract or extract_features

    def run(self, path: str, task: str = "st", progress=None) -> Result:
        import torch
        t0 = time.time()
        wav = self.loader(path)
        duration = len(wav) / SAMPLE_RATE
        wins = windows(duration, self.window)
        segs: list[Segment] = []
        for k in range(0, len(wins), self.batch):
            group = wins[k:k + self.batch]
            chunks = [wav[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)] for a, b in group]
            feats, mel = self.extract(self.fe, chunks)
            with torch.no_grad(), torch.autocast(self.device if self.device != "cpu" else "cpu",
                                                 dtype=torch.float16, enabled=self.device == "cuda"):
                texts = self.model.generate(feats.to(self.device), mel, task, max_new_tokens=self.max_new_tokens)
            for (a, b), text in zip(group, texts):
                for s, e, line in subtitle_lines(text.strip(), a, b):
                    segs.append(Segment(s, e, fr=line) if task == "st" else Segment(s, e, ar=line))
            if progress:
                progress((k + len(group)) / len(wins), desc=f"Fenêtres traitées : {k + len(group)}/{len(wins)}")
        peak = None
        if self.device == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1e9
        return Result(engine=self.name, label=self.label, task=task, duration=duration, segments=segs,
                      seconds=time.time() - t0, audio_tokens=int(duration * 12.5), peak_mem_gb=peak,
                      produces_arabic=self.produces_arabic, is_demo=False)


def load_engine(kind: str, code_dir: str | Path, ckpt: str | Path, llm: str = "Qwen/Qwen2.5-1.5B-Instruct",
                device: str | None = None) -> SpeechLLMEngine:
    """kind : "proj" ou "xattn". Charge Whisper-large-v3, Qwen et les poids entraînés du checkpoint."""
    import torch
    sys.path.insert(0, str(code_dir))
    import train  # noqa: E402  (code de speech_xattn)
    from speech_llm.training import prepare_stage

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location="cpu")
    args = ck.get("args", {})
    model, _tok, fe = train.build_model(args.get("llm", llm), device, args.get("max_target_tokens", 96), kind,
                                        args.get("xattn_every", 4), args.get("mem_stride", 4))
    stage = 2 if any("lora_" in k for k in ck["trainable"]) else 1
    prepare_stage(model, stage)                      # recrée les LoRA avant de charger leurs poids
    train.load_trainable_state(model, ck["trainable"])
    label = "Projecteur" if kind == "proj" else "Cross-attention"
    blurb = {"proj": "Whisper gelé, projecteur à convolutions, Qwen2.5-1.5B.",
             "xattn": "Whisper gelé, blocs d'attention croisée dans Qwen2.5-1.5B."}[kind]
    return SpeechLLMEngine(kind, label, blurb, model, fe, device=device)
