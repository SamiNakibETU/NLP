"""Moteurs chargés à la demande : baseline S0, SeamlessM4T v2 et S3 (projecteur).

Un seul groupe de modèles reste en mémoire à la fois (le T4 de Colab fait 16 Go) : demander un autre moteur
décharge le précédent. L'interface et l'API ne voient que des `Engine` ordinaires (`name`, `label`, `run`).
"""
from __future__ import annotations

import gc
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from engines import Engine, Result, Segment
from model_engine import SAMPLE_RATE, subtitle_lines, windows

SEAMLESS_ID = "facebook/seamless-m4t-v2-large"


class Pool:
    """Garde au plus un groupe de modèles chargé ; `get` décharge les autres avant de charger."""

    def __init__(self):
        self._lock = threading.Lock()
        self._group: str | None = None
        self._engines: dict[str, Engine] = {}

    def get(self, group: str, name: str, build) -> Engine:
        with self._lock:
            if self._group != group:
                self._engines = {}
                self._group = None
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                except ImportError:
                    pass
                self._engines, self._group = build(), group
            return self._engines[name]

    @property
    def loaded(self) -> str | None:
        return self._group


class LazyEngine(Engine):
    available = True
    status = "Chargé à la demande"

    def __init__(self, pool: Pool, group: str, name: str, label: str, blurb: str, build, produces_arabic: bool):
        self.pool, self.group, self.name, self.label, self.blurb = pool, group, name, label, blurb
        self.build, self.produces_arabic = build, produces_arabic

    def run(self, path: str, task: str = "st", progress=None) -> Result:
        if progress:
            progress(0.02, desc=f"Chargement de {self.label}")
        return self.pool.get(self.group, self.name, self.build).run(path, task, progress=progress)


def _load_wav(path: str) -> np.ndarray:
    import librosa
    return librosa.load(path, sr=SAMPLE_RATE, mono=True)[0]


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _peak_mem() -> float | None:
    try:
        import torch
        return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    except ImportError:
        return None


# ---------------------------------------------------------------- S0
class S0Engine(Engine):
    """Baseline : VAD, Whisper small, Opus-MT (code `code/demo/systeme_s0.py` du dépôt de l'équipe)."""
    name, label = "s0", "Baseline S0 (Whisper small + Opus-MT)"
    blurb = "Sans entraînement : transcription arabe par Whisper small, puis traduction du texte par Opus-MT."
    available, status, produces_arabic = True, "Chargé à la demande", True

    def __init__(self, repo_dir: str | Path):
        demo = Path(repo_dir) / "code" / "demo"
        if not (demo / "systeme_s0.py").is_file():
            raise FileNotFoundError(f"{demo}/systeme_s0.py introuvable : donne la racine du dépôt de l'équipe")
        sys.path.insert(0, str(demo))
        import systeme_s0
        self.mod = systeme_s0
        self.s0 = systeme_s0.SystemeS0()
        self.s0._charger_modeles_reels()

    def run(self, path: str, task: str = "st", progress=None) -> Result:
        t0 = time.time()
        audio, sr = self.mod.charger_audio_mono_16k(path)
        cfg = self.s0.config.get("vad", {})
        tranches = self.mod.decouper_audio_vad(
            audio, sr=sr, duree_max_s=float(cfg.get("duree_max_segment_s", 30.0)),
            silence_min_s=float(cfg.get("silence_min_coupure_s", 0.5)),
            seuil_energie_db=float(cfg.get("seuil_energie_db", -38.0)))
        segs: list[Segment] = []
        for i, (a, b, chunk) in enumerate(tranches, 1):
            ar = self.s0.transcrire_segment(chunk, sr=sr)
            fr = self.s0.traduire_texte(ar) if task == "st" else ""
            for j, (s, e, line) in enumerate(subtitle_lines(fr if task == "st" else ar, a, b)):
                segs.append(Segment(s, e, fr=line, ar=ar if j == 0 else None) if task == "st"
                            else Segment(s, e, ar=line))
            if progress:
                progress(i / max(1, len(tranches)), desc=f"Segments traités : {i}/{len(tranches)}")
        return Result(engine=self.name, label=self.label, task=task, duration=len(audio) / sr, segments=segs,
                      seconds=time.time() - t0, peak_mem_gb=_peak_mem(), produces_arabic=True)


# ---------------------------------------------------------------- Seamless
class SeamlessEngine(Engine):
    """SeamlessM4T v2 large : audio arabe -> texte français en un seul modèle, sans texte arabe intermédiaire."""
    name, label = "seamless", "SeamlessM4T v2 (direct)"
    blurb = "Un seul modèle pré-entraîné de Meta : le son arabe donne directement le texte français."
    available, status, produces_arabic = True, "Chargé à la demande", False

    def __init__(self, model_id: str = SEAMLESS_ID, beams: int = 5, window: float = 25.0):
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2Model
        self.device = _device()
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        try:                                   # transformers 5 : `dtype` ; versions 4 : `torch_dtype`
            model = SeamlessM4Tv2Model.from_pretrained(model_id, dtype=self.dtype)
        except TypeError:
            model = SeamlessM4Tv2Model.from_pretrained(model_id, torch_dtype=self.dtype)
        self.model = model.to(self.device).eval()
        self.beams, self.window = beams, window

    def _generate(self, chunk: np.ndarray, tgt: str) -> str:
        import torch
        try:
            inputs = self.processor(audio=chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        except (TypeError, ValueError):        # transformers < 4.47 n'accepte que `audios`
            inputs = self.processor(audios=chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {k: (v.to(self.device, dtype=self.dtype) if torch.is_floating_point(v) else v.to(self.device))
                  for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(**inputs, tgt_lang=tgt, generate_speech=False, text_num_beams=self.beams)
        ids = out[0] if isinstance(out, (tuple, list)) else getattr(out, "sequences", out)
        return self.processor.decode(ids[0].tolist() if ids.ndim > 1 else ids.tolist(), skip_special_tokens=True).strip()

    def run(self, path: str, task: str = "st", progress=None) -> Result:
        t0 = time.time()
        wav = _load_wav(path)
        duration = len(wav) / SAMPLE_RATE
        wins = windows(duration, self.window)
        tgt = "fra" if task == "st" else "arb"
        segs: list[Segment] = []
        for i, (a, b) in enumerate(wins, 1):
            text = self._generate(wav[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)], tgt)
            for s, e, line in subtitle_lines(text, a, b):
                segs.append(Segment(s, e, fr=line) if task == "st" else Segment(s, e, ar=line))
            if progress:
                progress(i / len(wins), desc=f"Fenêtres traitées : {i}/{len(wins)}")
        return Result(engine=self.name, label=self.label, task=task, duration=duration, segments=segs,
                      seconds=time.time() - t0, peak_mem_gb=_peak_mem(), produces_arabic=False)


# ---------------------------------------------------------------- registre
def build_lazy_engines(nlp_repo: str, s3_ckpt: str | None = None, only: set[str] | None = None) -> dict[str, Engine]:
    """Moteurs à la demande. `nlp_repo` : racine du dépôt de l'équipe (branche benchmarking).
    `only` : sous-ensemble de {"s0", "seamless", "s3", "s3_direct"} ; par défaut s0, seamless et s3."""
    pool = Pool()
    want = only or {"s0", "seamless", "s3"}
    out: dict[str, Engine] = {}
    if "s0" in want:
        out["s0"] = LazyEngine(pool, "s0", "s0", S0Engine.label, S0Engine.blurb,
                               lambda: {"s0": S0Engine(nlp_repo)}, True)
    if "seamless" in want:
        out["seamless"] = LazyEngine(pool, "seamless", "seamless", SeamlessEngine.label, SeamlessEngine.blurb,
                                     lambda: {"seamless": SeamlessEngine()}, False)
    if want & {"s3", "s3_direct"}:
        import s3_engine

        def build_s3():
            return s3_engine.load_s3_engines(nlp_repo, s3_ckpt)
        # Le temps 2 n'est pas prévu : le français direct recopie l'arabe. La solution S3 est donc la cascade
        # (encodeur + projecteur + Qwen pour la transcription, puis Qwen pour la traduction du texte), nommée `s3`.
        if "s3" in want:
            out["s3"] = LazyEngine(pool, "s3", "s3_cascade", "S3 (encodeur + projecteur + décodeur)",
                                   s3_engine.BLURB["cascade"], build_s3, True)
        if "s3_direct" in want:                      # essai seulement, à demander explicitement
            out["s3_direct"] = LazyEngine(pool, "s3", "s3", "S3, français direct (essai)",
                                          s3_engine.BLURB["direct"], build_s3, False)
    return out


def engines_from_env() -> dict[str, Engine]:
    repo = os.environ.get("PORTEPAROLE_NLP_REPO") or os.environ.get("PORTEPAROLE_S3_REPO")
    if not repo:
        return {}
    only = {x.strip() for x in os.environ.get("PORTEPAROLE_ENGINES", "").split(",") if x.strip()} or None
    return build_lazy_engines(repo, os.environ.get("PORTEPAROLE_S3_CKPT") or None, only)
