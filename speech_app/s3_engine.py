"""Moteur S3 de l'équipe : encodeur Cohere Transcribe Arabic gelé, projecteur entraîné, Qwen2.5-1.5B-Instruct.

Le code du modèle est celui du dépôt de l'équipe (branche `benchmarking` de SamiNakibETU/NLP) : donner la racine de
son dossier à `load_s3_engines`. Deux moteurs partagent les mêmes modèles chargés une seule fois :
  s3          français DIRECT : vecteurs audio projetés + consigne « Traduis en français. », sans texte arabe
  s3_cascade  cascade : transcription arabe, puis traduction du texte par le même Qwen (pour comparer)
Tâche « asr » (transcrire en arabe) : identique pour les deux.
Le projecteur chargé est celui du temps 1 (entraîné à transcrire). Dans `notebooks/s3_tester_audio.ipynb` de l'équipe, le
test de traduction directe sans temps 2 donne « arabe dans la sortie : 100 % » : le moteur `s3` avertit donc quand il recopie
l'arabe. Tant que le temps 2 n'est pas entraîné, c'est la cascade qui produit le français.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from engines import Engine, Result, Segment
from model_engine import SAMPLE_RATE, subtitle_lines

BLURB = {
    "direct": "Essai de français direct avec le projecteur du temps 1 : sans temps 2, le modèle recopie l'arabe.",
    "cascade": "Même modèle, en deux temps : transcription arabe, puis traduction du texte par Qwen.",
}
ARABIC_ALERT = 0.30   # au-delà de cette part de lettres arabes dans une « traduction française », on avertit


def arabic_share(text: str) -> float:
    """Part de lettres arabes parmi les lettres du texte."""
    letters = [c for c in text if c.isalpha()]
    return sum("\u0600" <= c <= "\u06ff" for c in letters) / len(letters) if letters else 0.0


class S3Engine(Engine):
    available = True
    status = "Modèle entraîné"

    def __init__(self, name: str, label: str, mode: str, transcripteur, loader=None):
        assert mode in ("direct", "cascade")
        self.name, self.label, self.mode, self.blurb = name, label, mode, BLURB[mode]
        self.produces_arabic = mode == "cascade"
        self.t = transcripteur
        self.loader = loader

    def _signal(self, path: str) -> np.ndarray:
        if self.loader:
            return self.loader(path)
        from transcrire_s3 import charger_audio  # fourni par le dépôt de l'équipe (sys.path réglé au chargement)
        return charger_audio(path)

    def run(self, path: str, task: str = "st", progress=None) -> Result:
        t0 = time.time()
        signal = self._signal(path)
        duration = len(signal) / SAMPLE_RATE
        if progress:
            progress(0.05, desc="Encodage et génération")
        segs: list[Segment] = []
        if task == "asr":
            for p in self.t.transcrire_signal(signal):
                segs += [Segment(s, e, ar=line) for s, e, line in subtitle_lines(p["texte"], p["debut_s"], p["fin_s"])]
        elif self.mode == "direct":
            for p in self.t.traduire_signal(signal):
                segs += [Segment(s, e, fr=line) for s, e, line in subtitle_lines(p["texte"], p["debut_s"], p["fin_s"])]
        else:
            pieces = self.t.transcrire_signal(signal)
            fr_texts = self.t.traduire([p["texte"] for p in pieces])
            for p, fr in zip(pieces, fr_texts):
                lines = subtitle_lines(fr, p["debut_s"], p["fin_s"])
                for i, (s, e, line) in enumerate(lines):
                    segs.append(Segment(s, e, fr=line, ar=p["texte"] if i == 0 else None))
                if not lines and p["texte"]:                       # traduction vide : on garde au moins l'arabe
                    segs.append(Segment(p["debut_s"], p["fin_s"], ar=p["texte"]))
        peak = None
        try:
            import torch
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / 1e9
        except ImportError:
            pass
        if progress:
            progress(1.0, desc="Terminé")
        res = Result(engine=self.name, label=self.label, task=task, duration=duration, segments=segs,
                     seconds=time.time() - t0, audio_tokens=int(duration * 12.5), peak_mem_gb=peak,
                     produces_arabic=self.produces_arabic, is_demo=False)
        if task == "st" and self.mode == "direct" and (share := arabic_share(res.fr)) > ARABIC_ALERT:
            res.warning = (f"Le modèle a recopié de l'arabe au lieu de traduire ({share:.0%} de lettres arabes). "
                           "Le temps 2 n'est pas entraîné : utilise « via le texte arabe » pour obtenir du français.")
        return res


def load_s3_engines(repo_dir: str | Path, checkpoint: str | Path | None = None,
                    section: str = "temps_1_50k") -> dict[str, S3Engine]:
    """Charge les vrais modèles (HF_TOKEN requis pour l'encodeur Cohere) et renvoie les moteurs `s3` et `s3_cascade`."""
    repo = Path(repo_dir)
    code = repo / "code" / "projecteur"
    if not (code / "transcrire_s3.py").is_file():
        raise FileNotFoundError(f"{code}/transcrire_s3.py introuvable : donne la racine du dépôt (branche benchmarking)")
    sys.path.append(str(code))                      # transcrire_s3 règle lui-même les autres dossiers
    import transcrire_s3

    t = transcrire_s3.TranscripteurS3.charger(section, Path(checkpoint) if checkpoint else None)
    # la cascade d'abord : c'est elle qui produit du français aujourd'hui (moteur par défaut dans l'interface)
    return {"s3_cascade": S3Engine("s3_cascade", "S3, via le texte arabe", "cascade", t),
            "s3": S3Engine("s3", "S3, français direct (essai)", "direct", t)}
