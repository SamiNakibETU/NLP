"""Moteurs de traduction de la parole. L'interface ne connaît que `Engine.run(chemin, tâche) -> Result`.

Aujourd'hui seuls des moteurs de DÉMONSTRATION existent : ils renvoient un texte d'exemple fixe, marqué
`is_demo=True`, et l'interface l'affiche comme tel. Pour brancher un vrai modèle, implémenter `Engine`
(voir `ProjectorEngine` et `XAttnEngine`) puis le déclarer dans `build_engines()`.
"""
from __future__ import annotations

import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DURATION = 14.0

# Phrases d'exemple écrites pour la démonstration (aucune donnée du corpus MGB2).
CANNED = [
    ("أهلا بكم في نشرة الأخبار لهذا المساء.", "Bonsoir et bienvenue dans ce journal."),
    ("تواصلت المحادثات بين الوفدين حتى ساعة متأخرة من الليل.",
     "Les pourparlers entre les deux délégations se sont poursuivis tard dans la nuit."),
    ("وأعلن المتحدث الرسمي أن الاجتماع المقبل سيعقد الأسبوع القادم.",
     "Le porte-parole officiel a annoncé que la prochaine réunion aurait lieu la semaine prochaine."),
    ("ويرى المراقبون أن الأسواق ستظل حذرة في الأيام المقبلة.",
     "Les observateurs estiment que les marchés resteront prudents dans les jours à venir."),
    ("وفي ما يلي أبرز العناوين.", "Voici les titres principaux."),
]
HEADLINE = "Les pourparlers se poursuivent, la prochaine réunion annoncée"


@dataclass
class Segment:
    start: float
    end: float
    fr: str | None = None
    ar: str | None = None


@dataclass
class Result:
    engine: str                      # nom court de la chaîne
    label: str                       # nom affiché
    task: str                        # "st" (audio -> français) ou "asr" (audio -> arabe)
    duration: float                  # secondes d'audio
    segments: list[Segment] = field(default_factory=list)
    seconds: float | None = None     # temps de traitement mesuré
    audio_tokens: int | None = None
    peak_mem_gb: float | None = None
    produces_arabic: bool = True     # la chaîne passe-t-elle par une transcription arabe ?
    confidences: list[float] | None = None   # une valeur [0,1] par mot du texte principal, si connue
    is_demo: bool = False
    headline: str | None = None
    warning: str | None = None      # à afficher à côté du résultat (ex. le modèle a recopié l'arabe)

    @property
    def fr(self) -> str:
        return " ".join(s.fr for s in self.segments if s.fr)

    @property
    def ar(self) -> str:
        return " ".join(s.ar for s in self.segments if s.ar)

    @property
    def main_text(self) -> str:
        return self.ar if self.task == "asr" else self.fr


class Engine:
    name = "base"
    label = "Moteur"
    blurb = ""
    available = False
    status = "Non branché"

    def run(self, path: str, task: str, progress=None) -> Result:
        raise NotImplementedError


def audio_duration(path: str) -> float:
    """Durée en secondes (soundfile, sinon module wave), ou une durée par défaut si illisible."""
    try:
        import soundfile as sf
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return DEFAULT_DURATION


def split_times(duration: float, target: float = 5.0) -> list[tuple[float, float]]:
    """Découpe [0, duration] en tranches contiguës d'environ `target` secondes."""
    n = max(1, round(duration / target))
    edges = [duration * i / n for i in range(n + 1)]
    return [(edges[i], edges[i + 1]) for i in range(n)]


def fake_confidences(text: str, seed: int) -> list[float]:
    """Confiances factices et déterministes, pour montrer la mise en forme uniquement."""
    out = []
    for i, w in enumerate(text.split()):
        h = (hash_str(w) + seed * 7919 + i * 104729) % 100
        out.append(0.45 + 0.35 * (h / 11) if h < 12 else 0.82 + 0.18 * (h / 99))
    return out


def hash_str(s: str) -> int:
    v = 5381
    for ch in s:
        v = (v * 33 + ord(ch)) & 0xFFFFFFFF
    return v


class DemoEngine(Engine):
    """Texte d'exemple fixe. Sert uniquement à mettre au point l'interface."""

    available = True
    status = "Démonstration"

    def __init__(self, name, label, blurb, produces_arabic, seconds, audio_tokens, mem):
        self.name, self.label, self.blurb = name, label, blurb
        self.produces_arabic, self._sec, self._tok, self._mem = produces_arabic, seconds, audio_tokens, mem

    def run(self, path: str, task: str, progress=None) -> Result:
        t0 = time.time()
        dur = audio_duration(path)
        time.sleep(0.5)  # simule un traitement
        segs = []
        for i, (a, b) in enumerate(split_times(dur)):
            ar, fr = CANNED[i % len(CANNED)]
            segs.append(Segment(a, b, fr=None if task == "asr" else fr,
                                ar=ar if (task == "asr" or self.produces_arabic) else None))
        res = Result(engine=self.name, label=self.label, task=task, duration=dur, segments=segs,
                     seconds=self._sec, audio_tokens=int(dur * 12.5) if self._tok else None,
                     peak_mem_gb=self._mem, produces_arabic=self.produces_arabic, is_demo=True,
                     headline=HEADLINE if task == "st" else "Transcription arabe de la dépêche")
        res.confidences = fake_confidences(res.main_text, hash_str(self.name))
        return res


class ProjectorEngine(Engine):
    """À brancher : audio -> Whisper gelé -> projecteur -> Qwen (projet speech_proj).

    Esquisse : charger `SpeechLLM` et `last.pt` (voir train.py : build_model, load_trainable_state), puis
    pour un audio  feats, mel = extract_features(fe, [load_segment({"audio": path})])
    et  model.generate(feats, mel, task).  Découper en morceaux de 30 s au maximum.
    """
    name, label = "projector", "Projecteur"
    blurb = "Whisper gelé, projecteur à convolutions, Qwen2.5-1.5B."


class XAttnEngine(Engine):
    """À brancher : même chose avec `SpeechLLMXAttn` (projet speech_xattn)."""
    name, label = "xattn", "Cross-attention"
    blurb = "Whisper gelé, blocs d'attention croisée dans Qwen2.5-1.5B."


class CascadeEngine(Engine):
    """À brancher : Whisper (transcription arabe) puis NLLB (arabe -> français)."""
    name, label = "cascade", "Chaîne classique"
    blurb = "Transcription arabe puis traduction automatique du texte."


def build_engines(demo: bool = True) -> dict[str, Engine]:
    """Ordre d'affichage : chaîne classique, projecteur, cross-attention."""
    if not demo:
        return {e.name: e for e in (CascadeEngine(), ProjectorEngine(), XAttnEngine())}
    return {
        "cascade": DemoEngine("cascade", "Chaîne classique", CascadeEngine.blurb, True, 2.1, False, 6.8),
        "projector": DemoEngine("projector", "Projecteur", ProjectorEngine.blurb, False, 0.9, True, 5.4),
        "xattn": DemoEngine("xattn", "Cross-attention", XAttnEngine.blurb, False, 1.4, True, 7.9),
    }


# ---------------------------------------------------------------- sous-titres
def _stamp(t: float, sep: str) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(res: Result) -> str:
    rows = []
    for i, s in enumerate(res.segments, 1):
        rows.append(f"{i}\n{_stamp(s.start, ',')} --> {_stamp(s.end, ',')}\n{(s.fr if res.task == 'st' else s.ar) or ''}\n")
    return "\n".join(rows)


def to_vtt(res: Result) -> str:
    rows = ["WEBVTT\n"]
    for s in res.segments:
        rows.append(f"{_stamp(s.start, '.')} --> {_stamp(s.end, '.')}\n{(s.fr if res.task == 'st' else s.ar) or ''}\n")
    return "\n".join(rows)


def write_subtitles(res: Result, folder: str | Path) -> list[str]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for ext, fn in (("srt", to_srt), ("vtt", to_vtt)):
        p = folder / f"sous_titres.{ext}"
        p.write_text(fn(res), encoding="utf-8")
        out.append(str(p))
    return out
