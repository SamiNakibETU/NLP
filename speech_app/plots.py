"""Journal d'entraînement : lecture de log.jsonl / val.jsonl et graphique au style du journal."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER, INK, RED, MUTED = "#f3ede0", "#1b1a17", "#8c1c13", "#7b7466"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_logs(folder: str | Path) -> tuple[list[dict], list[dict], bool]:
    """(entraînement, validation, is_demo). Sans journal, renvoie des données d'illustration."""
    folder = Path(folder)
    train, val = read_jsonl(folder / "log.jsonl"), read_jsonl(folder / "val.jsonl")
    if train:
        return train, val, False
    return (*demo_logs(), True)


def demo_logs() -> tuple[list[dict], list[dict]]:
    """Courbes d'illustration, sans rapport avec un vrai entraînement."""
    train = [{"step": s, "loss": 3.4 + 1.4 * math.exp(-s / 90) + 0.08 * math.sin(s / 4)} for s in range(1, 401)]
    val = [{"step": s, "val_loss": 3.4 + 1.3 * math.exp(-s / 100), "val_wer_sample": 1.0,
            "samples": [{"ref": "أهلا بكم في نشرة الأخبار", "hyp": "وفي ما يلي أبرز العناوين"}]} for s in (100, 200, 300, 400)]
    return train, val


def smooth(xs: list[float], w: int = 20) -> list[float]:
    out, acc = [], 0.0
    for i, x in enumerate(xs):
        acc += x
        if i >= w:
            acc -= xs[i - w]
        out.append(acc / min(i + 1, w))
    return out


def loss_figure(train: list[dict], val: list[dict], demo: bool):
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=140)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    steps = [r["step"] for r in train]
    ax.plot(steps, [r["loss"] for r in train], color=INK, alpha=0.18, lw=0.8)
    ax.plot(steps, smooth([r["loss"] for r in train]), color=INK, lw=1.4, label="entraînement (lissé)")
    if val:
        ax.plot([r["step"] for r in val], [r["val_loss"] for r in val], color=RED, lw=1.6, marker="o", ms=4,
                label="validation")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis="y", color=INK, alpha=0.12, lw=0.6)
    ax.set_xlabel("étape", color=INK, fontsize=9, family="serif")
    ax.set_ylabel("perte", color=INK, fontsize=9, family="serif")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    if demo:
        ax.text(0.5, 0.5, "DONNÉES D'ILLUSTRATION", transform=ax.transAxes, ha="center", va="center", rotation=12,
                fontsize=22, color=RED, alpha=0.25, family="serif")
    fig.tight_layout()
    return fig
