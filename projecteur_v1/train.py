"""Entraînement du projecteur (étape 1) puis projecteur + LoRA (étape 2).

Exemple (Colab T4) :
  python train.py --stage 1 --train-manifest manifest_dev_train.jsonl --val-manifest manifest_dev_val.jsonl \
      --epochs 3 --batch 8 --out /content/ckpt_stage1

Chaque ligne de manifeste : {"audio", "start", "end", "text_ar", "text_fr" (facultatif)}.
Reprise après coupure : relancer la même commande avec --resume <out>/last.pt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from speech_llm.audio import extract_features, load_segment
from speech_llm.model import SpeechLLM
from speech_llm.training import prepare_stage


# ---------------------------------------------------------------- données
def load_manifest(path: str, max_hours: float | None = None, seed: int = 0, need_fr: bool = False) -> list[dict]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r.get("text_ar", "").strip() and (not need_fr or r.get("text_fr", "").strip())]
    if max_hours is not None:
        random.Random(seed).shuffle(rows)
        kept, total = [], 0.0
        for r in rows:
            d = min(float(r["end"]) - float(r["start"]), 30.0) if "start" in r and "end" in r else 30.0
            if total + d > max_hours * 3600:
                break
            kept.append(r)
            total += d
        rows = kept
    return rows


def hours(rows: list[dict]) -> float:
    return sum(min(float(r["end"]) - float(r["start"]), 30.0) for r in rows if "start" in r and "end" in r) / 3600


class SegmentDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return load_segment(r), r["text_ar"], r.get("text_fr")


def make_collate(feature_extractor):
    def collate(items):
        feats, mel = extract_features(feature_extractor, [w for w, _, _ in items])
        fr = [f for _, _, f in items]
        return {"features": feats, "mel_lengths": mel, "ar": [a for _, a, _ in items],
                "fr": fr if all(fr) else None}
    return collate


# ---------------------------------------------------------------- métriques
def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


# ---------------------------------------------------------------- état sauvegardé
def trainable_state(model: SpeechLLM) -> dict:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model: SpeechLLM, state: dict) -> None:
    params = dict(model.named_parameters())
    for n, v in state.items():
        params[n].data.copy_(v)


def save_checkpoint(path: Path, model, opt, sched, scaler, step: int, extra: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"trainable": trainable_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict() if scaler else None, "step": step, **extra}, tmp)
    tmp.replace(path)  # écriture atomique : une coupure ne laisse pas un fichier à moitié écrit


# ---------------------------------------------------------------- évaluation
@torch.no_grad()
def evaluate(model: SpeechLLM, val_loader, device, amp: bool, max_batches: int = 20, n_gen: int = 8,
             max_new_tokens: int = 96) -> dict:
    model.eval()
    losses, wers, samples = [], [], []
    for b, batch in enumerate(val_loader):
        if b >= max_batches:
            break
        with torch.autocast(device, dtype=torch.float16, enabled=amp):
            losses.append(float(model(batch["features"], batch["mel_lengths"], "asr", batch["ar"]).loss))
            if b == 0:
                k = min(n_gen, len(batch["ar"]))
                hyp = model.generate(batch["features"][:k], batch["mel_lengths"][:k], "asr", max_new_tokens=max_new_tokens)
                for ref, h in zip(batch["ar"][:k], hyp):
                    wers.append(wer(ref, h))
                    samples.append({"ref": ref, "hyp": h})
    model.train()
    return {"val_loss": sum(losses) / max(1, len(losses)),
            "val_wer_sample": sum(wers) / max(1, len(wers)) if wers else None, "samples": samples[:3]}


# ---------------------------------------------------------------- boucle
def run_training(model: SpeechLLM, train_loader, val_loader, *, stage: int, tasks: list[str], epochs: int,
                 lr_proj: float, lr_lora: float, warmup: float, max_steps: int | None, eval_every: int,
                 out_dir: Path, device: str = "cpu", amp: bool = False, grad_clip: float = 1.0,
                 log_every: int = 20, resume: str | None = None, seed: int = 0, extra: dict | None = None) -> list[dict]:
    from transformers import get_cosine_schedule_with_warmup

    out_dir.mkdir(parents=True, exist_ok=True)
    prepare_stage(model, stage)
    groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n], "lr": lr_proj}]
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    if lora:
        groups.append({"params": lora, "lr": lr_lora})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    total = epochs * len(train_loader)
    total = min(total, max_steps) if max_steps else total
    sched = get_cosine_schedule_with_warmup(opt, int(warmup * total), total)
    scaler = torch.amp.GradScaler("cuda") if amp else None

    step = 0
    if resume:
        ck = torch.load(resume, map_location="cpu")
        load_trainable_state(model, ck["trainable"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        if scaler and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        step = ck["step"]
        print(f"reprise à l'étape {step}/{total}", flush=True)

    rng = random.Random(seed + step)
    log, t0 = [], time.time()
    model.train()
    while step < total:
        for batch in train_loader:
            if step >= total:
                break
            usable = [t for t in tasks if t == "asr" or batch["fr"] is not None]
            task = rng.choice(usable)
            targets = batch["ar"] if task == "asr" else batch["fr"]
            with torch.autocast(device, dtype=torch.float16, enabled=amp):
                loss = model(batch["features"], batch["mel_lengths"], task, targets).loss
            if not torch.isfinite(loss):
                print(f"étape {step}: perte non finie, batch ignoré", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
            else:
                loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], grad_clip))
            if scaler:
                scaler.step(opt); scaler.update()
            else:
                opt.step()
            sched.step()
            step += 1
            rec = {"step": step, "task": task, "loss": loss.item(), "grad_norm": gnorm, "lr": sched.get_last_lr()[0]}
            log.append(rec)
            with open(out_dir / "log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            if step % log_every == 0 or step == 1:
                recent = [r["loss"] for r in log[-log_every:]]
                print(f"étape {step}/{total} | perte {sum(recent)/len(recent):.4f} | gnorm {gnorm:.2f} | "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)
            if step % eval_every == 0 or step == total:
                if val_loader is not None:
                    ev = evaluate(model, val_loader, device, amp)
                    print(f"  >> validation étape {step}: perte {ev['val_loss']:.4f} | WER (échantillon) {ev['val_wer_sample']}", flush=True)
                    for s in ev["samples"][:2]:
                        print("     réf:", s["ref"], "\n     hyp:", s["hyp"], flush=True)
                    with open(out_dir / "val.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"step": step, **ev}, ensure_ascii=False) + "\n")
                save_checkpoint(out_dir / "last.pt", model, opt, sched, scaler, step, extra or {})
    return log


# ---------------------------------------------------------------- construction des vrais modèles
def build_model(llm_name: str, device: str, max_target_tokens: int) -> tuple[SpeechLLM, object, object]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor, WhisperModel

    def load(cls, name):
        try:
            return cls.from_pretrained(name, dtype=torch.float16)
        except TypeError:
            return cls.from_pretrained(name, torch_dtype=torch.float16)

    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    encoder = load(WhisperModel, "openai/whisper-large-v3").encoder
    llm = load(AutoModelForCausalLM, llm_name)
    tok = AutoTokenizer.from_pretrained(llm_name)
    model = SpeechLLM(encoder, llm, tok, max_target_tokens=max_target_tokens).to(device)
    return model, tok, fe


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--val-manifest")
    p.add_argument("--stage", type=int, choices=(1, 2), default=1)
    p.add_argument("--tasks", default=None, help="asr (étape 1) ou asr,st (étape 2)")
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr-proj", type=float, default=None)
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--max-target-tokens", type=int, default=96)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="/content/ckpt")
    p.add_argument("--resume")
    p.add_argument("--init", help="checkpoint dont on reprend seulement les poids (ex. projecteur de l'étape 1 pour l'étape 2)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"  # T4 : fp16 (pas de bf16)
    tasks = (a.tasks or ("asr" if a.stage == 1 else "asr,st")).split(",")
    lr_proj = a.lr_proj or (5e-4 if a.stage == 1 else 1e-4)  # valeurs de départ, non validées sur l'arabe

    train_rows = load_manifest(a.train_manifest, a.max_hours, a.seed, need_fr="st" in tasks)
    val_rows = load_manifest(a.val_manifest) if a.val_manifest else []
    print(f"entraînement : {len(train_rows)} segments, {hours(train_rows):.2f} h | validation : {len(val_rows)} segments", flush=True)

    model, tok, fe = build_model(a.llm, device, a.max_target_tokens)
    if a.init:
        prepare_stage(model, a.stage)  # crée les LoRA avant de charger si l'étape 2 en a besoin
        load_trainable_state(model, torch.load(a.init, map_location="cpu")["trainable"])
        print("poids initiaux chargés depuis", a.init, flush=True)
    collate = make_collate(fe)
    train_loader = DataLoader(SegmentDataset(train_rows), batch_size=a.batch, shuffle=True, collate_fn=collate,
                              num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    val_loader = DataLoader(SegmentDataset(val_rows), batch_size=a.batch, shuffle=False, collate_fn=collate,
                            num_workers=a.workers) if val_rows else None
    log = run_training(model, train_loader, val_loader, stage=a.stage, tasks=tasks, epochs=a.epochs, lr_proj=lr_proj,
                       lr_lora=a.lr_lora, warmup=a.warmup, max_steps=a.max_steps, eval_every=a.eval_every,
                       out_dir=Path(a.out), device=device, amp=amp, resume=a.resume, seed=a.seed, extra={"args": vars(a)})
    print("terminé. Dernières pertes :", [round(r["loss"], 3) for r in log[-5:]])


if __name__ == "__main__":
    main()
