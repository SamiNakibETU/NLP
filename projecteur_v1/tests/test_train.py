"""Boucle d'entraînement avec de faux modèles : la perte baisse, la reprise fonctionne."""
import json

import torch

from speech_llm.model import SpeechLLM
from speech_llm.tiny import FakeTokenizer, tiny_batch, tiny_encoder, tiny_llm
from train import load_manifest, run_training, wer


def make_model():
    torch.manual_seed(0)
    return SpeechLLM(tiny_encoder(), tiny_llm(), FakeTokenizer())


def fixed_loader(n=4):
    feats, mel = tiny_batch()
    batch = {"features": feats, "mel_lengths": mel, "ar": ["مرحبا", "اهلا بكم"], "fr": ["bonjour", "bienvenue"]}
    return [batch] * n


def test_loss_decreases_stage1(tmp_path):
    log = run_training(make_model(), fixed_loader(), fixed_loader(1), stage=1, tasks=["asr"], epochs=15,
                       lr_proj=3e-3, lr_lora=0, warmup=0.1, max_steps=None, eval_every=30, out_dir=tmp_path, log_every=1000)
    assert len(log) == 60
    assert sum(r["loss"] for r in log[-5:]) / 5 < sum(r["loss"] for r in log[:5]) / 5 - 0.3
    assert (tmp_path / "last.pt").exists() and (tmp_path / "val.jsonl").exists()


def test_stage2_trains_lora_and_both_tasks(tmp_path):
    m = make_model()
    log = run_training(m, fixed_loader(), None, stage=2, tasks=["asr", "st"], epochs=10, lr_proj=1e-3, lr_lora=3e-3,
                       warmup=0.1, max_steps=None, eval_every=1000, out_dir=tmp_path, log_every=1000)
    assert {r["task"] for r in log} == {"asr", "st"}
    assert any("lora_" in n and p.requires_grad for n, p in m.named_parameters())
    feats, mel = tiny_batch()
    assert len(m.eval().generate(feats, mel, "st", max_new_tokens=4)) == 2  # génération après l'étape 2


def test_resume_continues_from_saved_step(tmp_path):
    run_training(make_model(), fixed_loader(), None, stage=1, tasks=["asr"], epochs=2, lr_proj=1e-3, lr_lora=0,
                 warmup=0.1, max_steps=5, eval_every=5, out_dir=tmp_path, log_every=1000)
    assert torch.load(tmp_path / "last.pt")["step"] == 5
    log = run_training(make_model(), fixed_loader(), None, stage=1, tasks=["asr"], epochs=3, lr_proj=1e-3, lr_lora=0,
                       warmup=0.1, max_steps=8, eval_every=100, out_dir=tmp_path, log_every=1000,
                       resume=str(tmp_path / "last.pt"))
    assert [r["step"] for r in log] == [6, 7, 8]


def test_wer():
    assert wer("a b c d", "a b c d") == 0
    assert wer("a b c d", "a x c") == 0.5


def test_load_manifest_filters_and_caps_hours(tmp_path):
    rows = [{"audio": "x", "start": 0, "end": 10, "text_ar": "نص"} for _ in range(10)] + \
           [{"audio": "x", "start": 0, "end": 10, "text_ar": "  "}]
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    assert len(load_manifest(str(p))) == 10                          # texte vide retiré
    assert len(load_manifest(str(p), max_hours=25 / 3600)) == 2      # 25 s -> 2 segments de 10 s
