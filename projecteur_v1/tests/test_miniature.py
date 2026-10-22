"""Étape B : tout le circuit avec de petits modèles aléatoires, sans téléchargement."""
import torch

from speech_llm.model import PREFIX, SUFFIX, INSTRUCTIONS, SpeechLLM, encode_text
from speech_llm.projector import AudioProjector, audio_token_count, conv_out_len
from speech_llm.tiny import FakeTokenizer, tiny_batch, tiny_encoder, tiny_llm
from speech_llm.training import prepare_stage

TARGETS = ["مرحبا", "bonjour tout le monde"]


def make_model():
    torch.manual_seed(0)
    return SpeechLLM(tiny_encoder(), tiny_llm(), FakeTokenizer())


def test_projector_lengths():
    assert conv_out_len(500) == 250 and audio_token_count(500) == 125
    assert audio_token_count(1500) == 375
    proj = AudioProjector(32, 16)
    y, out = proj(torch.randn(2, 50, 32), torch.tensor([30, 50]))
    assert y.shape == (2, 13, 16)  # 50 -> 25 -> 13
    assert out.tolist() == [audio_token_count(30), audio_token_count(50)] == [8, 13]


def test_audio_embeddings_drop_padding():
    m = make_model()
    feats, mel = tiny_batch()
    audio = m.audio_embeddings(feats, mel)
    # mel utiles 60 -> encodeur 30 -> 15 -> 8 ; mel 100 -> 50 -> 25 -> 13
    assert [a.shape[0] for a in audio] == [8, 13]
    assert audio[0].shape[1] == m.llm.config.hidden_size


def test_sequence_assembly_and_label_masking():
    m = make_model()
    feats, mel = tiny_batch()
    audio = m.audio_embeddings(feats, mel)
    seqs, labels = m.build_sequences(audio, "st", TARGETS)
    n_pre = len(encode_text(m.tokenizer, PREFIX))
    n_suf = len(encode_text(m.tokenizer, SUFFIX.format(instruction=INSTRUCTIONS["st"])))
    for i, (s, lab) in enumerate(zip(seqs, labels)):
        tgt = encode_text(m.tokenizer, TARGETS[i]) + [m.tokenizer.eos_token_id]
        assert len(s) == n_pre + audio[i].shape[0] + n_suf + len(tgt)
        assert (lab[: len(s) - len(tgt)] == -100).all()      # rien hors cible
        assert lab[len(s) - len(tgt):].tolist() == tgt        # cible + EOS
    # rembourrage à droite pour l'entraînement
    emb, mask, lab = m.pad(seqs, labels, "right")
    assert mask[0, : len(seqs[0])].all() and not mask[0, len(seqs[0]):].any()
    assert (lab[mask == 0] == -100).all()


def test_stage1_only_projector_gets_gradients():
    m = make_model()
    params = prepare_stage(m, 1)
    feats, mel = tiny_batch()
    out = m(feats, mel, "asr", TARGETS)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert all(p.grad is not None for p in m.projector.parameters())
    assert all(p.grad is None for p in m.llm.parameters())
    assert all(p.grad is None for p in m.encoder.parameters())
    assert {id(p) for p in params} == {id(p) for p in m.projector.parameters()}


def test_stage2_projector_and_lora_get_gradients():
    m = make_model()
    params = prepare_stage(m, 2)
    feats, mel = tiny_batch()
    out = m(feats, mel, "st", TARGETS)
    out.loss.backward()
    lora = [(n, p) for n, p in m.llm.named_parameters() if "lora_" in n]
    base = [(n, p) for n, p in m.llm.named_parameters() if "lora_" not in n]
    assert lora and all(p.requires_grad for _, p in lora)
    assert all(not p.requires_grad for _, p in base)
    assert all(p.grad is not None for p in m.projector.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for _, p in lora)
    assert all(p.grad is None for _, p in base)
    assert all(p.grad is None for p in m.encoder.parameters())
    assert len(params) == len(list(m.projector.parameters())) + len(lora)


def test_encoder_stays_frozen_and_eval():
    m = make_model()
    m.train()
    assert not m.encoder.training
    assert all(not p.requires_grad for p in m.encoder.parameters())


def test_generation_runs_with_left_padding():
    m = make_model().eval()
    feats, mel = tiny_batch()
    texts = m.generate(feats, mel, "asr", max_new_tokens=5)
    assert len(texts) == 2 and all(isinstance(t, str) for t in texts)
    audio = m.audio_embeddings(feats, mel)
    seqs, labels = m.build_sequences(audio, "asr", None)
    _, mask, _ = m.pad(seqs, labels, "left")
    assert mask[0, -1] == 1 and mask[0, 0] == 0  # le plus court est rembourré à gauche
    assert mask[1].all()
