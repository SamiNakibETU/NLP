"""Chaîne complète : features Whisper -> encodeur gelé -> projecteur -> LLM décodeur."""
from __future__ import annotations

import torch
from torch import nn

from .audio import useful_encoder_frames
from .projector import AudioProjector

SYSTEM = "You are a helpful assistant."
INSTRUCTIONS = {
    "asr": "Transcribe the audio in Arabic.",
    "st": "Translate the audio into French.",
}
PREFIX = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
SUFFIX = "{instruction}<|im_end|>\n<|im_start|>assistant\n"


def encode_text(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False).input_ids)


class SpeechLLM(nn.Module):
    def __init__(self, encoder: nn.Module, llm: nn.Module, tokenizer, max_target_tokens: int = 256):
        super().__init__()
        self.encoder = encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_target_tokens = max_target_tokens
        self.projector = AudioProjector(encoder.config.d_model, llm.get_input_embeddings().embedding_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # l'encodeur reste toujours en eval
        return self

    @property
    def device(self):
        return self.projector.norm.weight.device

    @torch.no_grad()
    def _encode(self, features: torch.Tensor) -> torch.Tensor:
        dtype = next(self.encoder.parameters()).dtype
        return self.encoder(features.to(self.device, dtype)).last_hidden_state  # aucun masque

    def audio_embeddings(self, features: torch.Tensor, mel_lengths: torch.Tensor) -> list[torch.Tensor]:
        """Embeddings audio utiles de chaque exemple (rembourrage jeté) : liste de (n_i, llm_dim)."""
        enc = self._encode(features)
        enc_len = torch.tensor([min(useful_encoder_frames(int(n)), enc.shape[1]) for n in mel_lengths])
        proj, out_len = self.projector(enc.float(), enc_len)
        return [proj[i, : int(n)] for i, n in enumerate(out_len)]

    def build_sequences(self, audio: list[torch.Tensor], task: str, targets: list[str] | None):
        """[préfixe][tokens audio][instruction + début assistant][cible + EOS]. Labels -100 hors cible + EOS."""
        emb = self.llm.get_input_embeddings()
        dev = audio[0].device
        pre = torch.tensor(encode_text(self.tokenizer, PREFIX), device=dev)
        suf = torch.tensor(encode_text(self.tokenizer, SUFFIX.format(instruction=INSTRUCTIONS[task])), device=dev)
        seqs, labels = [], []
        for i, a in enumerate(audio):
            parts = [emb(pre), a.to(emb.weight.dtype), emb(suf)]
            lab = [-100] * (len(pre) + len(a) + len(suf))
            if targets is not None:
                tgt = encode_text(self.tokenizer, targets[i])[: self.max_target_tokens - 1]
                tgt = tgt + [self.tokenizer.eos_token_id]
                parts.append(emb(torch.tensor(tgt, device=dev)))
                lab += tgt
            seqs.append(torch.cat(parts))
            labels.append(torch.tensor(lab, device=dev))
        return seqs, labels

    @staticmethod
    def pad(seqs, labels, side: str):
        """Rembourre le lot : side="right" pour l'entraînement, "left" pour la génération."""
        B, L, H = len(seqs), max(len(s) for s in seqs), seqs[0].shape[-1]
        embeds = seqs[0].new_zeros(B, L, H)
        mask = torch.zeros(B, L, dtype=torch.long, device=embeds.device)
        lab = torch.full((B, L), -100, dtype=torch.long, device=embeds.device)
        for i, (s, l) in enumerate(zip(seqs, labels)):
            n = len(s)
            sl = slice(0, n) if side == "right" else slice(L - n, L)
            embeds[i, sl], mask[i, sl], lab[i, sl] = s, 1, l
        return embeds, mask, lab

    def forward(self, features, mel_lengths, task: str, targets: list[str]):
        audio = self.audio_embeddings(features, mel_lengths)
        seqs, labels = self.build_sequences(audio, task, targets)
        embeds, mask, lab = self.pad(seqs, labels, "right")
        return self.llm(inputs_embeds=embeds, attention_mask=mask, labels=lab)

    @torch.no_grad()
    def generate(self, features, mel_lengths, task: str, max_new_tokens: int = 128, **kw) -> list[str]:
        audio = self.audio_embeddings(features, mel_lengths)
        seqs, labels = self.build_sequences(audio, task, None)
        embeds, mask, _ = self.pad(seqs, labels, "left")
        out = self.llm.generate(inputs_embeds=embeds, attention_mask=mask, max_new_tokens=max_new_tokens,
                                do_sample=False, **kw)
        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in out]
