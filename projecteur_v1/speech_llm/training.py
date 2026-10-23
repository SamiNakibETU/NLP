"""Étapes d'entraînement : quels paramètres sont entraînables."""
from __future__ import annotations

from peft import LoraConfig, get_peft_model

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def prepare_stage(model, stage: int, lora_r: int = 16, lora_alpha: int = 32):
    """Étape 1 : projecteur seul. Étape 2 : projecteur + LoRA (r=16, alpha=32) sur le LLM."""
    for p in model.parameters():
        p.requires_grad = False
    if stage == 2 and not hasattr(model.llm, "peft_config"):
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        model.llm = get_peft_model(model.llm, cfg)  # gèle la base, entraîne les adaptateurs
    for p in model.projector.parameters():
        p.requires_grad = True
    return [p for p in model.parameters() if p.requires_grad]
