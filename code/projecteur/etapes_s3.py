"""Temps d'entraînement S3 : quels paramètres sont entraînables.

Repris de ``projecteur_v1/speech_llm/training.py`` (dépôt MLP). Les paramètres
LoRA sont lus dans ``configs/s3.yaml`` (section ``temps_2.lora``).
"""

from typing import Any, Dict, List

from torch import nn


def preparer_temps(modele: nn.Module, temps: int, config_lora: Dict[str, Any]) -> List[nn.Parameter]:
    """Temps 1 : projecteur seul. Temps 2 : projecteur + LoRA sur le LLM (base gelée).

    Renvoie la liste des paramètres entraînables, à donner à l'optimiseur.
    """
    if temps not in (1, 2):
        raise ValueError(f"temps doit valoir 1 ou 2 (reçu : {temps})")
    for p in modele.parameters():
        p.requires_grad = False
    if temps == 2 and not hasattr(modele.llm, "peft_config"):
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=config_lora["r"],
            lora_alpha=config_lora["lora_alpha"],
            lora_dropout=config_lora["lora_dropout"],
            target_modules=list(config_lora["target_modules"]),
            task_type="CAUSAL_LM",
        )
        modele.llm = get_peft_model(modele.llm, config)  # gèle la base, entraîne les adaptateurs
    if temps == 2:
        for nom, p in modele.llm.named_parameters():
            if "lora_" in nom:
                p.requires_grad = True
    for p in modele.projecteur.parameters():
        p.requires_grad = True
    return [p for p in modele.parameters() if p.requires_grad]
