"""Module de la cascade de sous-titrage S1 / S2 (section 4.1 et section 5)."""

from .entrainement_lora import (
    MiniLLMTraducteurLoRA,
    ProjectionAttentionLoRA,
    calculer_lr_cosinus_avec_echauffement,
    entrainer_lora_s2_essai,
    estimer_budget_calcul_s2,
    extraire_exemple_logits_softmax,
)
from .inference_s1_s2 import SystemeCascadeS1S2, executer_mode_essai_inference
from .preparation_exemples import (
    charger_config_s2,
    charger_lexique_noms,
    composer_jeu_entrainement_s2,
    construire_contenu_utilisateur_cascade,
    construire_messages_cascade,
    detecter_noms_propres_segment,
    encoder_exemple_avec_masquage_s2,
    formater_chat_template_cascade,
    generer_exemples_factices_s2,
)

__all__ = [
    "MiniLLMTraducteurLoRA",
    "ProjectionAttentionLoRA",
    "calculer_lr_cosinus_avec_echauffement",
    "entrainer_lora_s2_essai",
    "estimer_budget_calcul_s2",
    "extraire_exemple_logits_softmax",
    "SystemeCascadeS1S2",
    "executer_mode_essai_inference",
    "charger_config_s2",
    "charger_lexique_noms",
    "composer_jeu_entrainement_s2",
    "construire_contenu_utilisateur_cascade",
    "construire_messages_cascade",
    "detecter_noms_propres_segment",
    "encoder_exemple_avec_masquage_s2",
    "formater_chat_template_cascade",
    "generer_exemples_factices_s2",
]
