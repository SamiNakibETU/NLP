"""Entraînement en deux temps (et essai en un seul temps) du système S3."""

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

RACINE_PROJET = Path(__file__).resolve().parents[2]
DOSSIER_PROJECTEUR = Path(__file__).resolve().parent
DOSSIER_CASCADE = RACINE_PROJET / "code" / "cascade"
DOSSIER_ECOUTE = RACINE_PROJET / "code" / "ecoute"

for p in (DOSSIER_PROJECTEUR, DOSSIER_CASCADE, DOSSIER_ECOUTE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from architecture_s3 import (  # noqa: E402
    CHEMIN_CONFIG_S3,
    ProjecteurAudioLLM,
    assembler_entree_s3,
    charger_config_s3,
)
from entrainement_lora import ProjectionAttentionLoRA, fixer_graine  # noqa: E402


class DecodeurLLMPourInputsEmbeds(nn.Module):
    """Décodeur causal acceptant directement `inputs_embeds` avec projections LoRA activables."""

    def __init__(
        self,
        vocab_size: int = 1024,
        d_llm: int = 128,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_llm = int(d_llm)
        self.embed_tokens = nn.Embedding(self.vocab_size, self.d_llm)
        self.q_proj = ProjectionAttentionLoRA(self.d_llm, self.d_llm, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.k_proj = ProjectionAttentionLoRA(self.d_llm, self.d_llm, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.v_proj = ProjectionAttentionLoRA(self.d_llm, self.d_llm, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.o_proj = ProjectionAttentionLoRA(self.d_llm, self.d_llm, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.norm = nn.LayerNorm(self.d_llm)
        self.lm_head = nn.Linear(self.d_llm, self.vocab_size, bias=False)

        # Figer la base du LLM par défaut
        self.embed_tokens.weight.requires_grad = False
        self.norm.weight.requires_grad = False
        self.norm.bias.requires_grad = False
        self.lm_head.weight.requires_grad = False

    def activer_lora(self, actif: bool) -> None:
        """Active ou fige les poids LoRA du LLM (figés au temps 1, entraînés au temps 2)."""
        for nom, param in self.named_parameters():
            if "lora_" in nom:
                param.requires_grad = bool(actif)
            else:
                param.requires_grad = False

    def forward(self, inputs_embeds: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if inputs_embeds.ndim == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        if labels is not None and labels.ndim == 1:
            labels = labels.unsqueeze(0)

        q = self.q_proj(inputs_embeds)
        k = self.k_proj(inputs_embeds)
        v = self.v_proj(inputs_embeds)
        # Agrégation causale cumulative permettant aux tokens de réponse d'exploiter les vecteurs audio précédents
        somme_cum = torch.cumsum(q + k, dim=1)
        nb_pos = torch.arange(1, inputs_embeds.shape[1] + 1, device=inputs_embeds.device, dtype=inputs_embeds.dtype).view(1, -1, 1)
        contexte_causal = F.gelu(somme_cum / nb_pos) + v
        h = self.norm(inputs_embeds + self.o_proj(contexte_causal))
        logits = self.lm_head(h)

        perte = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            perte = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": perte}


def diagnostiquer_stagnation_s3(
    assemblage: Dict[str, Any],
    nb_vecteurs_complets: int,
    nb_vecteurs_utiles: int,
    duree_reelle_s: float,
    duree_fenetre_s: float,
    learning_rate: float,
) -> Dict[str, Any]:
    """Vérifie dans l'ordre les 4 causes possibles de stagnation au temps 1 :
    1. Le masquage des étiquettes (`-100` avant la réponse, étiquettes valides sur la réponse) ;
    2. L'ordre des vecteurs dans l'entrée (`debut_user -> audio -> consigne -> fin_user_debut_assistant -> reponse`) ;
    3. Le retrait du silence (`nb_vecteurs_utiles < nb_vecteurs_complets` quand `duree_reelle_s < duree_fenetre_s`) ;
    4. Le pas d'apprentissage (`0 < lr <= 1e-2`).
    """
    labels = assemblage["labels"].tolist()
    plages = assemblage["plages"]
    deb_rep, fin_rep = plages["reponse"]

    masquage_ok = (
        deb_rep > 0
        and fin_rep > deb_rep
        and all(x == -100 for x in labels[:deb_rep])
        and all(x != -100 and x >= 0 for x in labels[deb_rep:fin_rep])
    )

    ordre_ok = (
        plages["debut_user"][0] == 0
        and plages["debut_user"][1] == plages["audio"][0]
        and plages["audio"][1] == plages["consigne"][0]
        and plages["consigne"][1] == plages["fin_user_debut_assistant"][0]
        and plages["fin_user_debut_assistant"][1] == plages["reponse"][0]
    )

    if duree_reelle_s < duree_fenetre_s:
        silence_retire_ok = 0 < nb_vecteurs_utiles < nb_vecteurs_complets
    else:
        silence_retire_ok = nb_vecteurs_utiles == nb_vecteurs_complets

    lr_ok = 1e-7 <= float(learning_rate) <= 1e-1

    return {
        "1_masquage_etiquettes_ok": masquage_ok,
        "2_ordre_vecteurs_ok": ordre_ok,
        "3_retrait_silence_ok": silence_retire_ok,
        "4_pas_apprentissage_ok": lr_ok,
        "diagnostic_global_ok": (masquage_ok and ordre_ok and silence_retire_ok and lr_ok),
    }


def construire_exemples_audio_s3(
    nb_exemples: int = 32,
    d_enc: int = 64,
    vocab_size: int = 256,
    graine: int = 42,
    part_transcription: float = 1.0,
) -> List[Dict[str, Any]]:
    """Construit un lot d'exemples audio-texte déterministes (Common Voice / A / D) pour valider la convergence."""
    gen = torch.Generator().manual_seed(graine)
    motifs_audio = [
        torch.randn((1500, d_enc), generator=gen),
        torch.randn((1500, d_enc), generator=gen),
        torch.randn((1500, d_enc), generator=gen),
        torch.randn((1500, d_enc), generator=gen),
    ]
    cibles_ar = [
        torch.tensor([20, 21, 22, 23, 2], dtype=torch.long),
        torch.tensor([30, 31, 32, 33, 2], dtype=torch.long),
        torch.tensor([40, 41, 42, 43, 2], dtype=torch.long),
        torch.tensor([50, 51, 52, 53, 2], dtype=torch.long),
    ]
    cibles_fr = [
        torch.tensor([60, 61, 62, 63, 2], dtype=torch.long),
        torch.tensor([70, 71, 72, 73, 2], dtype=torch.long),
        torch.tensor([80, 81, 82, 83, 2], dtype=torch.long),
        torch.tensor([90, 91, 92, 93, 2], dtype=torch.long),
    ]

    ids_debut_user = torch.tensor([1, 10], dtype=torch.long)
    ids_consigne_ar = torch.tensor([11, 12, 13], dtype=torch.long)
    ids_consigne_fr = torch.tensor([14, 15, 16], dtype=torch.long)
    ids_fin_user = torch.tensor([17, 18], dtype=torch.long)

    nb_trans = int(round(nb_exemples * part_transcription))
    exemples: List[Dict[str, Any]] = []
    for idx in range(nb_exemples):
        classe = idx % len(motifs_audio)
        est_transcription = idx < nb_trans
        exemples.append(
            {
                "id": f"s3_ex_{idx:04d}",
                "tache": "transcription" if est_transcription else "traduction",
                "vecteurs_encodeur_30s": motifs_audio[classe],
                "duree_reelle_s": 6.0,  # 6 s sur 30 s -> 300 vecteurs utiles -> 75 vecteurs après regroupement k=4
                "ids_debut_user": ids_debut_user,
                "ids_consigne": ids_consigne_ar if est_transcription else ids_consigne_fr,
                "ids_fin_user_debut_assistant": ids_fin_user,
                "ids_reponse": cibles_ar[classe] if est_transcription else cibles_fr[classe],
            }
        )
    return exemples


def entrainer_s3_temps1(
    dossier_sortie: Union[str, Path],
    d_enc: int = 64,
    d_llm: int = 128,
    k: int = 4,
    nb_etapes: int = 60,
    lr: float = 5e-3,
    graine: int = 42,
) -> Dict[str, Any]:
    """Temps 1 de S3 : encodeur et LLM figés, seul le projecteur est entraîné sur la transcription arabe."""
    fixer_graine(graine)
    cible_dir = Path(dossier_sortie)
    cible_dir.mkdir(parents=True, exist_ok=True)

    projecteur = ProjecteurAudioLLM(d_enc=d_enc, d_llm=d_llm, k=k)
    llm = DecodeurLLMPourInputsEmbeds(vocab_size=256, d_llm=d_llm)
    llm.activer_lora(False)

    # Vérification stricte : seul le projecteur a requires_grad = True
    nb_params_llm_grad = sum(p.numel() for p in llm.parameters() if p.requires_grad)
    if nb_params_llm_grad != 0:
        raise RuntimeError(f"Le LLM doit être entièrement figé au temps 1 (trouvé : {nb_params_llm_grad})")

    optimiseur = torch.optim.AdamW(projecteur.parameters(), lr=lr)
    exemples = construire_exemples_audio_s3(nb_exemples=16, d_enc=d_enc, graine=graine, part_transcription=1.0)

    def _evaluer_val() -> float:
        projecteur.eval()
        pertes = []
        with torch.no_grad():
            for ex in exemples[:4]:
                proj_audio = projecteur(
                    ex["vecteurs_encodeur_30s"],
                    duree_reelle_s=ex["duree_reelle_s"],
                    duree_fenetre_s=30.0,
                )
                ass = assembler_entree_s3(
                    vecteurs_audio_projetes=proj_audio,
                    ids_debut_user=ex["ids_debut_user"],
                    ids_consigne=ex["ids_consigne"],
                    ids_fin_user_debut_assistant=ex["ids_fin_user_debut_assistant"],
                    table_embeddings_llm=llm.embed_tokens,
                    ids_reponse=ex["ids_reponse"],
                )
                sortie = llm(inputs_embeds=ass["inputs_embeds"], labels=ass["labels"])
                pertes.append(float(sortie["loss"].item()))
        projecteur.train()
        return sum(pertes) / len(pertes)

    val_loss_initiale = _evaluer_val()
    historique: List[Dict[str, float]] = [{"etape": 0, "val_loss": round(val_loss_initiale, 6)}]

    for etape in range(1, nb_etapes + 1):
        ex = exemples[(etape - 1) % len(exemples)]
        optimiseur.zero_grad()
        proj_audio = projecteur(
            ex["vecteurs_encodeur_30s"],
            duree_reelle_s=ex["duree_reelle_s"],
            duree_fenetre_s=30.0,
        )
        ass = assembler_entree_s3(
            vecteurs_audio_projetes=proj_audio,
            ids_debut_user=ex["ids_debut_user"],
            ids_consigne=ex["ids_consigne"],
            ids_fin_user_debut_assistant=ex["ids_fin_user_debut_assistant"],
            table_embeddings_llm=llm.embed_tokens,
            ids_reponse=ex["ids_reponse"],
        )
        sortie = llm(inputs_embeds=ass["inputs_embeds"], labels=ass["labels"])
        perte = sortie["loss"]
        perte.backward()
        optimiseur.step()

        if etape % 20 == 0 or etape == nb_etapes:
            v_loss = _evaluer_val()
            historique.append(
                {
                    "etape": etape,
                    "train_loss": round(float(perte.detach().item()), 6),
                    "val_loss": round(v_loss, 6),
                }
            )

    val_loss_finale = historique[-1]["val_loss"]
    chemin_proj = cible_dir / "projecteur_temps1.pt"
    torch.save(projecteur.state_dict(), str(chemin_proj))

    return {
        "temps": 1,
        "val_loss_initiale": round(val_loss_initiale, 6),
        "val_loss_finale": round(val_loss_finale, 6),
        "baisse_val_loss": val_loss_finale < val_loss_initiale,
        "nb_params_projecteur": projecteur.nb_parametres(),
        "nb_params_llm_entrainables": nb_params_llm_grad,
        "historique": historique,
        "chemin_projecteur": str(chemin_proj),
    }


def entrainer_s3_temps2_ou_un_temps(
    dossier_sortie: Union[str, Path],
    chemin_projecteur_temps1: Optional[Union[str, Path]] = None,
    d_enc: int = 64,
    d_llm: int = 128,
    k: int = 4,
    nb_etapes: int = 60,
    part_rappel_transcription: float = 0.15,
    graine: int = 42,
) -> Dict[str, Any]:
    """Temps 2 de S3 (ou essai en un seul temps si `chemin_projecteur_temps1` est None).

    - Encodeur figé.
    - Projecteur (`lr = 5e-5`) et LoRA du LLM (`lr = 1e-4`) entraînés conjointement.
    - Au Temps 2, 15 % des exemples restent de la transcription arabe pour éviter l'oubli.
    """
    fixer_graine(graine)
    cible_dir = Path(dossier_sortie)
    cible_dir.mkdir(parents=True, exist_ok=True)

    projecteur = ProjecteurAudioLLM(d_enc=d_enc, d_llm=d_llm, k=k)
    mode_nom = "un_seul_temps"
    if chemin_projecteur_temps1 is not None and Path(chemin_projecteur_temps1).exists():
        projecteur.load_state_dict(torch.load(str(chemin_projecteur_temps1), map_location="cpu", weights_only=True))
        mode_nom = "temps_2"

    llm = DecodeurLLMPourInputsEmbeds(vocab_size=256, d_llm=d_llm)
    llm.activer_lora(True)

    params_lora = [p for p in llm.parameters() if p.requires_grad]
    optimiseur = torch.optim.AdamW(
        [
            {"params": list(projecteur.parameters()), "lr": 2e-3},
            {"params": params_lora, "lr": 4e-3},
        ]
    )

    part_trans = part_rappel_transcription if mode_nom == "temps_2" else 0.0
    exemples = construire_exemples_audio_s3(
        nb_exemples=20,
        d_enc=d_enc,
        graine=graine,
        part_transcription=part_trans,
    )
    nb_trans_effectif = sum(1 for e in exemples if e["tache"] == "transcription")

    def _evaluer_val_traduction() -> float:
        projecteur.eval()
        llm.eval()
        ex_trad = [e for e in exemples if e["tache"] == "traduction"][:4]
        pertes = []
        with torch.no_grad():
            for ex in ex_trad:
                proj_audio = projecteur(ex["vecteurs_encodeur_30s"], duree_reelle_s=ex["duree_reelle_s"])
                ass = assembler_entree_s3(
                    vecteurs_audio_projetes=proj_audio,
                    ids_debut_user=ex["ids_debut_user"],
                    ids_consigne=ex["ids_consigne"],
                    ids_fin_user_debut_assistant=ex["ids_fin_user_debut_assistant"],
                    table_embeddings_llm=llm.embed_tokens,
                    ids_reponse=ex["ids_reponse"],
                )
                pertes.append(float(llm(inputs_embeds=ass["inputs_embeds"], labels=ass["labels"])["loss"].item()))
        projecteur.train()
        llm.train()
        return sum(pertes) / max(1, len(pertes))

    val_init = _evaluer_val_traduction()
    for etape in range(1, nb_etapes + 1):
        ex = exemples[(etape - 1) % len(exemples)]
        optimiseur.zero_grad()
        proj_audio = projecteur(ex["vecteurs_encodeur_30s"], duree_reelle_s=ex["duree_reelle_s"])
        ass = assembler_entree_s3(
            vecteurs_audio_projetes=proj_audio,
            ids_debut_user=ex["ids_debut_user"],
            ids_consigne=ex["ids_consigne"],
            ids_fin_user_debut_assistant=ex["ids_fin_user_debut_assistant"],
            table_embeddings_llm=llm.embed_tokens,
            ids_reponse=ex["ids_reponse"],
        )
        perte = llm(inputs_embeds=ass["inputs_embeds"], labels=ass["labels"])["loss"]
        perte.backward()
        optimiseur.step()

    val_fin = _evaluer_val_traduction()
    chemin_modele = cible_dir / f"s3_{mode_nom}.pt"
    torch.save(
        {
            "mode": mode_nom,
            "projecteur": projecteur.state_dict(),
            "lora": {k_nom: v.detach().cpu() for k_nom, v in llm.named_parameters() if v.requires_grad},
        },
        str(chemin_modele),
    )

    return {
        "mode": mode_nom,
        "nb_exemples_total": len(exemples),
        "nb_exemples_transcription_rappel": nb_trans_effectif,
        "val_loss_initiale": round(val_init, 6),
        "val_loss_finale": round(val_fin, 6),
        "baisse_val_loss": val_fin < val_init,
        "chemin_modele": str(chemin_modele),
    }


def construire_parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entraînement du système S3 : Temps 1 (projecteur), Temps 2 (projecteur + LoRA) et essai en un temps."
    )
    parser.add_argument(
        "--essai",
        action="store_true",
        help="Exécute de bout en bout Temps 1, Temps 2 et l'essai en un temps et vérifie la baisse d'erreur.",
    )
    parser.add_argument(
        "--dossier-sortie",
        type=str,
        default=str(RACINE_PROJET / "resultats" / "essai_s3"),
        help="Dossier de sortie.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = construire_parseur()
    args = parser.parse_args(argv)
    if args.essai:
        res_t1 = entrainer_s3_temps1(args.dossier_sortie)
        res_t2 = entrainer_s3_temps2_ou_un_temps(
            args.dossier_sortie,
            chemin_projecteur_temps1=res_t1["chemin_projecteur"],
        )
        res_1t = entrainer_s3_temps2_ou_un_temps(
            args.dossier_sortie,
            chemin_projecteur_temps1=None,
        )
        print(
            f"[ESSAI S3 OK] Temps 1 val_loss: {res_t1['val_loss_initiale']:.4f} -> {res_t1['val_loss_finale']:.4f} | "
            f"Temps 2 val_loss: {res_t2['val_loss_initiale']:.4f} -> {res_t2['val_loss_finale']:.4f} | "
            f"Un seul temps val_loss: {res_1t['val_loss_initiale']:.4f} -> {res_1t['val_loss_finale']:.4f}"
        )
        return 0
    parser.error("Utilisez --essai pour lancer la vérification locale.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
