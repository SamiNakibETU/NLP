"""Préparation des exemples d'entraînement et formatage du chat template pour S2.

Implémente :
- La composition dans les proportions 40 % (D manuel), 30 % (D écoute), 20 % (P presse), 10 % (D6 débuts).
- La détection des noms propres du lexique `donnees/noms/noms_v1.csv` présents dans un segment arabe.
- Le formatage via le chat template du LLM :
  * system : consigne de style (sous-titre français, phrases complètes, noms propres en forme française,
    chiffres en chiffres, aucun ajout) ;
  * user : les deux phrases précédentes déjà traduites si elles existent, la liste des noms propres
    présents dans le segment, puis le texte arabe ;
  * assistant : la traduction française.
- Le masquage des étiquettes (`-100` sur tout le prompt système + utilisateur, calcul de l'erreur
  uniquement sur les tokens de la réponse assistant).
"""

import csv
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import yaml

RACINE_PROJET = Path(__file__).resolve().parents[2]
DOSSIER_DONNEES = RACINE_PROJET / "code" / "donnees"
if str(DOSSIER_DONNEES) not in sys.path:
    sys.path.insert(0, str(DOSSIER_DONNEES))

from normalisation import normaliser_arabe  # noqa: E402

CHEMIN_CONFIG_S2 = RACINE_PROJET / "configs" / "s2_cascade.yaml"
CHEMIN_NOMS_DEFAUT = RACINE_PROJET / "donnees" / "noms" / "noms_v1.csv"


def charger_config_s2(chemin_config: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Charge la configuration YAML de la cascade S2."""
    cible = Path(chemin_config) if chemin_config else CHEMIN_CONFIG_S2
    if not cible.exists():
        raise FileNotFoundError(f"Configuration S2 introuvable : {cible}")
    with open(cible, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def charger_lexique_noms(chemin_csv: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Charge le lexique des noms propres (formes arabes + variantes -> forme française)."""
    cible = Path(chemin_csv) if chemin_csv else CHEMIN_NOMS_DEFAUT
    if not cible.exists():
        return []
    entrees: List[Dict[str, Any]] = []
    with open(cible, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            forme_ar = (row.get("forme_arabe") or "").strip()
            forme_fr = (row.get("forme_francaise") or "").strip()
            if not forme_ar or not forme_fr:
                continue
            variantes_brutes = (row.get("variantes") or "").split("|")
            toutes_formes = [forme_ar] + [v.strip() for v in variantes_brutes if v.strip()]
            formes_norm = [normaliser_arabe(x) for x in toutes_formes if normaliser_arabe(x)]
            entrees.append(
                {
                    "forme_arabe": forme_ar,
                    "forme_francaise": forme_fr,
                    "type": (row.get("type") or "").strip(),
                    "formes_normalisees": formes_norm,
                }
            )
    return entrees


def detecter_noms_propres_segment(
    texte_arabe: str,
    lexique: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Repère les noms propres du lexique présents dans `texte_arabe` (après normalisation arabe)."""
    if not texte_arabe or not lexique:
        return []
    texte_norm = normaliser_arabe(texte_arabe)
    if not texte_norm:
        return []

    trouves: List[Dict[str, str]] = []
    vus_fr = set()
    for item in lexique:
        forme_fr = item["forme_francaise"]
        if forme_fr in vus_fr:
            continue
        for fn in item.get("formes_normalisees", []):
            if fn and len(fn) >= 3 and fn in texte_norm:
                trouves.append(
                    {
                        "forme_arabe": item["forme_arabe"],
                        "forme_francaise": forme_fr,
                    }
                )
                vus_fr.add(forme_fr)
                break
    return trouves


def construire_contenu_utilisateur_cascade(
    texte_arabe: str,
    phrases_precedentes_fr: Optional[Sequence[str]] = None,
    noms_propres: Optional[Sequence[Dict[str, str]]] = None,
) -> str:
    """Construit le tour utilisateur avec contexte des phrases précédentes et noms propres.

    Ordre exigé :
    1. Les deux phrases précédentes déjà traduites si elles existent ;
    2. La liste des noms propres présents dans le segment ;
    3. Le texte arabe à traduire.
    """
    blocs: List[str] = []

    if phrases_precedentes_fr:
        dernieres = [p.strip() for p in phrases_precedentes_fr if p and p.strip()][-2:]
        if dernieres:
            blocs.append("Contexte précédent (français) :\n" + "\n".join(f"- {p}" for p in dernieres))

    if noms_propres:
        couples = [
            f"{n['forme_arabe']} -> {n['forme_francaise']}"
            for n in noms_propres
            if n.get("forme_arabe") and n.get("forme_francaise")
        ]
        if couples:
            blocs.append("Noms propres : " + " ; ".join(couples))

    blocs.append(f"Texte arabe : {texte_arabe.strip()}")
    return "\n\n".join(blocs)


def construire_messages_cascade(
    texte_arabe: str,
    traduction_fr: Optional[str] = None,
    phrases_precedentes_fr: Optional[Sequence[str]] = None,
    noms_propres: Optional[Sequence[Dict[str, str]]] = None,
    consigne_systeme: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Construit la liste de messages `[system, user, (assistant)]` pour le chat template."""
    sys_txt = (
        consigne_systeme
        or (
            "Tu es un système de sous-titrage français pour journaux télévisés en arabe. "
            "Produis un sous-titre français composé de phrases complètes, avec les noms propres "
            "dans leur forme française, les chiffres écrits en chiffres, sans aucun ajout ni commentaire."
        )
    ).strip()

    user_txt = construire_contenu_utilisateur_cascade(
        texte_arabe=texte_arabe,
        phrases_precedentes_fr=phrases_precedentes_fr,
        noms_propres=noms_propres,
    )

    messages = [
        {"role": "system", "content": sys_txt},
        {"role": "user", "content": user_txt},
    ]
    if traduction_fr is not None:
        messages.append({"role": "assistant", "content": traduction_fr.strip()})
    return messages


def formater_chat_template_cascade(
    messages: Sequence[Dict[str, str]],
    tokenizer: Optional[Any] = None,
    add_generation_prompt: bool = False,
) -> str:
    """Formate les messages via `tokenizer.apply_chat_template` ou le format ChatML Qwen3 standard."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    morceaux: List[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        morceaux.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    if add_generation_prompt:
        morceaux.append("<|im_start|>assistant\n")
    return "\n".join(morceaux)


def encoder_exemple_avec_masquage_s2(
    texte_arabe: str,
    traduction_fr: str,
    phrases_precedentes_fr: Optional[Sequence[str]] = None,
    noms_propres: Optional[Sequence[Dict[str, str]]] = None,
    consigne_systeme: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    max_seq_length: int = 512,
    vocab_size_factice: int = 2048,
) -> Dict[str, Any]:
    """Encode un exemple S2 et masque avec `-100` toutes les positions hors réponse assistant."""
    messages_prompt = construire_messages_cascade(
        texte_arabe=texte_arabe,
        traduction_fr=None,
        phrases_precedentes_fr=phrases_precedentes_fr,
        noms_propres=noms_propres,
        consigne_systeme=consigne_systeme,
    )
    messages_complets = construire_messages_cascade(
        texte_arabe=texte_arabe,
        traduction_fr=traduction_fr,
        phrases_precedentes_fr=phrases_precedentes_fr,
        noms_propres=noms_propres,
        consigne_systeme=consigne_systeme,
    )

    texte_prompt = formater_chat_template_cascade(
        messages_prompt, tokenizer=tokenizer, add_generation_prompt=True
    )
    texte_complet = formater_chat_template_cascade(
        messages_complets, tokenizer=tokenizer, add_generation_prompt=False
    )

    if tokenizer is not None and callable(tokenizer):
        ids_prompt = tokenizer(texte_prompt, add_special_tokens=False)["input_ids"]
        ids_complets = tokenizer(
            texte_complet,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
        )["input_ids"]
    else:
        # Tokenisation déterministe par mots/jetons bornée à vocab_size_factice pour le mode essai hors-ligne
        import hashlib

        def _tok_mot(m: str) -> int:
            h = int.from_bytes(hashlib.md5(m.encode("utf-8")).digest()[:4], "little")
            return 4 + (h % max(1, vocab_size_factice - 4))

        mots_prompt = texte_prompt.split()
        mots_reponse = (traduction_fr.strip() + " <|im_end|>").split()
        ids_reponse = [_tok_mot(m) for m in mots_reponse]
        max_prompt = max(1, max_seq_length - len(ids_reponse))
        ids_prompt = [_tok_mot(m) for m in mots_prompt[:max_prompt]]
        ids_complets = (ids_prompt + ids_reponse)[:max_seq_length]

    nb_masques = min(len(ids_prompt), len(ids_complets))
    labels = [-100] * nb_masques + list(ids_complets[nb_masques:])

    return {
        "input_ids": torch.tensor(ids_complets, dtype=torch.long),
        "attention_mask": torch.ones(len(ids_complets), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "nb_tokens_prompt_masques": nb_masques,
        "nb_tokens_reponse_supervises": sum(1 for x in labels if x != -100),
        "chat_template_prompt": texte_prompt,
        "chat_template_complet": texte_complet,
    }


def composer_jeu_entrainement_s2(
    exemples_d_manuel: Sequence[Dict[str, Any]],
    exemples_d_ecoute: Sequence[Dict[str, Any]],
    exemples_p_presse: Sequence[Dict[str, Any]],
    exemples_d6_debut: Sequence[Dict[str, Any]],
    nb_total: int = 200,
    proportions: Optional[Dict[str, float]] = None,
    graine: int = 42,
    lexique_noms: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Compose un jeu d'entraînement S2 selon les proportions exactes de la section 4.1 (40/30/20/10)."""
    props = proportions or {
        "d_transcription_manuelle": 0.40,
        "d_sortie_ecoute": 0.30,
        "p_presse": 0.20,
        "d6_debut_phrase": 0.10,
    }
    total_poids = sum(props.values())
    if abs(total_poids - 1.0) > 1e-6:
        raise ValueError(f"La somme des proportions doit valoir 1.0 (reçu : {total_poids})")

    n_d_man = int(round(nb_total * props["d_transcription_manuelle"]))
    n_d_eco = int(round(nb_total * props["d_sortie_ecoute"]))
    n_p = int(round(nb_total * props["p_presse"]))
    n_d6 = nb_total - (n_d_man + n_d_eco + n_p)

    rng = random.Random(graine)

    def _tirer(pool: Sequence[Dict[str, Any]], k: int, source_tag: str) -> List[Dict[str, Any]]:
        if not pool or k <= 0:
            return []
        res: List[Dict[str, Any]] = []
        for idx in range(k):
            item = dict(pool[idx % len(pool)])
            if source_tag == "d_sortie_ecoute":
                ar_txt = item.get("ar_ecoute") or item.get("ar") or ""
                fr_txt = item.get("fr") or ""
            elif source_tag == "d6_debut_phrase":
                ar_txt = item.get("ar_debut") or item.get("ar") or ""
                fr_txt = item.get("fr_debut") or item.get("fr") or ""
            else:
                ar_txt = item.get("ar") or ""
                fr_txt = item.get("fr") or ""

            noms = item.get("noms_propres")
            if noms is None and lexique_noms is not None:
                noms = detecter_noms_propres_segment(ar_txt, lexique_noms)

            res.append(
                {
                    "id": f"{source_tag}_{idx:05d}",
                    "source_part": source_tag,
                    "ar": str(ar_txt).strip(),
                    "fr": str(fr_txt).strip(),
                    "phrases_precedentes_fr": item.get("phrases_precedentes_fr", []),
                    "noms_propres": noms or [],
                }
            )
        return res

    jeu = (
        _tirer(exemples_d_manuel, n_d_man, "d_transcription_manuelle")
        + _tirer(exemples_d_ecoute, n_d_eco, "d_sortie_ecoute")
        + _tirer(exemples_p_presse, n_p, "p_presse")
        + _tirer(exemples_d6_debut, n_d6, "d6_debut_phrase")
    )
    rng.shuffle(jeu)

    comptes = {
        "d_transcription_manuelle": n_d_man,
        "d_sortie_ecoute": n_d_eco,
        "p_presse": n_p,
        "d6_debut_phrase": n_d6,
        "total": len(jeu),
    }
    return jeu, comptes


def generer_exemples_factices_s2(
    nb_total: int = 200,
    graine: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Génère `nb_total` exemples factices (marqués comme tels) respectant les 4 sources 40/30/20/10."""
    lexique = [
        {
            "forme_arabe": "بيل كلينتون",
            "forme_francaise": "Bill Clinton",
            "type": "personne",
            "formes_normalisees": [normaliser_arabe("بيل كلينتون"), normaliser_arabe("كلينتون")],
        },
        {
            "forme_arabe": "الأمم المتحدة",
            "forme_francaise": "Nations unies",
            "type": "organisation",
            "formes_normalisees": [normaliser_arabe("الأمم المتحدة")],
        },
    ]

    d_manuel = [
        {
            "ar": "عقد مجلس الأمن التابع للأمم المتحدة اجتماعا طارئا لمناقشة الأوضاع في المنطقة.",
            "fr": "Le Conseil de sécurité des Nations unies a tenu une réunion d'urgence pour discuter de la situation dans la région.",
            "phrases_precedentes_fr": [
                "Bonsoir et bienvenue dans ce journal télévisé.",
                "Les tensions diplomatiques restent au cœur de l'actualité internationale.",
            ],
        },
        {
            "ar": "التقى الرئيس الأسبق بيل كلينتون بالوفد الدبلوماسي لعرض مبادرة السلام.",
            "fr": "L'ancien président Bill Clinton a rencontré la délégation diplomatique pour présenter l'initiative de paix.",
            "phrases_precedentes_fr": [
                "Les pourparlers se poursuivent depuis trois jours.",
            ],
        },
    ]
    d_ecoute = [
        {
            "ar_ecoute": "اعلن وزير الماليه اليوم عن ميزانيه عام 2026 بقيمه 45 مليار دولار",
            "fr": "Le ministre des Finances a annoncé aujourd'hui le budget 2026 d'un montant de 45 milliards de dollars.",
            "phrases_precedentes_fr": [
                "Passons maintenant aux nouvelles économiques.",
            ],
        }
    ]
    p_presse = [
        {
            "ar": "سجلت الأسواق الأوروبية ارتفاعا ملحوظا بنسبة 2.5 بالمئة عقب إعلان البنك المركزي.",
            "fr": "Les marchés européens ont enregistré une hausse notable de 2,5 % après l'annonce de la banque centrale.",
            "phrases_precedentes_fr": [],
        }
    ]
    d6_debut = [
        {
            "ar_debut": "أكد المتحدث باسم الأمم المتحدة أن المساعدات",
            "fr_debut": "Le porte-parole des Nations unies a confirmé que l'aide",
            "phrases_precedentes_fr": [
                "Sur le terrain humanitaire, les convois attendent toujours à la frontière.",
            ],
        }
    ]

    return composer_jeu_entrainement_s2(
        exemples_d_manuel=d_manuel,
        exemples_d_ecoute=d_ecoute,
        exemples_p_presse=p_presse,
        exemples_d6_debut=d6_debut,
        nb_total=nb_total,
        graine=graine,
        lexique_noms=lexique,
    )
