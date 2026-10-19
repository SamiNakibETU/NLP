"""Comparaison des modèles d'écoute et des LLM candidats.

Principes :
- Section 3.1 (Modèle d'écoute) :
  Candidats : `openai/whisper-large-v3`, Cohere Transcribe Arabic (`CohereLabs/cohere-transcribe-03-2026`),
  `Qwen/Qwen3-ASR-1.7B`. Si un modèle n'existe pas sous le nom attendu, le signaler au lieu de le remplacer.
  Choix sur les 20 extraits de réglage (ou FLEURS en attendant R) : on mesure le WER après `normaliser_arabe`
  et la vitesse (RTF). On retient le plus précis. Si deux modèles sont à moins de 3 points l'un de l'autre
  (proposé), on retient celui dont l'encodeur est le plus simple à réutiliser dans S3.
- Section 3.2 (LLM) :
  Candidats : `Qwen/Qwen3-4B`, Gemma 4 E4B (`google/gemma-4-E4B-it`) en mode texte,
  `Qwen/Qwen3-1.7B` comme option légère. Choix en traduction de la transcription manuelle vers le français,
  sans entraînement, au score chrF++.
  La dimension des vecteurs du LLM (`hidden_size`) se lit dans son `config.json`.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import yaml

RACINE_PROJET = Path(__file__).resolve().parents[2]
DOSSIER_ECOUTE = Path(__file__).resolve().parent
DOSSIER_MESURES = RACINE_PROJET / "code" / "mesures"

for p in (DOSSIER_ECOUTE, DOSSIER_MESURES):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from encodeur import lire_dimension_encodeur, lire_dimension_llm  # noqa: E402
from scores import calculer_chrf_bleu, calculer_rtf, calculer_wer_cer  # noqa: E402
from tableau_resultats import CHEMIN_RESULTATS_DEFAUT, enregistrer_resultat, lire_resultats  # noqa: E402

CHEMIN_CONFIG_DEFAUT = RACINE_PROJET / "configs" / "choix_briques.yaml"


def charger_config_choix_briques(chemin_config: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Charge `configs/choix_briques.yaml`."""
    cible = Path(chemin_config) if chemin_config else CHEMIN_CONFIG_DEFAUT
    if not cible.exists():
        raise FileNotFoundError(f"Configuration introuvable : {cible}")
    with open(cible, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def choisir_meilleur_modele_ecoute(
    resultats_candidats: Sequence[Dict[str, Any]],
    ecart_max_pts: float = 3.0,
) -> Dict[str, Any]:
    """Sélectionne le modèle d'écoute selon la règle de la section 3.1.

    - Ignore les candidats dont le score WER est `"non mesuré"` ou `None`.
    - Identifie le WER minimal `wer_min`.
    - Parmi tous les candidats situés à moins de `ecart_max_pts` points de `wer_min`
      (`wer_pct <= wer_min + ecart_max_pts`), retient celui dont l'encodeur est le plus
      simple à réutiliser dans S3 (`simplicite_s3` minimal, puis WER minimal en cas d'égalité).
    """
    valides = []
    for c in resultats_candidats:
        wer = c.get("wer_pct")
        if isinstance(wer, (int, float)) and not isinstance(wer, bool):
            valides.append(c)

    if not valides:
        return {
            "retenu": None,
            "motif": "Aucun candidat n'a de score WER numérique mesuré.",
            "candidats_proches": [],
        }

    wer_min = min(float(c["wer_pct"]) for c in valides)
    meilleur_brut = min(valides, key=lambda c: float(c["wer_pct"]))

    proches = [
        c for c in valides if (float(c["wer_pct"]) - wer_min) <= float(ecart_max_pts) + 1e-9
    ]
    proches_tries = sorted(
        proches,
        key=lambda c: (int(c.get("simplicite_s3", 99)), float(c["wer_pct"]), float(c.get("rtf", 999.0))),
    )
    retenu = proches_tries[0]

    if len(proches) > 1 and retenu["nom"] != meilleur_brut["nom"]:
        motif = (
            f"Écart de WER <= {ecart_max_pts} pts entre {retenu['nom']} ({retenu['wer_pct']} %) "
            f"et {meilleur_brut['nom']} ({meilleur_brut['wer_pct']} %) : choix de l'encodeur "
            f"le plus simple pour S3 (simplicite_s3={retenu.get('simplicite_s3')})."
        )
    else:
        motif = (
            f"Modèle le plus précis retenu ({retenu['nom']}, WER={retenu['wer_pct']} %, "
            f"simplicite_s3={retenu.get('simplicite_s3')})."
        )

    return {
        "retenu": retenu,
        "wer_min": round(wer_min, 2),
        "ecart_max_pts": float(ecart_max_pts),
        "candidats_proches": [c["nom"] for c in proches_tries],
        "motif": motif,
    }


def choisir_meilleur_llm(resultats_candidats: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Sélectionne le LLM candidat ayant le meilleur score chrF++."""
    valides = [
        c
        for c in resultats_candidats
        if isinstance(c.get("chrf_pp"), (int, float)) and not isinstance(c.get("chrf_pp"), bool)
    ]
    if not valides:
        return {
            "retenu": None,
            "motif": "Aucun candidat LLM n'a de score chrF++ numérique mesuré.",
        }

    tries = sorted(
        valides,
        key=lambda c: (-float(c["chrf_pp"]), float(c.get("rtf", 999.0))),
    )
    retenu = tries[0]
    return {
        "retenu": retenu,
        "motif": f"Meilleur score chrF++ sans entraînement ({retenu['nom']} : chrF++={retenu['chrf_pp']}).",
    }


def charger_echantillon_fleurs_ar_fr(nb_extraits: int = 20, split: str = "test") -> List[Dict[str, Any]]:
    """Charge et aligne par identifiant `id` les extraits arabe (`ar_eg`) et français (`fr_fr`) de FLEURS."""
    from datasets import load_dataset

    ds_ar = load_dataset("google/fleurs", "ar_eg", split=split, streaming=True)
    ds_fr = load_dataset("google/fleurs", "fr_fr", split=split, streaming=True)

    table_fr: Dict[int, str] = {}
    for ex_fr in ds_fr:
        table_fr[int(ex_fr["id"])] = str(ex_fr["transcription"]).strip()
        if len(table_fr) >= nb_extraits * 4:
            break

    fiches: List[Dict[str, Any]] = []
    for ex_ar in ds_ar:
        ex_id = int(ex_ar["id"])
        if ex_id in table_fr:
            audio_info = ex_ar["audio"]
            arr = np.asarray(audio_info["array"], dtype=np.float32)
            sr = int(audio_info["sampling_rate"])
            duree_s = max(0.05, len(arr) / float(sr))
            fiches.append(
                {
                    "id": f"fleurs_{ex_id}",
                    "ar": str(ex_ar["transcription"]).strip(),
                    "fr": table_fr[ex_id],
                    "duree_s": round(duree_s, 3),
                    "audio": arr,
                    "sr": sr,
                }
            )
            if len(fiches) >= nb_extraits:
                break
    return fiches


def evaluer_candidats_ecoute_reel(
    fiches: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    jeu: str = "fleurs_ar_fr",
    chemin_csv: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Évalue sur GPU/CPU les candidats d'écoute configurés et enregistre les scores dans resultats.csv."""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    refs_ar = [str(f["ar"]) for f in fiches]
    duree_totale = sum(float(f["duree_s"]) for f in fiches)
    resultats: List[Dict[str, Any]] = []

    for cand in config.get("candidats_ecoute", []):
        nom = cand["nom"]
        modele_id = cand["modele_id"]
        trust_code = bool(cand.get("trust_remote_code", False))
        try:
            proc = AutoProcessor.from_pretrained(modele_id, trust_remote_code=trust_code)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                modele_id, torch_dtype=dtype, trust_remote_code=trust_code
            ).to(device).eval()
            d_enc = lire_dimension_encodeur(model.config)

            t0 = time.perf_counter()
            hyps: List[str] = []
            for f in fiches:
                inputs = proc(f["audio"], sampling_rate=int(f["sr"]), return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    out_ids = model.generate(**inputs, max_new_tokens=128)
                txt = proc.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
                hyps.append(txt)
            temps_s = time.perf_counter() - t0
            scores_asr = calculer_wer_cer(refs_ar, hyps)
            rtf = calculer_rtf(temps_s, duree_totale)

            entree = {
                "nom": nom,
                "modele_id": modele_id,
                "d_enc": d_enc,
                "simplicite_s3": cand.get("simplicite_s3", 99),
                "wer_pct": scores_asr["wer_pct"],
                "cer_pct": scores_asr["cer_pct"],
                "rtf": rtf,
                "statut": "mesuré",
            }
        except Exception as exc:
            entree = {
                "nom": nom,
                "modele_id": modele_id,
                "d_enc": cand.get("d_enc", "non mesuré"),
                "simplicite_s3": cand.get("simplicite_s3", 99),
                "wer_pct": "non mesuré",
                "cer_pct": "non mesuré",
                "rtf": "non mesuré",
                "statut": f"non mesuré ({type(exc).__name__}: {exc})",
            }

        resultats.append(entree)
        enregistrer_resultat(
            systeme=f"ECOUTE:{nom}",
            version_modele=modele_id,
            jeu=jeu,
            mesure="WER (%)",
            valeur=entree["wer_pct"],
            intervalle=entree["statut"],
            configuration="configs/choix_briques.yaml",
            chemin_csv=chemin_csv,
        )
        enregistrer_resultat(
            systeme=f"ECOUTE:{nom}",
            version_modele=modele_id,
            jeu=jeu,
            mesure="RTF",
            valeur=entree["rtf"],
            intervalle=entree["statut"],
            configuration="configs/choix_briques.yaml",
            chemin_csv=chemin_csv,
        )

    return resultats


def evaluer_candidats_llm_reel(
    fiches: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    jeu: str = "fleurs_ar_fr",
    chemin_csv: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Évalue les LLM candidats en traduction zéro-shot (4 bits nf4, float16 sur T4, température 0)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    refs_fr = [str(f["fr"]) for f in fiches]
    duree_totale = sum(float(f.get("duree_s", 5.0)) for f in fiches)
    consigne = str(config.get("consigne_traduction_zero_shot", "Traduis en français.")).strip()
    cfg_load = config.get("chargement_llm", {})
    max_new_tokens = int(cfg_load.get("max_new_tokens", 128))
    resultats: List[Dict[str, Any]] = []

    for cand in config.get("candidats_llm", []):
        nom = cand["nom"]
        modele_id = cand["modele_id"]
        try:
            tok = AutoTokenizer.from_pretrained(modele_id)
            kwargs_model: Dict[str, Any] = {"device_map": "auto"}
            if device == "cuda" and cfg_load.get("quantification_4bits", True):
                kwargs_model["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=cfg_load.get("bnb_4bit_quant_type", "nf4"),
                    bnb_4bit_compute_dtype=torch.float16,
                )
                kwargs_model["torch_dtype"] = torch.float16
            else:
                kwargs_model["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32

            model = AutoModelForCausalLM.from_pretrained(modele_id, **kwargs_model).eval()
            d_llm = lire_dimension_llm(model.config)

            t0 = time.perf_counter()
            hyps: List[str] = []
            for f in fiches:
                messages = [
                    {"role": "system", "content": consigne},
                    {"role": "user", "content": str(f["ar"])},
                ]
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tok(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                    )
                gen_ids = out[0][inputs["input_ids"].shape[1] :]
                hyps.append(tok.decode(gen_ids, skip_special_tokens=True).strip())

            temps_s = time.perf_counter() - t0
            scores_mt = calculer_chrf_bleu(refs_fr, hyps)
            rtf = calculer_rtf(temps_s, duree_totale)

            entree = {
                "nom": nom,
                "modele_id": modele_id,
                "d_llm": d_llm,
                "chrf_pp": scores_mt["chrf_pp"],
                "bleu": scores_mt["bleu"],
                "rtf": rtf,
                "statut": "mesuré",
            }
        except Exception as exc:
            entree = {
                "nom": nom,
                "modele_id": modele_id,
                "d_llm": cand.get("hidden_size", "non mesuré"),
                "chrf_pp": "non mesuré",
                "bleu": "non mesuré",
                "rtf": "non mesuré",
                "statut": f"non mesuré ({type(exc).__name__}: {exc})",
            }

        resultats.append(entree)
        enregistrer_resultat(
            systeme=f"LLM:{nom}",
            version_modele=modele_id,
            jeu=jeu,
            mesure="chrF++",
            valeur=entree["chrf_pp"],
            intervalle=entree["statut"],
            configuration="configs/choix_briques.yaml",
            chemin_csv=chemin_csv,
        )
        enregistrer_resultat(
            systeme=f"LLM:{nom}",
            version_modele=modele_id,
            jeu=jeu,
            mesure="BLEU",
            valeur=entree["bleu"],
            intervalle=entree["statut"],
            configuration="configs/choix_briques.yaml",
            chemin_csv=chemin_csv,
        )

    return resultats


def enregistrer_etat_non_mesure_local(
    config: Optional[Dict[str, Any]] = None,
    chemin_csv: Optional[Union[str, Path]] = None,
) -> List[Dict[str, str]]:
    """Inscrit dans `resultats/resultats.csv` l'état honnête 'non mesuré' tant que le run Kaggle GPU n'a pas tourné."""
    cfg = config or charger_config_choix_briques()
    cible = Path(chemin_csv) if chemin_csv else CHEMIN_RESULTATS_DEFAUT
    existantes = {(r["systeme"], r["jeu"], r["mesure"]) for r in lire_resultats(cible)}
    ajoutees: List[Dict[str, str]] = []

    a_inscrire = [
        ("S0", "openai/whisper-small+Helsinki-NLP/opus-mt-ar-fr", "fleurs_ar_fr", "chrF++", "configs/s0.yaml"),
        ("S0", "openai/whisper-small+Helsinki-NLP/opus-mt-ar-fr", "fleurs_ar_fr", "BLEU", "configs/s0.yaml"),
        ("S0", "openai/whisper-small+Helsinki-NLP/opus-mt-ar-fr", "fleurs_ar_fr", "WER (%)", "configs/s0.yaml"),
        ("S0", "openai/whisper-small+Helsinki-NLP/opus-mt-ar-fr", "fleurs_ar_fr", "RTF", "configs/s0.yaml"),
    ]
    for cand in cfg.get("candidats_ecoute", []):
        a_inscrire.append(
            (f"ECOUTE:{cand['nom']}", cand["modele_id"], "fleurs_ar_fr", "WER (%)", "configs/choix_briques.yaml")
        )
        a_inscrire.append(
            (f"ECOUTE:{cand['nom']}", cand["modele_id"], "R", "WER (%)", "configs/choix_briques.yaml")
        )
    for cand in cfg.get("candidats_llm", []):
        a_inscrire.append(
            (f"LLM:{cand['nom']}", cand["modele_id"], "fleurs_ar_fr", "chrF++", "configs/choix_briques.yaml")
        )
        a_inscrire.append(
            (f"LLM:{cand['nom']}", cand["modele_id"], "R", "chrF++", "configs/choix_briques.yaml")
        )

    for systeme, ver, jeu, mesure, conf in a_inscrire:
        if (systeme, jeu, mesure) not in existantes:
            ajoutees.append(
                enregistrer_resultat(
                    systeme=systeme,
                    version_modele=ver,
                    jeu=jeu,
                    mesure=mesure,
                    valeur="non mesuré",
                    intervalle="en_attente_gpu_kaggle_et_jeu_R",
                    configuration=conf,
                    chemin_csv=cible,
                )
            )
    return ajoutees


def construire_jeu_essai_briques() -> List[Dict[str, Any]]:
    """Construit un mini-jeu factice contrôlé (marqué comme tel) pour le mode --essai."""
    sr = 16000
    duree_s = 3.0
    t = np.linspace(0.0, duree_s, int(sr * duree_s), endpoint=False, dtype=np.float32)
    audio = 0.15 * np.sin(2.0 * np.pi * 200.0 * t).astype(np.float32)
    return [
        {
            "id": "essai_briques_01",
            "ar": "أعلن البنك المركزي اليوم عن خفض أسعار الفائدة لدعم الاستثمار.",
            "fr": "La banque centrale a annoncé aujourd'hui une baisse des taux d'intérêt pour soutenir l'investissement.",
            "duree_s": duree_s,
            "type": "plateau",
            "audio": audio,
            "sr": sr,
        },
        {
            "id": "essai_briques_02",
            "ar": "تواصل فرق الإنقاذ عملها في المناطق المتضررة من الفيضانات.",
            "fr": "Les équipes de secours poursuivent leur travail dans les zones touchées par les inondations.",
            "duree_s": duree_s,
            "type": "reportage",
            "audio": audio,
            "sr": sr,
        },
    ]


def evaluer_briques_mode_essai(
    config: Optional[Dict[str, Any]] = None,
    dossier_sortie: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Exécute la comparaison de bout en bout sur des exemples factices contrôlés (mode --essai)."""
    cfg = config or charger_config_choix_briques()
    fiches = construire_jeu_essai_briques()
    cible_dir = Path(dossier_sortie) if dossier_sortie else (RACINE_PROJET / "resultats" / "essai_choix_briques")
    cible_dir.mkdir(parents=True, exist_ok=True)
    chemin_csv = cible_dir / "resultats_essai_briques.csv"

    refs_ar = [f["ar"] for f in fiches]
    refs_fr = [f["fr"] for f in fiches]
    duree_totale = sum(float(f["duree_s"]) for f in fiches)

    # Simulation contrôlée : whisper-large-v3 et cohere-transcribe ont un écart < 3 pts de WER
    simulations_ecoute = {
        "whisper-large-v3": {
            "hyps": [
                "أعلن البنك المركزي اليوم عن خفض أسعار الفائدة لدعم الاستثمار",
                "تواصل فرق الإنقاذ عملها في المناطق المتضررة من الفيضانات",
            ],
            "temps_s": 0.8,
        },
        "cohere-transcribe-arabic": {
            "hyps": [
                "اعلن البنك المركزي اليوم عن خفض اسعار الفائده لدعم الاستثمار",
                "تواصل فرق الانقاذ عملها في المناطق المتضرره من الفيضانات",
            ],
            "temps_s": 0.9,
        },
        "qwen3-asr-1.7b": {
            "hyps": [
                "أعلن البنك اليوم عن خفض أسعار الفائدة لدعم",
                "تواصل فرق الإنقاذ عملها في المناطق",
            ],
            "temps_s": 0.7,
        },
    }

    resultats_ecoute: List[Dict[str, Any]] = []
    for cand in cfg.get("candidats_ecoute", []):
        nom = cand["nom"]
        sim = simulations_ecoute.get(nom, simulations_ecoute["whisper-large-v3"])
        scores_asr = calculer_wer_cer(refs_ar, sim["hyps"])
        rtf = calculer_rtf(sim["temps_s"], duree_totale)
        entree = {
            "nom": nom,
            "modele_id": cand["modele_id"],
            "simplicite_s3": cand.get("simplicite_s3", 99),
            "wer_pct": scores_asr["wer_pct"],
            "cer_pct": scores_asr["cer_pct"],
            "rtf": rtf,
            "jeu": "essai_factice",
        }
        resultats_ecoute.append(entree)
        enregistrer_resultat(
            systeme=f"ECOUTE:{nom}",
            version_modele=cand["modele_id"],
            jeu="essai_factice",
            mesure="WER (%)",
            valeur=scores_asr["wer_pct"],
            intervalle="mode_essai_factice",
            configuration="configs/choix_briques.yaml",
            chemin_csv=chemin_csv,
        )

    ecart_max = float(cfg.get("regles_choix", {}).get("ecart_choix_ecoute_max_pts", 3.0))
    choix_ecoute = choisir_meilleur_modele_ecoute(resultats_ecoute, ecart_max_pts=ecart_max)

    simulations_llm = {
        "qwen3-4b": {
            "hyps": [
                "La banque centrale a annoncé aujourd'hui une baisse des taux d'intérêt pour soutenir l'investissement.",
                "Les équipes de secours poursuivent leur travail dans les zones touchées par les inondations.",
            ],
            "temps_s": 1.1,
        },
        "gemma-4-e4b": {
            "hyps": [
                "La banque centrale a annoncé aujourd'hui une réduction des taux pour soutenir l'investissement.",
                "Les équipes de secours continuent leur travail dans les régions touchées par les inondations.",
            ],
            "temps_s": 1.2,
        },
        "qwen3-1.7b": {
            "hyps": [
                "La banque centrale annonce une baisse des taux d'intérêt aujourd'hui.",
                "Les secours travaillent dans les zones inondées.",
            ],
            "temps_s": 0.6,
        },
    }

    resultats_llm: List[Dict[str, Any]] = []
    for cand in cfg.get("candidats_llm", []):
        nom = cand["nom"]
        sim = simulations_llm.get(nom, simulations_llm["qwen3-4b"])
        scores_mt = calculer_chrf_bleu(refs_fr, sim["hyps"])
        rtf = calculer_rtf(sim["temps_s"], duree_totale)
        entree = {
            "nom": nom,
            "modele_id": cand["modele_id"],
            "chrf_pp": scores_mt["chrf_pp"],
            "bleu": scores_mt["bleu"],
            "rtf": rtf,
            "jeu": "essai_factice",
        }
        resultats_llm.append(entree)
        enregistrer_resultat(
            systeme=f"LLM:{nom}",
            version_modele=cand["modele_id"],
            jeu="essai_factice",
            mesure="chrF++",
            valeur=scores_mt["chrf_pp"],
            intervalle="mode_essai_factice",
            configuration="configs/choix_briques.yaml",
            chemin_csv=chemin_csv,
        )

    choix_llm = choisir_meilleur_llm(resultats_llm)

    synthese = {
        "mode": "essai_factice",
        "resultats_ecoute": resultats_ecoute,
        "choix_ecoute": choix_ecoute,
        "resultats_llm": resultats_llm,
        "choix_llm": choix_llm,
        "chemin_csv": str(chemin_csv),
    }
    chemin_json = cible_dir / "synthese_essai_briques.json"
    with open(chemin_json, "w", encoding="utf-8") as f:
        json.dump(synthese, f, ensure_ascii=False, indent=2)
    return synthese


def construire_parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare les modèles d'écoute et les LLM candidats."
    )
    parser.add_argument(
        "--essai",
        action="store_true",
        help="Exécute la comparaison sur un mini-jeu factice contrôlé sans télécharger les modèles lourds.",
    )
    parser.add_argument(
        "--enregistrer-non-mesure",
        action="store_true",
        help="Initialise resultats/resultats.csv avec les lignes 'non mesuré' en attente du run Kaggle GPU et du jeu R.",
    )
    parser.add_argument(
        "--evaluer-reel",
        action="store_true",
        help="Lance l'évaluation réelle des candidats ASR et LLM sur FLEURS (ou sur --references pour le jeu R).",
    )
    parser.add_argument(
        "--jeu",
        type=str,
        default="fleurs_ar_fr",
        help="Nom du jeu évalué en mode réel (fleurs_ar_fr ou R).",
    )
    parser.add_argument(
        "--nb-extraits",
        type=int,
        default=20,
        help="Nombre d'extraits FLEURS à évaluer en attendant le jeu R (défaut: 20).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CHEMIN_CONFIG_DEFAUT),
        help="Chemin du fichier de configuration YAML (défaut: configs/choix_briques.yaml).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = construire_parseur()
    args = parser.parse_args(argv)
    cfg = charger_config_choix_briques(args.config)

    if args.enregistrer_non_mesure:
        lignes = enregistrer_etat_non_mesure_local(cfg)
        print(f"[TABLEAU RESULTATS] {len(lignes)} lignes initialisées dans {CHEMIN_RESULTATS_DEFAUT}")
        return 0

    if args.essai:
        res = evaluer_briques_mode_essai(cfg)
        print(
            f"[ESSAI BRIQUES OK] Écoute retenue : {res['choix_ecoute']['retenu']['nom']} | "
            f"LLM retenu : {res['choix_llm']['retenu']['nom']}"
        )
        return 0

    if args.evaluer_reel:
        fiches = charger_echantillon_fleurs_ar_fr(nb_extraits=args.nb_extraits)
        res_ecoute = evaluer_candidats_ecoute_reel(fiches, cfg, jeu=args.jeu)
        res_llm = evaluer_candidats_llm_reel(fiches, cfg, jeu=args.jeu)
        ecart_max = float(cfg.get("regles_choix", {}).get("ecart_choix_ecoute_max_pts", 3.0))
        choix_e = choisir_meilleur_modele_ecoute(res_ecoute, ecart_max_pts=ecart_max)
        choix_l = choisir_meilleur_llm(res_llm)
        print(json.dumps({"choix_ecoute": choix_e, "choix_llm": choix_l}, ensure_ascii=False, indent=2))
        return 0

    parser.error("Spécifiez --essai, --enregistrer-non-mesure ou --evaluer-reel.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
