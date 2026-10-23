"""Fonctions communes aux kernels Kaggle (FLEURS, gabarits de traduction, génération, métriques).

Ce fichier est copié dans le jeu de données privé ``nlp-code`` (voir kaggle/preparer_datasets.py).
Il ne lit aucun secret : le jeton HF n'est demandé qu'au kernel qui charge S3, via kaggle_secrets.
"""
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

URL_FLEURS = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
MOTIF_ARABE = re.compile(r"[؀-ۿ]")
MOTIF_INTRODUCTION = re.compile(r"^(voici|pour traduire)[^:\n]{0,80}:\s*", re.IGNORECASE)


def trouver_racine_code() -> Path:
    """Dossier du jeu de données nlp-code (contient code/ et configs/)."""
    for f in Path("/kaggle/input").rglob("kaggle_commun.py"):
        if (f.parent / "code").is_dir():
            return f.parent
    raise FileNotFoundError("jeu de données nlp-code introuvable")


def preparer_chemins(racine: Path) -> None:
    for sous in ("code/projecteur", "code/ecoute", "code/demo", "code/mesures", "code/donnees", "."):
        p = str(racine / sous)
        if p not in sys.path:
            sys.path.insert(0, p)


def pip(*paquets: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *paquets], check=True)


# ----------------------------------------------------------------------------------- FLEURS
def telecharger(url: str, dest: Path) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for bloc in r.iter_content(1 << 20):
                f.write(bloc)
    return dest


def lire_tsv(chemin: Path) -> List[List[str]]:
    return [l.rstrip("\n").split("\t") for l in open(chemin, encoding="utf-8") if l.strip()]


def charger_fleurs(dossier: str = "/kaggle/working/fleurs", avec_audio: bool = True,
                   limite: Optional[int] = None) -> List[Dict[str, Any]]:
    """FLEURS ar_eg test apparié au français (fr_fr test) par identifiant de phrase.

    Colonnes du tsv : id, fichier, transcription brute, transcription normalisée, graphèmes, nb échantillons, genre.
    Une phrase peut être lue par plusieurs locuteurs (même id, plusieurs lignes) : on garde la 1re ligne de
    chaque id, côté arabe comme côté français, soit une phrase par id présent dans les deux langues.
    Retourne [{id, fichier, ar, fr, audio (float32 16 kHz) si avec_audio}].
    """
    d = Path(dossier)
    ar = lire_tsv(telecharger(f"{URL_FLEURS}/ar_eg/test.tsv", d / "ar_eg.tsv"))
    fr = lire_tsv(telecharger(f"{URL_FLEURS}/fr_fr/test.tsv", d / "fr_fr.tsv"))
    fr_par_id: Dict[str, str] = {}
    for r in fr:
        fr_par_id.setdefault(r[0], r[2])
    paires, vus = [], set()
    for r in ar:
        if r[0] in vus or r[0] not in fr_par_id:
            continue
        vus.add(r[0])
        paires.append({"id": r[0], "fichier": r[1], "ar": r[2], "fr": fr_par_id[r[0]]})
    if limite:
        paires = paires[:limite]
    if avec_audio:
        import soundfile as sf

        archive = telecharger(f"{URL_FLEURS}/ar_eg/audio/test.tar.gz", d / "test.tar.gz")
        voulus = {p["fichier"] for p in paires}
        dossier_audio = d / "audio"
        dossier_audio.mkdir(exist_ok=True)
        if not all((dossier_audio / f).exists() for f in voulus):
            with tarfile.open(archive, "r:gz") as tar:
                for m in tar:
                    nom = Path(m.name).name
                    if m.isfile() and nom in voulus:
                        with open(dossier_audio / nom, "wb") as out:
                            out.write(tar.extractfile(m).read())
        for p in paires:
            signal, sr = sf.read(str(dossier_audio / p["fichier"]), dtype="float32")
            if signal.ndim > 1:
                signal = signal.mean(axis=1)
            if sr != 16000:
                import librosa

                signal = librosa.resample(signal, orig_sr=sr, target_sr=16000)
            p["audio"] = signal
            p["duree_s"] = len(signal) / 16000
    return paires


# ----------------------------------------------------------------------------------- traduction
def charger_prompts(racine: Path) -> Dict[str, Any]:
    import yaml

    return yaml.safe_load((racine / "configs" / "s3.yaml").read_text(encoding="utf-8"))["traduction_texte"]


def messages_traduction(prompts: Dict[str, Any], texte: str, ctx: Optional[List[str]] = None,
                        systeme: Optional[str] = None, exemples: Optional[List[Dict[str, str]]] = None):
    """Messages ChatML identiques à ceux de S3 (traduction_texte de configs/s3.yaml)."""
    if ctx:
        utilisateur = prompts["gabarit_contexte"].format(contexte="\n".join(ctx), texte=texte.strip())
    else:
        utilisateur = prompts["gabarit"].format(texte=texte.strip())
    msgs = [{"role": "system", "content": systeme or prompts["systeme"]}]
    for ex in exemples or []:
        msgs.append({"role": "user", "content": prompts["gabarit"].format(texte=ex["ar"])})
        msgs.append({"role": "assistant", "content": ex["fr"]})
    msgs.append({"role": "user", "content": utilisateur})
    return msgs


def nettoyer_traduction(texte: str) -> str:
    texte = MOTIF_INTRODUCTION.sub("", texte.strip()).strip()
    if len(texte) >= 2 and texte[0] in "\"«“" and texte[-1] in "\"»”":
        texte = texte[1:-1].strip()
    return texte


def traduire(modele, tokenizer, prompts: Dict[str, Any], textes: List[str], contextes: Optional[List[Optional[List[str]]]] = None,
             taille_lot: int = 16, max_nouveaux_tokens: int = 300, **options) -> List[str]:
    """Génération gloutonne par lots (rembourrage à gauche), avec le gabarit de S3 et le même nettoyage."""
    import torch
    from modele_s3 import retirer_boucles

    contextes = contextes or [None] * len(textes)
    invites = [tokenizer.apply_chat_template(messages_traduction(prompts, t, c, options.get("systeme"), options.get("exemples")),
                                             tokenize=False, add_generation_prompt=True)
               for t, c in zip(textes, contextes)]
    ordre = sorted(range(len(textes)), key=lambda i: -len(invites[i]))
    sorties = [""] * len(textes)
    tokenizer.padding_side = "left"
    gen_kwargs = {k: v for k, v in options.items() if k in ("do_sample", "temperature", "top_k", "top_p", "num_return_sequences")}
    if not gen_kwargs.get("do_sample"):
        gen_kwargs = {"do_sample": False}
    for debut in range(0, len(ordre), taille_lot):
        idx = ordre[debut: debut + taille_lot]
        lot = tokenizer([invites[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(modele.device)
        limite = min(max_nouveaux_tokens, int(2.0 * max(len(tokenizer(textes[i]).input_ids) for i in idx)) + 20)
        with torch.no_grad():
            ids = modele.generate(**lot, max_new_tokens=limite, pad_token_id=tokenizer.pad_token_id,
                                  repetition_penalty=1.0, **gen_kwargs)
        for i, seq in zip(idx, ids):
            texte = tokenizer.decode(seq[lot["input_ids"].shape[1]:], skip_special_tokens=True)
            sorties[i] = nettoyer_traduction(retirer_boucles(texte.strip(), 4))
    return sorties


# ----------------------------------------------------------------------------------- métriques
def metriques_traduction(refs: List[str], hyps: List[str]) -> Dict[str, Any]:
    from sacrebleu.metrics import BLEU, CHRF

    bleu, chrf = BLEU(), CHRF(word_order=2)
    b, c = bleu.corpus_score(hyps, [refs]), chrf.corpus_score(hyps, [refs])
    return {"bleu": round(b.score, 2), "chrf_pp": round(c.score, 2),
            "signature_bleu": str(bleu.get_signature()), "signature_chrf_pp": str(chrf.get_signature())}


def chrf_phrase(ref: str, hyp: str) -> float:
    from sacrebleu.metrics import CHRF

    return round(CHRF(word_order=2).sentence_score(hyp, [ref]).score, 2)


def comet(srcs: List[str], hyps: List[str], refs: List[str], batch_size: int = 32) -> Dict[str, Any]:
    """COMET Unbabel/wmt22-comet-da ; renvoie {"comet": None, "motif": ...} si indisponible."""
    try:
        from comet import download_model, load_from_checkpoint

        modele = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        sortie = modele.predict([{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)],
                                batch_size=batch_size, gpus=1)
        return {"comet": round(float(sortie.system_score), 4), "scores_segments": [float(x) for x in sortie.scores]}
    except Exception as exc:  # noqa: BLE001
        return {"comet": None, "scores_segments": None, "motif": f"COMET indisponible : {type(exc).__name__}: {exc}"}


def ecrire_json(chemin, objet) -> None:
    Path(chemin).parent.mkdir(parents=True, exist_ok=True)
    Path(chemin).write_text(json.dumps(objet, ensure_ascii=False, indent=2), encoding="utf-8")


def jeton_hf() -> Optional[str]:
    """Jeton HF du secret Kaggle HF_TOKEN (jamais affiché) ; None s'il est absent."""
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN") or None
    except Exception:  # noqa: BLE001
        return None
