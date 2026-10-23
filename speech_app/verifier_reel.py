#!/usr/bin/env python3
"""Vérification de bout en bout de l'appli Kalam sur un VRAI GPU, avec de la vraie parole arabe (FLEURS ar_eg).

    python verifier_reel.py                       tout (20 à 30 min sur un T4, première fois : téléchargements)
    python verifier_reel.py --rapide              direct + rapport + lecture vocale (3 à 4 min une fois les modèles en cache)
    python verifier_reel.py --serveur http://localhost:7860 --rapide     contrôle le serveur DÉJÀ lancé (Colab)

Étapes (chacune donne PASS / FAIL / SKIP et des chiffres) :
  donnees   FLEURS ar_eg test -> un « JT » (une voix, 60-90 s) et un « dialogue » (deux voix, 45 s)
  direct    moteur réel + FluxIncremental, paquets de 250 ms à vitesse réelle : retard, réécriture, temps par passe
  serveur   uvicorn api:app, client WebSocket /ws/live, spectateur /ws/regarder, dépôt /api/live/fichier
  rapport   POST /api/sessions/{sid}/rapport : transcript, résumé daté, sous-titres, nombre de locuteurs, temps, VRAM
  tts       GET /api/tts
  depot     POST /api/translate (s3, seamless, s0) : srt/vtt, sid, temps, VRAM, un seul groupe en mémoire
  lora      comme « direct » avec l'adaptateur modeles/lora_traduction, s'il existe

Écrit verification_reelle.json (option --sortie). Le jeton HF_TOKEN n'est lu que dans l'environnement, jamais écrit.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import queue
import re
import statistics
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.request
import wave
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
RACINE = ICI.parent
SR = 16000
URL_FLEURS = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/ar_eg"
ARABE = re.compile("[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
PAS_S = 0.25                      # paquets de 250 ms, comme le navigateur
JOURNAL: list[str] = []
BILAN: dict = {"etapes": {}, "journal": JOURNAL}
SORTIE_JSON = Path("verification_reelle.json")
FACTICE = False                   # --factice : moteurs simulés, sans GPU, pour tester ce script lui-même


# ---------------------------------------------------------------- journal et bilan
def log(msg: str) -> None:
    ligne = f"[{time.strftime('%H:%M:%S')}] {msg}"
    JOURNAL.append(ligne)
    print(ligne, flush=True)


def secret(texte: str) -> str:
    """Retire le jeton HF de tout texte qui irait dans le journal ou le JSON."""
    jeton = os.environ.get("HF_TOKEN") or ""
    return texte.replace(jeton, "***") if len(jeton) > 8 else texte


def sauver() -> None:
    SORTIE_JSON.write_text(json.dumps(BILAN, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def executer(nom: str, fonction, *args, **kw) -> dict:
    """Lance une étape ; ses « verifs » (nom -> bool) décident PASS/FAIL ; une exception donne FAIL avec la cause."""
    log(f"===== étape {nom} =====")
    t0 = time.time()
    try:
        res = fonction(*args, **kw) or {}
    except Exception as exc:                                    # noqa: BLE001
        res = {"verifs": {"sans_exception": False}, "erreur": f"{type(exc).__name__}: {secret(str(exc))[:400]}",
               "trace": secret(traceback.format_exc())[-1500:]}
        log(f"  EXCEPTION {res['erreur']}")
    if res.get("statut") != "SKIP":
        verifs = res.get("verifs", {})
        res["statut"] = "PASS" if verifs and all(verifs.values()) else "FAIL"
    res["duree_etape_s"] = round(time.time() - t0, 1)
    BILAN["etapes"][nom] = res
    for k, v in res.get("verifs", {}).items():
        log(f"  [{'ok' if v else 'ÉCHEC'}] {k}")
    log(f"===== {nom} : {res['statut']} ({res['duree_etape_s']} s) =====")
    sauver()
    return res


# ---------------------------------------------------------------- petits outils
def pourcentile(valeurs: list[float], p: float) -> float | None:
    if not valeurs:
        return None
    v = sorted(valeurs)
    return round(v[min(len(v) - 1, int(np.ceil(p * len(v))) - 1)], 3)


def stats(valeurs: list[float]) -> dict:
    if not valeurs:
        return {"n": 0}
    return {"n": len(valeurs), "moyenne": round(statistics.fmean(valeurs), 3), "p95": pourcentile(valeurs, 0.95),
            "max": round(max(valeurs), 3)}


def normaliser_mots(texte: str) -> list[str]:
    t = re.sub("[ً-ٰٟـ]", "", texte)
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ة", "ه")):
        t = t.replace(a, b)
    return re.sub(r"[^\w\s]", " ", t).split()


def wer(ref: str, hyp: str) -> float:
    r, h = normaliser_mots(ref), normaliser_mots(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i, mr in enumerate(r, 1):
        prec, d[0] = d[0], i
        for j, mh in enumerate(h, 1):
            prec, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prec + (mr != mh))
    return d[len(h)] / len(r)


def srt_valide(texte: str) -> int:
    """Nombre de cues d'un .srt bien formé (numéro, 'hh:mm:ss,mmm --> hh:mm:ss,mmm', texte), 0 si le format est cassé."""
    blocs = [b for b in re.split(r"\n\s*\n", texte.replace("\r\n", "\n").strip()) if b.strip()]
    motif = re.compile(r"^\d+\n(\d\d:\d\d:\d\d,\d{3}) --> (\d\d:\d\d:\d\d,\d{3})\n.+", re.S)
    for b in blocs:
        if not motif.match(b.strip()):
            return 0
    return len(blocs)


def ecrire_wav(chemin: Path, signal: np.ndarray) -> None:
    pcm = (np.clip(signal, -1, 1) * 32767).astype("<i2")
    with wave.open(str(chemin), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm.tobytes())


def duree_wav(chemin: Path) -> float:
    with wave.open(str(chemin), "rb") as f:
        return f.getnframes() / f.getframerate()


def telecharger(url: str, dest: Path) -> Path:
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    log(f"  téléchargement {url.split('/')[-1]}")
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as out:
        while bloc := r.read(1 << 20):
            out.write(bloc)
    tmp.replace(dest)
    return dest


def nvidia_smi_mo() -> int | None:
    try:
        sortie = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10).stdout.split()
        return int(sortie[0])
    except Exception:                                           # noqa: BLE001
        return None


class EchantillonneurGpu:
    """Relève la mémoire GPU occupée (tous processus) chaque seconde ; les pics sont rangés par phase."""

    def __init__(self):
        self.phase = "debut"
        self.pics: dict[str, int] = {}
        self._stop = threading.Event()
        self._fil = threading.Thread(target=self._boucle, daemon=True)

    def demarrer(self):
        self._fil.start()
        return self

    def _boucle(self):
        while not self._stop.is_set():
            mo = nvidia_smi_mo()
            if mo is not None:
                self.pics[self.phase] = max(self.pics.get(self.phase, 0), mo)
            self._stop.wait(1.0)

    def marquer(self, phase: str):
        self.phase = phase
        self.pics.setdefault(phase, nvidia_smi_mo() or 0)

    def arreter(self):
        self._stop.set()


GPU = EchantillonneurGpu()


# ---------------------------------------------------------------- a. données : vraie parole arabe
def lire_tsv(chemin: Path) -> list[dict]:
    lignes = []
    for l in chemin.read_text(encoding="utf-8").splitlines():
        c = l.split("\t")
        if len(c) >= 7 and c[5].isdigit():
            lignes.append({"id": c[0], "fichier": c[1], "ar": c[2], "duree_s": int(c[5]) / SR, "genre": c[6].strip().upper()})
    return lignes


def extraire_audios(archive: Path, fichiers: set[str], dest: Path) -> dict[str, np.ndarray]:
    import soundfile as sf
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, np.ndarray] = {}
    deja = {f: dest / f for f in fichiers if (dest / f).is_file()}
    manquants = fichiers - set(deja)
    if manquants:
        with tarfile.open(archive, "r:gz") as tar:
            for m in tar:
                nom = Path(m.name).name
                if m.isfile() and nom in manquants:
                    (dest / nom).write_bytes(tar.extractfile(m).read())
                    manquants.discard(nom)
                    if not manquants:
                        break
    for f in fichiers:
        chemin = dest / f
        if not chemin.is_file():
            continue
        x, sr = sf.read(str(chemin), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=SR)
        out[f] = x
    return out


def empreintes_ecapa(audios: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Un embedding ECAPA par énoncé (sert à regrouper les énoncés d'une même voix) ; {} si speechbrain manque."""
    try:
        import torch
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier
        clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                             savedir=str(ICI / "cache" / "ecapa"),
                                             run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"})
        out = {}
        for f, x in audios.items():
            with torch.inference_mode():
                v = clf.encode_batch(torch.from_numpy(x).float().unsqueeze(0).to(clf.device))
            e = v.squeeze().detach().cpu().numpy()
            out[f] = e / max(np.linalg.norm(e), 1e-9)
        del clf
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out
    except Exception as exc:                                    # noqa: BLE001
        log(f"  ECAPA indisponible pour regrouper les voix ({type(exc).__name__}) : choix par genre seulement")
        return {}


def meilleur_groupe(fichiers: list[str], emb: dict[str, np.ndarray], seuil: float) -> list[str]:
    """Plus grand groupe d'énoncés dont la similarité cosinus avec un énoncé pivot dépasse le seuil (même voix)."""
    if not emb:
        return fichiers
    meilleur: list[str] = []
    for p in fichiers:
        g = [f for f in fichiers if float(emb[p] @ emb[f]) >= seuil]
        if len(g) > len(meilleur):
            meilleur = g
    return meilleur


def grouper_voix(fichiers: list[str], emb: dict[str, np.ndarray], seuil: float) -> list[list[str]]:
    """Groupes d'énoncés de même voix (similarité cosinus à un pivot >= seuil), du plus grand au plus petit."""
    if not emb:
        return [fichiers]
    restants, groupes = list(fichiers), []
    while restants:
        g = max(([f for f in restants if float(emb[p] @ emb[f]) >= seuil] for p in restants), key=len)
        groupes.append(g)
        restants = [f for f in restants if f not in g]
    return groupes


def assembler(audios: list[np.ndarray], silence_s: float = 0.6) -> np.ndarray:
    pause = np.zeros(int(silence_s * SR), dtype=np.float32)
    morceaux = [pause]
    for a in audios:
        morceaux += [a, pause]
    return np.concatenate(morceaux)


def preparer_donnees_factices(travail: Path) -> dict:
    import live_moteurs
    travail.mkdir(parents=True, exist_ok=True)
    jt = live_moteurs.signal_demo([0, 1, 2, 3, 0, 1, 2, 3], duree_phrase_s=7.0, pause_s=0.8)
    dialogue = live_moteurs.signal_demo([0, 1, 2, 3, 0, 1, 2], duree_phrase_s=6.0, pause_s=0.8)
    ecrire_wav(travail / "jt.wav", jt)
    ecrire_wav(travail / "dialogue.wav", dialogue)
    info = {"jt": {"fichier": str(travail / "jt.wav"), "duree_s": round(len(jt) / SR, 1), "n_phrases": 8, "ref_ar": None},
            "dialogue": {"fichier": str(travail / "dialogue.wav"), "duree_s": round(len(dialogue) / SR, 1), "n_locuteurs_attendus": 2}}
    return {"verifs": {"signaux_factices": True}, "donnees": {"factice": True}, "_info": info}


def preparer_donnees(travail: Path, rapide: bool) -> dict:
    if FACTICE:
        return preparer_donnees_factices(travail)
    tsv = lire_tsv(telecharger(f"{URL_FLEURS}/test.tsv", travail / "ar_eg.tsv"))
    vus, candidats = set(), []
    for l in tsv:                                    # tous genres : ar_eg test ne contient parfois que des voix de femmes
        if 4.0 <= l["duree_s"] <= 10.0 and l["id"] not in vus and len(candidats) < 150:
            vus.add(l["id"])
            candidats.append(l)
    log(f"  candidats : {len(candidats)} énoncés de 4 à 10 s ; genres : {sorted({l['genre'] for l in candidats})}")
    archive = telecharger(f"{URL_FLEURS}/audio/test.tar.gz", travail / "test.tar.gz")
    audios = extraire_audios(archive, {l["fichier"] for l in candidats}, travail / "audio")
    par_fichier = {l["fichier"]: l for l in candidats}
    emb = empreintes_ecapa(audios)
    fichiers = [l["fichier"] for l in candidats if l["fichier"] in audios]
    groupes = grouper_voix(fichiers, emb, 0.45)
    log(f"  voix (groupes d'énoncés de même locuteur d'après ECAPA) : tailles {[len(g) for g in groupes[:6]]}")
    voix_a = groupes[0]
    voix_b = next((g for g in groupes[1:] if len(g) >= 3 and float(emb[g[0]] @ emb[voix_a[0]]) < 0.35), None) if emb else None
    if voix_b is None:                               # pas de séparation fiable : deux moitiés, signalé dans le JSON
        moitie = max(1, len(fichiers) // 2)
        voix_a, voix_b = fichiers[:moitie], fichiers[moitie:]
    n_jt, cible_jt = (4, 30.0) if rapide else (8, 62.0)
    jt, total = [], 0.0
    for f in voix_a:
        jt.append(f)
        total += par_fichier[f]["duree_s"]
        if len(jt) >= n_jt and total >= cible_jt:
            break
    signal_jt = assembler([audios[f] for f in jt])
    # dialogue : deux voix qui alternent
    a_dial = [f for f in voix_a if f not in jt] or voix_a
    tours, total, i = [], 0.0, 0
    while total < 45.0 and i < 40:
        liste = a_dial if i % 2 == 0 else voix_b
        f = liste[(i // 2) % len(liste)]
        tours.append((f, "A" if i % 2 == 0 else "B"))
        total += par_fichier[f]["duree_s"] + 0.6
        i += 1
    signal_dialogue = assembler([audios[f] for f, _ in tours])
    ecrire_wav(travail / "jt.wav", signal_jt)
    ecrire_wav(travail / "dialogue.wav", signal_dialogue)
    info = {
        "jt": {"fichier": str(travail / "jt.wav"), "duree_s": round(len(signal_jt) / SR, 1), "n_phrases": len(jt),
               "ref_ar": " ".join(par_fichier[f]["ar"] for f in jt),
               "phrases": [{"fichier": f, "duree_s": round(par_fichier[f]["duree_s"], 1)} for f in jt]},
        "dialogue": {"fichier": str(travail / "dialogue.wav"), "duree_s": round(len(signal_dialogue) / SR, 1),
                     "tours": [{"fichier": f, "locuteur": s} for f, s in tours], "n_locuteurs_attendus": 2},
        "voix_verifiees_par_ecapa": bool(emb), "voix_b_separee_par_ecapa": bool(emb) and voix_b is not None,
        "tailles_groupes_de_voix": [len(g) for g in groupes[:8]],
    }
    verifs = {"jt_60_90s" if not rapide else "jt_court_ok": (60 <= info["jt"]["duree_s"] <= 90) if not rapide else info["jt"]["duree_s"] >= 20,
              "dialogue_45s_ou_plus": info["dialogue"]["duree_s"] >= 40,
              "deux_voix_distinctes": len({s for _, s in tours}) == 2 and info["voix_b_separee_par_ecapa"]}
    return {"verifs": verifs, "donnees": {"jt_duree_s": info["jt"]["duree_s"], "jt_phrases": len(jt),
                                          "dialogue_duree_s": info["dialogue"]["duree_s"], "dialogue_tours": len(tours),
                                          "voix_verifiees_par_ecapa": bool(emb)}, "_info": info}


# ---------------------------------------------------------------- b. direct : moteur réel + FluxIncremental
class MoteurChronometre:
    """Enrobe le moteur de direct et chronomètre chacune de ses trois méthodes."""

    def __init__(self, moteur):
        self.m = moteur
        self.temps: dict[str, list[float]] = {"transcrire": [], "provisoire": [], "definitif": []}
        self.nom = getattr(moteur, "nom", "?")
        self.lora_actif = getattr(moteur, "lora_actif", False)

    def _mesure(self, cle, f, *a):
        t0 = time.perf_counter()
        r = f(*a)
        self.temps[cle].append(time.perf_counter() - t0)
        return r

    def transcrire(self, signal):
        return self._mesure("transcrire", self.m.transcrire, signal)

    def traduire_provisoire(self, ar):
        return self._mesure("provisoire", self.m.traduire_provisoire, ar)

    def traduire_definitif(self, ar, contexte):
        return self._mesure("definitif", self.m.traduire_definitif, ar, contexte)


def rejouer(flux, signal: np.ndarray, vitesse: float = 1.0) -> tuple[list[dict], float]:
    """Envoie `signal` par paquets de 250 ms à vitesse réelle (un fil producteur) ; le consommateur fait comme
    live.SessionLive._boucle : il avale tout ce qui attend, puis traite. Renvoie les messages, chacun daté
    (`recu_a_s` : secondes réelles depuis le premier paquet), et la durée totale."""
    file: queue.Queue = queue.Queue()
    pas = int(PAS_S * SR)
    t0 = time.perf_counter()

    def produire():
        for i in range(0, len(signal), pas):
            attente = t0 + (i / SR) / vitesse - time.perf_counter()
            if attente > 0:
                time.sleep(attente)
            file.put(signal[i:i + pas])
        file.put(None)

    threading.Thread(target=produire, daemon=True).start()
    messages: list[dict] = []
    fini = False
    while not fini:
        lot = [file.get()]
        while lot[-1] is not None:
            try:
                lot.append(file.get_nowait())
            except queue.Empty:
                break
        for p in lot:
            if p is None:
                fini = True
            else:
                flux.ajouter(p)
        for m in flux.traiter():
            messages.append(dict(m, recu_a_s=round(time.perf_counter() - t0, 3)))
    for m in flux.terminer():
        messages.append(dict(m, recu_a_s=round(time.perf_counter() - t0, 3)))
    return messages, time.perf_counter() - t0


def analyser_messages(messages: list[dict], n_phrases: int, ref_ar: str | None) -> tuple[dict, dict]:
    """Contrôles sur le flux de messages (gris puis blanc, pas d'arabe, pas de doublon) ; renvoie (verifs, chiffres)."""
    provisoires = [m for m in messages if m["type"] == "provisoire"]
    definitifs = [m for m in messages if m["type"] == "definitif" and not m.get("vide")]
    ids_gris = set()
    avec_gris = 0
    for m in messages:
        if m["type"] == "provisoire":
            ids_gris.add(m["id_segment"])
        elif m["type"] == "definitif" and not m.get("vide") and m["id_segment"] in ids_gris:
            avec_gris += 1
    textes_fr = [m["fr"] for m in provisoires + definitifs]
    nb_arabe = sum(1 for t in textes_fr if ARABE.search(t))
    ids_def = [m["id_segment"] for m in definitifs]
    doublons_id = len(ids_def) - len(set(ids_def))
    norm = [re.sub(r"\W+", " ", m["fr"].lower()).strip() for m in definitifs]
    doublons_texte = len(norm) - len(set(norm))
    retards_msg = [m["recu_a_s"] - m["fin_s"] for m in definitifs]          # borne basse : fin_s inclut le silence final
    chiffres = {
        "n_provisoires": len(provisoires), "n_definitifs": len(definitifs), "n_phrases_attendues": n_phrases,
        "definitifs_precedes_d_un_gris": avec_gris, "messages_avec_arabe": nb_arabe,
        "doublons_id": doublons_id, "doublons_texte": doublons_texte,
        "retard_reel_vs_fin_segment_s": stats(retards_msg),
    }
    if ref_ar:
        chiffres["wer_s3_pct"] = round(100 * wer(ref_ar, " ".join(m.get("ar", "") for m in definitifs)), 1)
    verifs = {
        "des_phrases_definitives": len(definitifs) >= max(1, int(0.6 * n_phrases)),
        "gris_avant_blanc_(>=60%)": bool(definitifs) and avec_gris >= 0.6 * len(definitifs),
        "aucun_caractere_arabe_dans_fr": nb_arabe == 0,
        "pas_de_doublon": doublons_id == 0 and doublons_texte == 0,
    }
    return verifs, chiffres


def importer_flux():
    for d in (str(RACINE / "code" / "direct"), str(ICI)):
        if d not in sys.path:
            sys.path.insert(0, d)
    import flux_incremental
    return flux_incremental


def liberer_gpu_local() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def etape_direct(info: dict, sessions: Path, lora: str | None = None, nom_session: str = "direct") -> dict:
    import torch
    sys.path.insert(0, str(ICI))
    cuda = torch.cuda.is_available()
    os.environ["PORTEPAROLE_SESSIONS"] = str(sessions)
    import live_moteurs
    import sessions as modsessions
    flux_mod = importer_flux()
    GPU.marquer(f"{nom_session}_chargement")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    if FACTICE:
        moteur = live_moteurs.MoteurFactice(delai_s=0.05)
    else:
        moteur = live_moteurs.charger_moteur_direct(str(RACINE), None, lora if lora else "")
    t_charge = time.time() - t0
    vram_charge = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0
    log(f"  moteur chargé en {t_charge:.0f} s ({moteur.nom}, LoRA actif : {moteur.lora_actif}), pic VRAM {vram_charge:.1f} Go")
    chrono = MoteurChronometre(moteur)
    # échauffement : la première passe CUDA est lente et fausserait le retard
    chrono.m.transcrire(np.zeros(SR * 2, dtype=np.float32) + 0.01 * np.random.randn(SR * 2).astype(np.float32))
    chrono.temps = {"transcrire": [], "provisoire": [], "definitif": []}
    GPU.marquer(f"{nom_session}_inference")
    flux = flux_mod.FluxIncremental(chrono, flux_mod.reglages_direct_live())
    import soundfile as sf
    signal, _ = sf.read(info["jt"]["fichier"], dtype="float32")
    log(f"  rejeu de {len(signal) / SR:.0f} s d'audio à vitesse réelle")
    messages, duree = rejouer(flux, signal)
    bilan = flux.bilan()
    verifs, chiffres = analyser_messages(messages, info["jt"]["n_phrases"], info["jt"]["ref_ar"])
    # session enregistrée comme le fait live.SessionLive
    session = modsessions.Session.creer("verification", moteur.nom, f"verification {nom_session}")
    for m in messages:
        if m["type"] == "definitif" and not m.get("vide"):
            session.ajouter_segment({"id": m["id_segment"], "debut_s": m["debut_s"], "fin_s": m["fin_s"], "ar": m["ar"], "fr": m["fr"]})
    session.ecrire_audio(flux.signal())
    meta = session.meta()
    meta["bilan_direct"] = bilan
    session.ecrire_meta(meta)
    verifs["session_ecrite"] = session.chemin_audio.is_file() and len(session.segments()) == chiffres["n_definitifs"]
    retard_moyen = bilan.get("retard_moyen_s")
    pic = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0
    exemples = [{"ar": m["ar"][:80], "fr": m["fr"][:120], "retard_s": m.get("retard_s")} for m in messages
                if m["type"] == "definitif" and not m.get("vide")][:4]
    res = {
        "verifs": verifs,
        "chiffres": {**chiffres, "retard_moyen_s": retard_moyen, "retard_p95_s": bilan.get("retard_p95_s"),
                     "retard_max_s": bilan.get("retard_max_s"), "taux_reecriture": bilan.get("taux_reecriture"),
                     "bilan_flux": bilan, "duree_rejeu_s": round(duree, 1), "duree_audio_s": round(len(signal) / SR, 1),
                     "chargement_s": round(t_charge, 1), "vram_pic_go": round(pic, 2),
                     "temps_par_passe_s": {k: stats(v) for k, v in chrono.temps.items()}, "lora_actif": moteur.lora_actif},
        "exemples": exemples, "sid_session": session.sid,
    }
    del flux, chrono, moteur
    liberer_gpu_local()
    return res


# ---------------------------------------------------------------- c. serveur
def lancer_serveur(port: int, sessions: Path, journal: Path) -> subprocess.Popen:
    env = dict(os.environ, PORTEPAROLE_NLP_REPO=str(RACINE), PORTEPAROLE_ENGINES="s0,seamless,s3", PORTEPAROLE_MT="off",
               PORTEPAROLE_SESSIONS=str(sessions), PORTEPAROLE_API_TOKEN="", PYTHONIOENCODING="utf-8")
    env.pop("PORTEPAROLE_LORA", None)
    if FACTICE:
        env.update(PORTEPAROLE_LIVE="factice", PORTEPAROLE_LLM_RAPPORT="off", PORTEPAROLE_TTS="off")
    return subprocess.Popen([sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", str(port)],
                            cwd=str(ICI), env=env, stdout=open(journal, "w"), stderr=subprocess.STDOUT)


def attendre(url: str, delai: float = 240.0) -> bool:
    import requests
    fin = time.time() + delai
    while time.time() < fin:
        try:
            if requests.get(url, timeout=3).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(1.5)
    return False


def precharger(base: str, delai: float = 900.0) -> float:
    import requests
    t0 = time.time()
    requests.post(f"{base}/api/live/precharger", timeout=30).raise_for_status()
    if FACTICE:
        return 0.0
    while time.time() - t0 < delai:
        if requests.get(f"{base}/api/live/statut", timeout=10).json().get("groupe_charge") == "direct":
            return time.time() - t0
        time.sleep(2)
    raise TimeoutError("le moteur de direct ne s'est pas chargé")


class ClientWS:
    """Client WebSocket qui range tous les messages JSON reçus (fil dédié)."""

    def __init__(self, url: str):
        from websockets.sync.client import connect
        self.ws = connect(url, max_size=None, open_timeout=30)
        self.messages: list[dict] = []
        self.ferme = threading.Event()
        self.fil = threading.Thread(target=self._lire, daemon=True)
        self.fil.start()

    def _lire(self):
        try:
            for brut in self.ws:
                if isinstance(brut, str):
                    self.messages.append(dict(json.loads(brut), recu=time.time()))
        except Exception:                                       # noqa: BLE001 : fermeture
            pass
        self.ferme.set()

    def attendre_type(self, type_: str, delai: float, etat: str | None = None):
        fin = time.time() + delai
        while time.time() < fin:
            for m in self.messages:
                if m.get("type") == type_ and (etat is None or m.get("etat") == etat):
                    return m
            time.sleep(0.2)
        return None

    def envoyer(self, donnee):
        self.ws.send(donnee)

    def fermer(self):
        try:
            self.ws.close()
        except Exception:                                       # noqa: BLE001
            pass


def etape_serveur(base: str, info: dict, rapide: bool, etat: dict) -> dict:
    import requests
    import soundfile as sf
    verifs, chiffres = {}, {}
    GPU.marquer("serveur_chargement")
    t_charge = precharger(base)
    chiffres["precharge_s"] = round(t_charge, 1)
    log(f"  moteur de direct chargé par le serveur en {t_charge:.0f} s")
    GPU.marquer("serveur_direct")
    signal, _ = sf.read(info["jt"]["fichier"], dtype="float32")
    pcm = (np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes()
    ws_url = base.replace("http", "ws", 1)
    emetteur = ClientWS(f"{ws_url}/ws/live?source=micro&titre=verification-jt")
    pret = emetteur.attendre_type("etat", 120, "pret") or emetteur.attendre_type("etat", 1, "en_cours")
    verifs["serveur_pret"] = pret is not None
    if pret is None:
        return {"verifs": verifs, "erreur": "le serveur n'a jamais annoncé « pret »", "chiffres": chiffres}
    sid = pret["sid"]
    spectateur = ClientWS(f"{ws_url}/ws/regarder/{sid}")
    t0 = time.time()
    octets = int(PAS_S * SR) * 2
    for i in range(0, len(pcm), octets):
        attente = t0 + (i / 2 / SR) - time.time()
        if attente > 0:
            time.sleep(attente)
        emetteur.envoyer(pcm[i:i + octets])
    emetteur.envoyer(json.dumps({"type": "stop"}))
    fin = emetteur.attendre_type("fin", 180)
    time.sleep(1.0)
    spectateur.attendre_type("fin", 20)
    emetteur.fermer()
    spectateur.fermer()
    etat["sid_jt"] = sid
    msgs_e, msgs_s = emetteur.messages, spectateur.messages
    for m in msgs_e:
        m["recu_a_s"] = round(m["recu"] - t0, 3)
    verifs["message_fin_recu"] = fin is not None
    verifs["bilan_dans_fin"] = bool(fin and fin.get("bilan", {}).get("retard_moyen_s") is not None)
    v, c = analyser_messages([m for m in msgs_e if m.get("type") in ("provisoire", "definitif")], info["jt"]["n_phrases"], info["jt"]["ref_ar"])
    verifs.update({f"emetteur_{k}": ok for k, ok in v.items()})
    chiffres.update(c)
    chiffres["bilan_serveur"] = (fin or {}).get("bilan")
    ids_e = [m["id_segment"] for m in msgs_e if m.get("type") == "definitif" and not m.get("vide")]
    ids_s = [m["id_segment"] for m in msgs_s if m.get("type") == "definitif" and not m.get("vide")]
    verifs["spectateur_recoit_les_memes_definitifs"] = bool(ids_e) and ids_e == ids_s
    verifs["spectateur_recoit_la_fin"] = any(m.get("type") == "fin" for m in msgs_s)
    verifs["etat_pret_puis_en_cours"] = any(m.get("etat") == "en_cours" for m in msgs_e)
    # session sur disque
    r = requests.get(f"{base}/api/sessions/{sid}/fichiers/audio.wav", timeout=30)
    verifs["audio_session_ecrit"] = r.ok and len(r.content) > 40000
    seg = requests.get(f"{base}/api/sessions/{sid}/fichiers/segments.jsonl", timeout=30)
    verifs["segments_session_ecrits"] = seg.ok and len(seg.text.strip().splitlines()) == len(ids_e)
    chiffres["sid_jt"] = sid
    # dialogue déposé comme fichier (rejeu serveur par ffmpeg -re)
    if not rapide:
        GPU.marquer("serveur_fichier")
        with open(info["dialogue"]["fichier"], "rb") as f:
            dep = requests.post(f"{base}/api/live/fichier", files={"file": ("dialogue.wav", f, "audio/wav")},
                                data={"titre": "verification-dialogue"}, timeout=120)
        verifs["depot_fichier_accepte"] = dep.ok
        if dep.ok:
            sid_d = dep.json()["sid"]
            requests.post(f"{base}/api/live/{sid_d}/demarrer", timeout=30).raise_for_status()
            debut = time.time()
            etat_d = {}
            while time.time() - debut < info["dialogue"]["duree_s"] + 240:
                etat_d = requests.get(f"{base}/api/live/{sid_d}/etat", timeout=30).json()
                if etat_d.get("termine"):
                    break
                time.sleep(2)
            verifs["dialogue_termine_sans_erreur"] = bool(etat_d.get("termine")) and not etat_d.get("erreur")
            chiffres["dialogue"] = {"sid": sid_d, "segments": etat_d.get("segments"), "bilan": etat_d.get("bilan"),
                                    "erreur": etat_d.get("erreur"), "duree_s": round(time.time() - debut, 1)}
            etat["sid_dialogue"] = sid_d
            verifs["dialogue_a_des_segments"] = (etat_d.get("segments") or 0) >= 3
    return {"verifs": verifs, "chiffres": chiffres}


# ---------------------------------------------------------------- d. rapport
def etape_ecapa(info: dict, travail: Path) -> dict:
    """Mesure la séparation des voix par ECAPA sur le dialogue : similarités intra/inter locuteur, nombre de groupes par seuil."""
    sys.path.insert(0, str(ICI))
    import session_report as sr
    import soundfile as sf
    tours = info["dialogue"].get("tours")
    if not tours:
        return {"statut": "SKIP", "raison": "pas de tours connus (mode factice)"}
    audios = {}
    for f in {t["fichier"] for t in tours}:
        x, _ = sf.read(str(travail / "audio" / f), dtype="float32")
        audios[f] = x
    entiers = empreintes_ecapa(audios)
    # même chose sur des fragments de 3 s : les segments du direct sont courts
    fragments = {f"{f}#{k}": x[k * 3 * SR:(k + 1) * 3 * SR] for f, x in audios.items() for k in range(len(x) // (3 * SR))}
    courts = empreintes_ecapa(fragments)
    res = {}
    for nom, emb, etiq in (("enonces", entiers, lambda c: next(t["locuteur"] for t in tours if t["fichier"] == c)),
                           ("fragments_3s", courts, lambda c: next(t["locuteur"] for t in tours if t["fichier"] == c.split("#")[0]))):
        if not emb:
            continue
        cles = sorted(emb)
        lab = [etiq(c) for c in cles]
        m = np.stack([emb[c] for c in cles])
        sim = m @ m.T
        mem = np.array([[a == b for b in lab] for a in lab])
        iu = np.triu_indices(len(cles), 1)
        intra, inter = sim[iu][mem[iu]], sim[iu][~mem[iu]]
        par_seuil = {}
        for seuil in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45):
            g = sr.regrouper_embeddings(m, seuil)
            pur = sum(max(sum(1 for a, b in zip(g, lab) if a == k and b == l) for l in set(lab)) for k in set(g)) / len(g)
            par_seuil[str(seuil)] = {"n_groupes": len(set(g)), "purete": round(pur, 3)}
        res[nom] = {"n": len(cles), "sim_intra_moyenne": round(float(intra.mean()), 3), "sim_intra_p5": round(float(np.percentile(intra, 5)), 3),
                    "sim_inter_moyenne": round(float(inter.mean()), 3), "sim_inter_p95": round(float(np.percentile(inter, 95)), 3),
                    "par_seuil": par_seuil, "deux_groupes_forces_purete": round(
                        (lambda g: sum(max(sum(1 for a, b in zip(g, lab) if a == k and b == l) for l in set(lab)) for k in set(g)) / len(g))(
                            sr.regrouper_embeddings(m, 0.4, 2)), 3)}
        log(f"  ECAPA {nom} : intra {res[nom]['sim_intra_moyenne']} (p5 {res[nom]['sim_intra_p5']}), inter {res[nom]['sim_inter_moyenne']} "
            f"(p95 {res[nom]['sim_inter_p95']}) ; groupes par seuil {({k: v['n_groupes'] for k, v in par_seuil.items()})}")
    verifs = {"ecapa_mesure": bool(res)}
    if "enonces" in res:
        verifs["intra_superieur_a_inter"] = res["enonces"]["sim_intra_moyenne"] > res["enonces"]["sim_inter_moyenne"] + 0.1
    return {"verifs": verifs, "chiffres": res}


def etape_perf(base: str, info: dict, etat: dict, sessions: Path) -> dict:
    """Rapport sur une session de ~10 min (60 segments) fabriquée à partir du dialogue : objectif < 90 s."""
    import soundfile as sf
    os.environ["PORTEPAROLE_SESSIONS"] = str(sessions)
    sys.path.insert(0, str(ICI))
    import sessions as modsessions
    source = etat.get("sid_dialogue") or etat.get("sid_jt")
    if not source:
        return {"statut": "SKIP", "raison": "aucune session source"}
    base_segs = modsessions.Session.ouvrir(source).segments()
    signal, _ = sf.read(info["dialogue"]["fichier"], dtype="float32")
    n_rep = int(np.ceil(600 / (len(signal) / SR)))
    session = modsessions.Session.creer("verification", "perf", "perf 10 minutes")
    session.ecrire_audio(np.tile(signal, n_rep))
    toutes = [dict(s, id=f"{r}-{s['id']}", debut_s=s["debut_s"] + r * len(signal) / SR, fin_s=s["fin_s"] + r * len(signal) / SR)
              for r in range(n_rep) for s in base_segs]
    pas = max(1, len(toutes) // 60)
    for s in toutes[::pas][:60]:
        session.ajouter_segment(s)
    etat_perf = {"sid_dialogue": session.sid}
    res = etape_rapport(base, etat_perf, False)
    c = res.get("chiffres", {})
    c["duree_audio_s"] = round(n_rep * len(signal) / SR)
    c["n_segments_session"] = len(session.segments())
    res.setdefault("verifs", {})["rapport_sous_90_s"] = bool(c.get("temps_total_s")) and c["temps_total_s"] < 90
    res["verifs"].pop("deux_locuteurs_detectes", None)
    return res


def etape_rapport(base: str, etat: dict, rapide: bool) -> dict:
    import requests
    sid = etat.get("sid_dialogue") if not rapide else None
    sid = sid or etat.get("sid_jt")
    if not sid:
        return {"statut": "SKIP", "raison": "aucune session issue de l'étape serveur"}
    attendus = 2 if etat.get("sid_dialogue") and sid == etat["sid_dialogue"] else None
    GPU.marquer("rapport")
    r = requests.post(f"{base}/api/sessions/{sid}/rapport", json={}, timeout=60)
    verifs = {"rapport_accepte_202": r.status_code == 202}
    if r.status_code != 202:
        return {"verifs": verifs, "erreur": f"POST rapport : {r.status_code} {r.text[:200]}"}
    t0 = time.time()
    etapes: list[tuple[float, str]] = []
    resultat = {}
    while time.time() - t0 < 1500:
        resultat = requests.get(f"{base}/api/sessions/{sid}/rapport", timeout=30).json()
        e = resultat.get("etape") or ""
        if not etapes or etapes[-1][1] != e:
            etapes.append((round(time.time() - t0, 1), e))
        if resultat.get("etat") in ("pret", "erreur"):
            break
        time.sleep(1)
    total = time.time() - t0
    verifs["rapport_pret"] = resultat.get("etat") == "pret"
    if not verifs["rapport_pret"]:
        return {"verifs": verifs, "erreur": str(resultat.get("erreur"))[:500], "chiffres": {"temps_total_s": round(total, 1), "etapes": etapes}}
    rapport = resultat["rapport"]
    fichier = lambda nom: requests.get(f"{base}/api/sessions/{sid}/fichiers/{nom}", timeout=60)   # noqa: E731
    transcript, resume, srt = fichier("transcript.md"), fichier("resume.md"), fichier("sous_titres.srt")
    verifs["transcript_md"] = transcript.ok and "Locuteur" in transcript.text and len(transcript.text) > 200
    verifs["resume_md_non_vide"] = resume.ok and len(resume.text) > 200
    controle = rapport.get("controle_horodatages", {})
    verifs["resume_a_des_horodatages"] = controle.get("cites", 0) > 0
    verifs["horodatages_valides"] = controle.get("cites", 0) > 0 and not controle.get("invalides")
    verifs["srt_bien_forme"] = srt.ok and srt_valide(srt.text) > 0
    verifs["resume_par_LLM"] = rapport.get("methode_resume") == "LLM"
    verifs["revision_par_LLM"] = str(rapport.get("methode_revision", "")).startswith("LLM")
    verifs["diarisation_pyannote"] = str(rapport.get("methode_diarisation", "")).startswith("pyannote")
    if attendus:
        verifs["deux_locuteurs_detectes"] = rapport.get("n_locuteurs") == attendus
    modifies = sum(1 for s in rapport.get("segments", []) if s.get("modifie"))
    chiffres = {
        "sid": sid, "temps_total_s": round(total, 1), "temps_etapes_s": rapport.get("temps_etapes_s"), "duree_calcul_s": rapport.get("duree_calcul_s"),
        "etapes_chronologie_s": etapes, "n_locuteurs": rapport.get("n_locuteurs"), "n_locuteurs_attendus": attendus,
        "methode_diarisation": rapport.get("methode_diarisation"), "methode_revision": rapport.get("methode_revision"),
        "methode_resume": rapport.get("methode_resume"), "controle_horodatages": controle,
        "segments": len(rapport.get("segments", [])), "segments_modifies_par_relecture": modifies,
        "avertissements": rapport.get("avertissements"), "vram_pic_mo": GPU.pics.get("rapport"),
        "locuteurs_par_segment": [s.get("locuteur") for s in rapport.get("segments", [])],
        "resume_extrait": resume.text[:700] if resume.ok else None,
    }
    return {"verifs": verifs, "chiffres": chiffres}


# ---------------------------------------------------------------- e. lecture vocale
def etape_tts(base: str) -> dict:
    import requests
    phrase = "Bonsoir, voici les titres du journal : les pourparlers se sont poursuivis tard dans la nuit."
    t0 = time.time()
    r = requests.get(f"{base}/api/tts", params={"texte": phrase}, timeout=180)
    dt = time.time() - t0
    verifs = {"reponse_200": r.status_code == 200}
    chiffres = {"temps_s": round(dt, 2), "octets": len(r.content), "type": r.headers.get("content-type")}
    if r.ok:
        verifs["audio_non_vide"] = len(r.content) > 5000
        if "wav" in (r.headers.get("content-type") or ""):
            tmp = Path("tts_essai.wav")
            tmp.write_bytes(r.content)
            chiffres["duree_audio_s"] = round(duree_wav(tmp), 2)
            verifs["duree_plausible"] = 2.0 <= chiffres["duree_audio_s"] <= 20.0
            tmp.unlink()
        chiffres["moteur"] = "piper" if "wav" in (r.headers.get("content-type") or "") else "edge-tts (repli)"
    else:
        chiffres["erreur"] = r.text[:300]
    # Piper seul : le repli sur edge-tts cache sinon un Piper cassé
    sys.path.insert(0, str(ICI))
    try:
        import tts
        sortie = Path("tts_piper_essai.wav")
        t1 = time.time()
        tts._piper(phrase + " Deuxième essai.", sortie)
        chiffres["piper_direct_s"] = round(time.time() - t1, 2)
        verifs["piper_fonctionne_seul"] = sortie.is_file() and sortie.stat().st_size > 5000
        sortie.unlink(missing_ok=True)
    except Exception as exc:                                    # noqa: BLE001
        verifs["piper_fonctionne_seul"] = False
        chiffres["piper_erreur"] = secret(f"{type(exc).__name__}: {exc}")[:300]
    return {"verifs": verifs, "chiffres": chiffres}


# ---------------------------------------------------------------- f. dépôt de fichier, un moteur après l'autre
def etape_depot(base: str, info: dict, moteurs: tuple = ("s3", "seamless", "s0"), etat: dict | None = None) -> dict:
    import requests
    verifs, chiffres, groupes_vus = {}, {"moteurs": {}}, {}
    for moteur in moteurs:
        GPU.marquer(f"depot_{moteur}")
        t0 = time.time()
        with open(info["jt"]["fichier"], "rb") as f:
            r = requests.post(f"{base}/api/translate", files={"file": ("jt.wav", f, "audio/wav")},
                              data={"engine": moteur, "wait": "false"}, timeout=120)
        if not r.ok:
            verifs[f"{moteur}_accepte"] = False
            chiffres["moteurs"][moteur] = {"erreur": f"POST {r.status_code} {r.text[:200]}"}
            continue
        job_id = r.json()["job_id"]
        job = {}
        while time.time() - t0 < 1200:
            job = requests.get(f"{base}/api/jobs/{job_id}", timeout=30).json()
            if job.get("status") in ("done", "error"):
                break
            time.sleep(1.5)
        duree = time.time() - t0
        res = job.get("result") or {}
        ok = job.get("status") == "done"
        verifs[f"{moteur}_termine"] = ok
        c = {"temps_total_s": round(duree, 1), "temps_moteur_s": res.get("seconds"), "rtf": res.get("rtf"),
             "peak_mem_gb": res.get("peak_mem_gb"), "n_segments": len(res.get("segments", [])), "sid": res.get("sid"),
             "erreur": job.get("error"), "vram_pic_mo_nvidia_smi": None}
        if ok:
            verifs[f"{moteur}_srt_vtt"] = srt_valide(res.get("srt", "")) > 0 and res.get("vtt", "").startswith("WEBVTT")
            verifs[f"{moteur}_a_du_francais"] = bool(res.get("segments")) and not any(
                ARABE.search(s.get("fr") or "") for s in res.get("segments", []))
            if res.get("sid"):
                seg = requests.get(f"{base}/api/sessions/{res['sid']}/fichiers/segments.jsonl", timeout=30)
                verifs[f"{moteur}_session_creee"] = seg.ok and bool(seg.text.strip())
            else:
                verifs[f"{moteur}_session_creee"] = False
        groupe = requests.get(f"{base}/api/engines", timeout=30).json().get("loaded_group")
        c["groupe_charge_apres"] = groupe
        verifs[f"{moteur}_seul_groupe_charge"] = groupe == moteur
        c["vram_apres_mo"] = nvidia_smi_mo()
        c["vram_pic_mo_nvidia_smi"] = GPU.pics.get(f"depot_{moteur}")
        chiffres["moteurs"][moteur] = c
        log(f"  {moteur} : {c['temps_total_s']} s (moteur {c['temps_moteur_s']} s), groupe en mémoire : {groupe}, "
            f"VRAM pic {c['vram_pic_mo_nvidia_smi']} Mo")
    if etat is not None and not etat.get("sid_dialogue") and moteurs:
        # Pas de session de dialogue issue du direct : on dépose le dialogue (moteur le plus léger) pour tester le rapport.
        moteur = "s0" if "s0" in moteurs else moteurs[-1]
        with open(info["dialogue"]["fichier"], "rb") as f:
            r = requests.post(f"{base}/api/translate", files={"file": ("dialogue.wav", f, "audio/wav")},
                              data={"engine": moteur, "wait": "true"}, timeout=1200)
        sid_d = r.json().get("sid") if r.ok else None
        verifs["dialogue_depose_pour_le_rapport"] = bool(sid_d)
        if sid_d:
            etat["sid_dialogue"] = sid_d
            etat["moteur_dialogue"] = moteur
        chiffres["dialogue_depose"] = {"moteur": moteur, "sid": sid_d, "http": r.status_code}
    pics = [c.get("vram_pic_mo_nvidia_smi") or 0 for c in chiffres["moteurs"].values()]
    verifs["vram_sous_15_Go_a_chaque_etape"] = bool(pics) and max(pics) < 15000
    return {"verifs": verifs, "chiffres": chiffres}


# ---------------------------------------------------------------- orchestration
def environnement() -> dict:
    from importlib import metadata
    out: dict = {"python": sys.version.split()[0]}
    for p in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "pyannote.audio", "speechbrain", "piper-tts",
              "edge-tts", "fastapi", "uvicorn", "websockets", "gradio", "librosa", "huggingface_hub"):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    try:
        import torch
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        out["gpu_go"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None
    except Exception:                                           # noqa: BLE001
        pass
    out["hf_token_present"] = bool(os.environ.get("HF_TOKEN"))
    out["ffmpeg"] = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True).stdout.strip() or None
    return out


def main(argv=None) -> int:
    global SORTIE_JSON, FACTICE
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rapide", action="store_true", help="direct + rapport + lecture vocale, JT court (3 à 4 min)")
    p.add_argument("--serveur", help="URL d'un serveur déjà lancé (Colab) ; sinon uvicorn est lancé ici")
    p.add_argument("--etapes", default="", help="liste d'étapes séparées par des virgules (donnees,direct,serveur,rapport,tts,depot,lora)")
    p.add_argument("--moteurs", default="s3,seamless,s0", help="moteurs de l'étape depot (sans jeton HF : seamless,s0)")
    p.add_argument("--sortie", default="verification_reelle.json")
    p.add_argument("--travail", default=str(ICI / "cache" / "verification"), help="dossier des données FLEURS et des sessions")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--factice", action="store_true", help="essai du script lui-même avec des moteurs simulés (sans GPU)")
    a = p.parse_args(argv)
    FACTICE = a.factice
    SORTIE_JSON = Path(a.sortie)
    travail = Path(a.travail)
    travail.mkdir(parents=True, exist_ok=True)
    sessions = travail / "sessions"
    defaut = ["donnees", "serveur", "rapport", "tts"] if a.rapide else ["donnees", "direct", "serveur", "rapport", "tts", "depot", "lora"]
    if a.serveur and not a.etapes:
        defaut = [e for e in defaut if e != "direct"]
    etapes = [e for e in (a.etapes.split(",") if a.etapes else defaut) if e]
    BILAN["environnement"] = environnement()
    BILAN["mode"] = "rapide" if a.rapide else "complet"
    log(f"environnement : {json.dumps(BILAN['environnement'], ensure_ascii=False)}")
    GPU.demarrer()
    etat: dict = {}
    serveur = None
    base = (a.serveur or f"http://127.0.0.1:{a.port}").rstrip("/")

    def donnees():
        res = executer("donnees", preparer_donnees, travail, a.rapide)
        etat["info"] = res.pop("_info", None)
        sauver()
        return res

    try:
        if any(e in etapes for e in ("direct", "serveur", "rapport", "depot", "lora")) and "donnees" not in etapes:
            etapes.insert(0, "donnees")
        for e in etapes:
            if e == "donnees":
                donnees()
                if BILAN["etapes"]["donnees"]["statut"] != "PASS" and not etat.get("info"):
                    log("pas de données : les étapes suivantes sont sautées")
                    break
            elif e == "direct":
                executer("direct", etape_direct, etat["info"], sessions)
            elif e in ("serveur", "rapport", "perf", "tts", "depot") and not a.serveur and serveur is None:
                serveur = lancer_serveur(a.port, sessions, travail / "serveur.log")
                log(f"  uvicorn lancé (pid {serveur.pid}), port {a.port}")
                if not attendre(f"{base}/api/health"):
                    log("  le serveur ne répond pas ; fin du journal : " + secret((travail / "serveur.log").read_text(errors="replace")[-600:]))
                    BILAN["etapes"][e] = {"statut": "FAIL", "verifs": {"serveur_demarre": False}}
                    break
            if e == "serveur":
                executer("serveur", etape_serveur, base, etat["info"], a.rapide, etat)
            elif e == "rapport":
                executer("rapport", etape_rapport, base, etat, a.rapide)
            elif e == "ecapa":
                executer("ecapa", etape_ecapa, etat["info"], travail)
            elif e == "perf":
                executer("perf", etape_perf, base, etat["info"], etat, sessions)
            elif e == "tts":
                executer("tts", etape_tts, base)
            elif e == "depot":
                executer("depot", etape_depot, base, etat["info"], tuple(x for x in a.moteurs.split(",") if x), etat)
            elif e == "lora":
                lora = RACINE / "modeles" / "lora_traduction"
                if (lora / "adapter_config.json").is_file():
                    if serveur is not None:                       # libère la VRAM du serveur avant de charger le moteur ici
                        serveur.terminate()
                        serveur.wait(30)
                        serveur = None
                        time.sleep(3)
                    executer("lora", etape_direct, etat["info"], sessions, str(lora), "lora")
                else:
                    BILAN["etapes"]["lora"] = {"statut": "SKIP", "raison": "modeles/lora_traduction/adapter_config.json absent"}
                    log("===== lora : SKIP (adaptateur absent) =====")
        if a.serveur and any(e in etapes for e in ("rapport", "depot")):
            try:                                                  # le rapport et le dépôt ont déchargé le direct : on le recharge
                log(f"  rechargement du direct pour la suite de la démonstration ({precharger(base):.0f} s)")
            except Exception as exc:                              # noqa: BLE001
                log(f"  rechargement du direct impossible : {exc}")
    finally:
        GPU.arreter()
        BILAN["vram_pics_mo_par_phase"] = GPU.pics
        if serveur is not None:
            serveur.terminate()
            try:
                serveur.wait(30)
            except subprocess.TimeoutExpired:
                serveur.kill()
            try:
                BILAN["fin_journal_serveur"] = secret((travail / "serveur.log").read_text(errors="replace")[-2500:])
            except OSError:
                pass
        sauver()
    print("\n================ BILAN ================")
    for nom, r in BILAN["etapes"].items():
        echecs = [k for k, v in r.get("verifs", {}).items() if not v]
        print(f"{r['statut']:5} {nom:9} {r.get('duree_etape_s', '')} s" + (f"   échecs : {', '.join(echecs)}" if echecs else "")
              + (f"   {r['erreur']}" if r.get("erreur") else ""))
        if r.get("chiffres"):
            c = r["chiffres"]
            cles = ("retard_moyen_s", "retard_p95_s", "retard_max_s", "taux_reecriture", "vram_pic_go", "n_locuteurs",
                    "methode_diarisation", "temps_total_s", "wer_s3_pct", "chargement_s", "precharge_s", "moteur")
            print("        " + ", ".join(f"{k}={c[k]}" for k in cles if k in c and c[k] is not None))
    print(f"détail : {SORTIE_JSON.resolve()}")
    return 0 if all(r["statut"] in ("PASS", "SKIP") for r in BILAN["etapes"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
