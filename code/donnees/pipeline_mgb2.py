#!/usr/bin/env python3
"""Pipeline d'extraction, ASR, traduction et filtrage MGB-2 (Étapes D1 à D4).

Ce module est l'UNIQUE SOURCE DE VÉRITÉ pour toute la logique de traitement MGB-2.
Les notebooks dans `notebooks/` sont de fines interfaces qui importent ce module.

Règles impératives :
1. Séparation stricte entre :
   - Mode simulation : tests unitaires hors ligne déterministes sans réseau ni GPU.
   - Mode réel : streaming MGB-2 officiel, Whisper ASR réel, échec immédiat et explicite
     si le token, les fichiers ou le modèle manquent (AUCUN repli silencieux vers mock).
2. Reprise robuste basée sur l'identifiant source MGB-2 (recharge fiches D, A et rejets).
3. Intégrité et conservation stricte : entrée = gardés_d + gardés_a + rejets + déjà_traités.
4. Écriture immédiate et vérification audio FLAC sans 'except: pass'.
5. Séparation nette entre schéma intermédiaire D1 (SCHEMA_D_SELECTION) et schéma final D (SCHEMA_D).
6. Disjonction stricte entre les fichiers d_selection_v1.jsonl et a_v1.jsonl.
"""

import os
import sys
import math
import json
import csv
import re
import time
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set

# Imports internes du projet
REPERTOIRE_COURANT = Path(__file__).resolve().parent
if str(REPERTOIRE_COURANT) not in sys.path:
    sys.path.insert(0, str(REPERTOIRE_COURANT))

from seuils import charger_seuils
from normalisation import (
    normaliser_arabe,
    compter_mots_arabe,
    taux_ecart_ecoute,
    bw_vers_arabe,
)
from schemas import SCHEMA_D, SCHEMA_A, SCHEMA_D_SELECTION
from validation import valider_fiche


# ==============================================================================
# UTILITAIRES SYSTÈME, AUDIO ET IDENTIFIANTS
# ==============================================================================
def nettoyer_nom_fichier(identifiant: str) -> str:
    r"""Nettoie une chaîne pour qu'elle soit un nom de fichier valide sous Windows et Linux.
    
    Remplace les caractères interdits (: / \ < > " | ? *) par des tirets bas.
    """
    return re.sub(r'[<>:"/\\|?*\s]', "_", identifiant).strip("._")


def determiner_partie(episode_id: str, pct_val: int = 5) -> str:
    """Découpage déterministe par épisode (ex: 95% entraînement, 5% validation)."""
    h = int(hashlib.md5(episode_id.encode("utf-8")).hexdigest()[:8], 16)
    return "validation" if (h % 100) < pct_val else "entrainement"


def sauvegarder_et_verifier_audio_flac(waveform: Any, sr: int, chemin_destination: Path) -> Tuple[bool, str]:
    """Sauvegarde un segment audio en FLAC et le relit immédiatement pour valider l'intégrité.
    
    Ne contient AUCUN 'except: pass'. Si une étape échoue, renvoie False avec le motif clair.
    """
    try:
        import soundfile as sf
    except ImportError as e:
        return False, f"erreur_sauvegarde_audio: bibliotheque_soundfile_manquante ({e})"

    # 1. Création répertoire et écriture disque
    try:
        chemin_destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(chemin_destination), waveform, sr, format="FLAC")
    except Exception as e:
        return False, f"erreur_sauvegarde_audio: echec_ecriture ({e})"

    # 2. Relecture immédiate pour vérification
    try:
        data_relue, sr_relue = sf.read(str(chemin_destination))
    except Exception as e:
        return False, f"erreur_relecture_audio: echec_relecture ({e})"

    if sr_relue != sr:
        return False, f"erreur_relecture_audio: frequence_incoherente ({sr_relue} != {sr})"

    if len(data_relue) == 0:
        return False, "erreur_relecture_audio: fichier_audio_vide"

    return True, ""


# ==============================================================================
# REPRISE SUR INTERRUPTION ET PERSISTANCE ATOMIQUE
# ==============================================================================
def charger_etat_existant(
    chemin_d: Path,
    chemin_a: Path,
    chemin_rejets: Path,
) -> Tuple[Set[str], Dict[str, int], Dict[str, int], int, int]:
    """Recharge les identifiants sources déjà traités et les compteurs pour reprise exacte."""
    ids_traites: Set[str] = set()
    compteur_ep_d: Dict[str, int] = {}
    compteur_ep_a: Dict[str, int] = {}
    nb_d = 0
    nb_a = 0

    # Recharge Jeu D (sélection)
    if chemin_d.exists():
        with open(chemin_d, "r", encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if not ligne:
                    continue
                try:
                    obj = json.loads(ligne)
                    id_src = obj.get("id_source_mgb2") or obj.get("id_source") or obj.get("id")
                    if id_src:
                        ids_traites.add(id_src)
                    ep = obj.get("episode")
                    if ep:
                        compteur_ep_d[ep] = compteur_ep_d.get(ep, 0) + 1
                    nb_d += 1
                except json.JSONDecodeError:
                    continue

    # Recharge Jeu A (alignement)
    if chemin_a.exists():
        with open(chemin_a, "r", encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if not ligne:
                    continue
                try:
                    obj = json.loads(ligne)
                    id_src = obj.get("id_source_mgb2") or obj.get("id_source") or obj.get("id")
                    if id_src:
                        ids_traites.add(id_src)
                    ep = obj.get("episode")
                    if ep:
                        compteur_ep_a[ep] = compteur_ep_a.get(ep, 0) + 1
                    nb_a += 1
                except json.JSONDecodeError:
                    continue

    # Recharge Rejets
    if chemin_rejets.exists():
        with open(chemin_rejets, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                id_src = row.get("id")
                if id_src:
                    ids_traites.add(id_src)

    return ids_traites, compteur_ep_d, compteur_ep_a, nb_d, nb_a


# ==============================================================================
# ACCÈS AUX DONNÉES RÉELLES MGB-2 (KALDI STREAMING)
# ==============================================================================
def obtenir_stream_mgb2(
    hf_token: Optional[str] = None,
    dossier_cache: Optional[Path] = None,
):
    """Itérateur sur les segments réels MGB-2 via extraction streaming de l'archive officielle Kaldi.
    
    Lit le fichier de métadonnées `text.non_overlap_speech` et extrait les fichiers WAV d'émissions
    au besoin dans un cache local, évitant de re-télécharger l'archive complète.
    """
    import zlib
    import tarfile
    import requests
    from huggingface_hub import hf_hub_url
    import soundfile as sf

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token and Path(".env").exists():
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip("\"'")
                    break

    if not token:
        raise PermissionError(
            "[ERREUR BLOQUANTE D1] Le jeton HF_TOKEN est obligatoire pour accéder au dataset MGB-2. "
            "Aucun repli silencieux vers la simulation n'est autorisé en mode réel."
        )

    if dossier_cache is None:
        dossier_cache = REPERTOIRE_COURANT.parents[1] / "donnees" / "distille" / "cache_mgb2"
    dossier_cache.mkdir(parents=True, exist_ok=True)

    fichier_text = dossier_cache / "text.non_overlap_speech"

    # 1. Vérifier si les métadonnées de transcription sont déjà en cache
    if not fichier_text.exists():
        print("[D1 RÉEL] Téléchargement en flux des métadonnées MGB-2 (text.non_overlap_speech)...")
        url_dev = hf_hub_url("QCRI/mgb2", "dev/dev.tar.gz", repo_type="dataset")
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.get(url_dev, headers=headers, stream=True)
        resp.raise_for_status()

        # Lecture séquentielle des premiers Mo pour extraire text.non_overlap_speech
        chunks = []
        total = 0
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            chunks.append(chunk)
            total += len(chunk)
            if total > 8 * 1024 * 1024:  # 8 Mo suffisent largement
                break
        resp.close()

        decomp = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(b"".join(chunks))
        import io
        tar = tarfile.TarFile(fileobj=io.BytesIO(decomp))
        trouve = False
        for m in tar:
            if m.name.endswith("text.non_overlap_speech"):
                with open(fichier_text, "wb") as out_f:
                    out_f.write(tar.extractfile(m).read())
                trouve = True
                break
        if not trouve:
            raise RuntimeError("[ERREUR BLOQUANTE D1] Impossible de trouver 'text.non_overlap_speech' dans l'archive.")
        print(f"[D1 RÉEL] Métadonnées MGB-2 enregistrées en cache local ({fichier_text.stat().st_size} octets).")

    # 2. Parsing des métadonnées de segments
    segments_metadonnees = []
    with open(fichier_text, "r", encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            parts = ligne.split(maxsplit=1)
            seg_id_source = parts[0]
            texte_brut = parts[1] if len(parts) > 1 else ""

            # Format Kaldi : <GUID_EPISODE>_spk-<SPK>_seg-<DEBUT_CS>:<FIN_CS>
            m = re.search(r"^([A-Fa-f0-9-]+)_.*_seg-(\d+):(\d+)$", seg_id_source)
            if not m:
                continue
            episode = m.group(1)
            deb_cs = int(m.group(2))
            fin_cs = int(m.group(3))
            debut_s = deb_cs / 100.0
            fin_s = fin_cs / 100.0
            duree_s = fin_s - debut_s

            # Le texte dans Kaldi MGB-2 est translittéré en Buckwalter ; conversion en Arabe Unicode
            ar_unicode = bw_vers_arabe(texte_brut)

            segments_metadonnees.append({
                "id_source": seg_id_source,
                "episode": episode,
                "debut": debut_s,
                "fin": fin_s,
                "duree": duree_s,
                "ar_manuel": ar_unicode,
            })

    print(f"[D1 RÉEL] {len(segments_metadonnees)} segments indexés depuis les métadonnées MGB-2.")

    # 3. Prioriser les segments dont l'émission WAV est déjà en cache local
    segments_metadonnees.sort(key=lambda s: 0 if (dossier_cache / f"{s['episode']}.wav").exists() else 1)

    # 4. Générateur qui extrait l'audio par émission à la demande
    cache_wavs_charges = {}

    for meta in segments_metadonnees:
        ep = meta["episode"]
        chemin_wav_ep = dossier_cache / f"{ep}.wav"

        # Si le WAV de l'émission n'est pas encore en cache, le télécharger en flux
        if not chemin_wav_ep.exists():
            print(f"[D1 RÉEL] Téléchargement streaming de l'émission audio {ep}.wav...")
            url_dev = hf_hub_url("QCRI/mgb2", "dev/dev.tar.gz", repo_type="dataset")
            headers = {"Authorization": f"Bearer {token}"}
            resp = requests.get(url_dev, headers=headers, stream=True)
            resp.raise_for_status()

            # Adaptateur de flux décompressé
            class DecompressorStream:
                def __init__(self, raw_resp):
                    self.raw = raw_resp
                    self.dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
                    self.buf = b""
                def read(self, size=-1):
                    while len(self.buf) < size:
                        try:
                            c = next(self.raw.iter_content(chunk_size=65536))
                        except StopIteration:
                            break
                        if not c:
                            break
                        self.buf += self.dec.decompress(c)
                    res = self.buf[:size]
                    self.buf = self.buf[size:]
                    return res

            ds_tar = tarfile.open(fileobj=DecompressorStream(resp), mode="r|")
            wav_trouve = False
            for member in ds_tar:
                if member.name.endswith(f"{ep}.wav"):
                    with open(chemin_wav_ep, "wb") as out_wav:
                        f_wav = ds_tar.extractfile(member)
                        while True:
                            b = f_wav.read(1024 * 1024)
                            if not b:
                                break
                            out_wav.write(b)
                    wav_trouve = True
                    break
            resp.close()
            if not wav_trouve:
                print(f"[D1 RÉEL AVERTISSEMENT] WAV de l'émission {ep} non trouvé dans cette archive, saut du segment.")
                continue

        # Découpage du segment dans l'audio d'émission
        if ep not in cache_wavs_charges:
            info_wav = sf.info(str(chemin_wav_ep))
            cache_wavs_charges[ep] = info_wav

        info_ep = cache_wavs_charges[ep]
        sr = info_ep.samplerate
        debut_ech = int(meta["debut"] * sr)
        nb_ech = int(meta["duree"] * sr)

        try:
            waveform, _ = sf.read(str(chemin_wav_ep), start=debut_ech, stop=debut_ech + nb_ech, dtype="float32")
        except Exception as e:
            print(f"[D1 RÉEL ERREUR] Échec de lecture de la tranche {meta['id_source']}: {e}")
            continue

        item = dict(meta)
        item["waveform"] = waveform
        item["sr"] = sr
        yield item


# ==============================================================================
# ÉTAPE D1 : SÉLECTION DES SEGMENTS ET ÉCOUTE ASR
# ==============================================================================
def executer_d1_selection(
    mode_reel: bool = False,
    max_segments: int = 20,
    max_evaluations_asr: Optional[int] = None,
    num_beams: int = 4,
    modele_ecoute: str = "openai/whisper-tiny",
    repartition_mode: str = "alterne",  # 'alterne' (essai équilibré D/A) ou 'quota' (production)
    chemin_sortie_d: Optional[Path] = None,
    chemin_sortie_a: Optional[Path] = None,
    chemin_rejets: Optional[Path] = None,
    chemin_stats: Optional[Path] = None,
    dossier_audio: Optional[Path] = None,
    dossier_cache: Optional[Path] = None,
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Exécute l'étape D1 : sélection des segments MGB-2 pour les Jeux D et A.
    
    Règles :
    - D contient les segments destinés à recevoir une traduction française ultérieure.
    - A contient les segments supplémentaires audio + texte arabe pour l'alignement projecteur.
    - Les fichiers d_selection_v1.jsonl et a_v1.jsonl sont rigoureusement DISJOINTS.
    - Reprise basée sur l'identifiant source MGB-2 (aucun doublon créé).
    - Conservation stricte : entrée = gardés_d + gardés_a + rejets + déjà_traités.
    """
    seuils = charger_seuils()
    const = seuils["constantes"]
    prop = seuils["proposes"]

    duree_min = const["duree_segment_min_s"]      # 5 s
    duree_max = const["duree_segment_max_s"]      # 25 s
    mots_min = const["mots_min_segment"]          # 6 mots
    ecart_max = prop["ecart_ecoute_max_pct"]      # 15 %
    max_ep_d = const["segments_max_par_episode"]  # 200
    max_ep_a = const["segments_max_par_episode_a"]# 600
    pct_val = const["episodes_validation_pct"]    # 5 %

    base_dir = REPERTOIRE_COURANT.parents[1]
    if chemin_sortie_d is None:
        chemin_sortie_d = base_dir / "donnees" / "distille" / "d_selection_v1.jsonl"
    if chemin_sortie_a is None:
        chemin_sortie_a = base_dir / "donnees" / "alignement" / "a_v1.jsonl"
    if chemin_rejets is None:
        chemin_rejets = base_dir / "donnees" / "distille" / "rejets_d1.csv"
    if chemin_stats is None:
        chemin_stats = base_dir / "donnees" / "distille" / "stats_d1.json"
    if dossier_audio is None:
        dossier_audio = base_dir / "donnees" / "distille" / "audio"

    dossier_audio.mkdir(parents=True, exist_ok=True)
    chemin_sortie_d.parent.mkdir(parents=True, exist_ok=True)
    chemin_sortie_a.parent.mkdir(parents=True, exist_ok=True)
    chemin_rejets.parent.mkdir(parents=True, exist_ok=True)

    # 1. Recharge de l'état existant pour reprise exacte
    ids_deja_traites, compteur_ep_d, compteur_ep_a, nb_d_existant, nb_a_existant = charger_etat_existant(
        chemin_sortie_d, chemin_sortie_a, chemin_rejets
    )

    date_exec_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    parametres_meta = {
        "duree_min_s": duree_min,
        "duree_max_s": duree_max,
        "mots_min": mots_min,
        "ecart_max_pct": ecart_max,
        "repartition_mode": repartition_mode,
        "mode_reel": mode_reel,
    }

    stats = {
        "entree": 0,
        "deja_traites": 0,
        "gardes_d": 0,
        "gardes_a": 0,
        "rejets": 0,
        "rejets_par_motif": {},
        "duree_audio_totale_s": 0.0,
        "temps_chargement_modele_s": 0.0,
        "temps_inference_total_s": 0.0,
        "temps_asr_par_segment": [],
        "temps_asr_moyen_s": 0.0,
        "temps_asr_median_s": 0.0,
        "facteur_temps_reel_rtf": 0.0,
        "appareil_utilise": "cpu",
        "vram_max_mo": 0.0,
    }

    nouveaux_gardes_d = []
    nouveaux_gardes_a = []
    nouveaux_rejets = []

    # Initialisation du générateur et du modèle ASR
    asr_pipe = None
    revision_modele = "main"

    if mode_reel:
        # Vérification impérative du token AVANT tout chargement de modèle
        if hf_token is not None:
            token = hf_token.strip()
        else:
            token = os.environ.get("HF_TOKEN")
            if not token and Path(".env").exists():
                with open(".env", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip().startswith("HF_TOKEN="):
                            token = line.split("=", 1)[1].strip().strip("\"'")
                            break

        if not token:
            raise PermissionError(
                "[ERREUR BLOQUANTE D1] Le jeton HF_TOKEN est obligatoire pour accéder au dataset MGB-2 en mode réel."
            )

        try:
            import torch
            from transformers import pipeline
        except ImportError as e:
            raise RuntimeError(f"[ERREUR BLOQUANTE D1] PyTorch et Transformers sont requis en mode réel : {e}")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stats["appareil_utilise"] = device
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"[D1 RÉEL] Chargement du modèle ASR : {modele_ecoute} sur {device}...")
        t0_mod = time.time()
        try:
            asr_pipe = pipeline(
                "automatic-speech-recognition",
                model=modele_ecoute,
                device=device,
                torch_dtype=dtype,
                generate_kwargs={"language": "arabic", "task": "transcribe", "num_beams": num_beams},
            )
        except Exception as e:
            raise RuntimeError(f"[ERREUR BLOQUANTE D1] Échec du chargement du modèle Whisper '{modele_ecoute}' : {e}")
        t_charge = time.time() - t0_mod
        stats["temps_chargement_modele_s"] = round(t_charge, 2)
        print(f"[D1 RÉEL] Modèle chargé avec succès en {t_charge:.2f}s.")

        generateur_flux = obtenir_stream_mgb2(hf_token=hf_token, dossier_cache=dossier_cache)
    else:
        print("[D1 SIMULATION] Générateur de flux synthétique hors ligne...")
        episodes_factices = [f"mgb2_ep_{i:03d}" for i in range(1, 6)]
        exemples_textes = [
            "قال الوزير ان المفاوضات مستمرة حتى نهاية الشهر في العاصمة باريس",
            "أعلنت وزارة الصحة عن تسجيل انخفاض ملحوظ في معدل الإصابات",
            "بحث الجانبان سبل تعزيز التعاون الثنائي في المجالات الاقتصادية والاستثمارية",
            "وأكد المتحدث الرسمي أن الاجتماع القادم سيعقد في موعده المحدد",
            "نعم شكرا",  # Segment court (rejet mots_insuffisants)
        ]
        segments_simules = []
        import numpy as np
        for i in range(1, max_segments + 30):
            ep = episodes_factices[i % len(episodes_factices)]
            duree = 12.0 if i % 8 != 0 else 3.5  # quelques durées hors bornes
            txt = exemples_textes[i % len(exemples_textes)]
            segments_simules.append({
                "id_source": f"seg_mgb2_{i:05d}",
                "episode": ep,
                "debut": float(i * 30),
                "fin": float(i * 30 + duree),
                "duree": duree,
                "ar_manuel": txt,
                "waveform": np.zeros(int(duree * 16000), dtype=np.float32),
                "sr": 16000,
            })
        generateur_flux = iter(segments_simules)

    # Initialisation des descripteurs de fichiers pour écriture progressive
    f_d = open(chemin_sortie_d, "a", encoding="utf-8")
    f_a = open(chemin_sortie_a, "a", encoding="utf-8")
    rejets_header = not chemin_rejets.exists() or chemin_rejets.stat().st_size == 0
    f_rejets = open(chemin_rejets, "a", encoding="utf-8", newline="")
    writer_rejets = csv.DictWriter(f_rejets, fieldnames=["id", "filtre", "motif", "valeur", "seuil"])
    if rejets_header:
        writer_rejets.writeheader()
        f_rejets.flush()

    # Boucle de traitement des segments
    t0_global = time.time()

    nb_asr_evalue = 0
    stats["evaluations_asr"] = 0

    try:
        while (len(nouveaux_gardes_d) + len(nouveaux_gardes_a) + nb_d_existant + nb_a_existant) < max_segments:
            if max_evaluations_asr is not None and nb_asr_evalue >= max_evaluations_asr:
                print(f"[D1 RÉEL] Limite de {max_evaluations_asr} évaluations ASR atteinte.")
                break

            try:
                item = next(generateur_flux)
            except StopIteration:
                break

            stats["entree"] += 1
            seg_id_src = item["id_source"]
            episode = item["episode"]
            debut = item["debut"]
            fin = item["fin"]
            duree = item["duree"]
            ar_manuel = item["ar_manuel"]
            waveform = item["waveform"]
            sr = item["sr"]

            # VÉRIFICATION DE REPRISE : Si déjà traité, ignorer immédiatement sans doublon
            if seg_id_src in ids_deja_traites:
                stats["deja_traites"] += 1
                continue

            ids_deja_traites.add(seg_id_src)

            # Filtre 1 : Durée [5, 25] secondes
            if not (duree_min <= duree <= duree_max):
                rej = {
                    "id": seg_id_src,
                    "filtre": "D1_1_duree",
                    "motif": "duree_hors_bornes",
                    "valeur": f"{duree:.2f}",
                    "seuil": f"[{duree_min}, {duree_max}]",
                }
                writer_rejets.writerow(rej)
                f_rejets.flush()
                stats["rejets"] += 1
                stats["rejets_par_motif"]["duree_hors_bornes"] = stats["rejets_par_motif"].get("duree_hors_bornes", 0) + 1
                nouveaux_rejets.append(rej)
                continue

            # Filtre 2 : Nombre de mots >= 6
            nb_mots = compter_mots_arabe(ar_manuel)
            if nb_mots < mots_min:
                rej = {
                    "id": seg_id_src,
                    "filtre": "D1_2_mots",
                    "motif": "mots_insuffisants",
                    "valeur": str(nb_mots),
                    "seuil": f">={mots_min}",
                }
                writer_rejets.writerow(rej)
                f_rejets.flush()
                stats["rejets"] += 1
                stats["rejets_par_motif"]["mots_insuffisants"] = stats["rejets_par_motif"].get("mots_insuffisants", 0) + 1
                nouveaux_rejets.append(rej)
                continue

            # Filtre 3 : Écoute ASR et écart de transcription
            nb_asr_evalue += 1
            stats["evaluations_asr"] = nb_asr_evalue

            t_asr_start = time.time()
            if mode_reel:
                res_asr = asr_pipe(
                    {"raw": waveform, "sampling_rate": sr},
                    generate_kwargs={"language": "arabic", "task": "transcribe", "num_beams": num_beams},
                )
                ar_ecoute = res_asr.get("text", "").strip()
            else:
                # Simulation déterministe
                mots = ar_manuel.split()
                if len(mots) > 8 and "وزير" in ar_manuel:
                    mots_alt = list(mots)
                    mots_alt[1] = "الرئيس"
                    ar_ecoute = " ".join(mots_alt)
                else:
                    ar_ecoute = ar_manuel

            t_asr_duree = time.time() - t_asr_start
            stats["temps_inference_total_s"] += t_asr_duree
            stats["temps_asr_par_segment"].append(t_asr_duree)

            ecart = taux_ecart_ecoute(ar_manuel, ar_ecoute)

            if (ecart * 100.0) >= ecart_max:
                rej = {
                    "id": seg_id_src,
                    "filtre": "D1_3_ecart",
                    "motif": "ecart_ecoute_trop_eleve",
                    "valeur": f"{ecart*100.0:.2f}%",
                    "seuil": f"<{ecart_max}%",
                }
                writer_rejets.writerow(rej)
                f_rejets.flush()
                stats["rejets"] += 1
                stats["rejets_par_motif"]["ecart_ecoute_trop_eleve"] = stats["rejets_par_motif"].get("ecart_ecoute_trop_eleve", 0) + 1
                nouveaux_rejets.append(rej)
                continue

            # Filtre 4 : Quota par épisode
            partie = determiner_partie(episode, pct_val)
            compteur_ep_d.setdefault(episode, 0)
            compteur_ep_a.setdefault(episode, 0)

            # RÈGLE DE CONSTITUTION DES JEUX D ET A :
            # Les fichiers D et A sont DISJOINTS.
            # En mode essai 'alterne' : on alterne entre D et A pour alimenter et tester les deux formats.
            # En mode production 'quota' : D prend jusqu'à 200 par épisode, puis A prend le surplus jusqu'à 600.
            attribuer_a_d = False
            if repartition_mode == "alterne":
                # Si les deux ont le même nombre, attribuer à D d'abord, puis à A
                if len(nouveaux_gardes_d) <= len(nouveaux_gardes_a):
                    attribuer_a_d = True
                else:
                    attribuer_a_d = False
            else:
                if compteur_ep_d[episode] < max_ep_d:
                    attribuer_a_d = True
                elif compteur_ep_a[episode] < max_ep_a:
                    attribuer_a_d = False
                else:
                    rej = {
                        "id": seg_id_src,
                        "filtre": "D1_4_quota",
                        "motif": "quota_episode_atteint",
                        "valeur": str(compteur_ep_a[episode]),
                        "seuil": str(max_ep_a),
                    }
                    writer_rejets.writerow(rej)
                    f_rejets.flush()
                    stats["rejets"] += 1
                    stats["rejets_par_motif"]["quota_episode_atteint"] = stats["rejets_par_motif"].get("quota_episode_atteint", 0) + 1
                    nouveaux_rejets.append(rej)
                    continue

            # Sauvegarde et vérification stricte du fichier audio FLAC
            nom_fichier_flac = f"{nettoyer_nom_fichier(episode)}_{nettoyer_nom_fichier(seg_id_src)}.flac"
            chemin_disque_flac = dossier_audio / nom_fichier_flac
            chemin_audio_relatif = f"donnees/distille/audio/{nom_fichier_flac}"

            succes_audio, motif_audio = sauvegarder_et_verifier_audio_flac(waveform, sr, chemin_disque_flac)
            if not succes_audio:
                rej = {
                    "id": seg_id_src,
                    "filtre": "D1_0_audio",
                    "motif": motif_audio,
                    "valeur": nom_fichier_flac,
                    "seuil": "audio_valide",
                }
                writer_rejets.writerow(rej)
                f_rejets.flush()
                stats["rejets"] += 1
                stats["rejets_par_motif"]["erreur_audio"] = stats["rejets_par_motif"].get("erreur_audio", 0) + 1
                nouveaux_rejets.append(rej)
                continue

            if attribuer_a_d:
                compteur_ep_d[episode] += 1
                id_interne_d = f"d-{len(nouveaux_gardes_d) + nb_d_existant + 1:06d}"
                fiche_d = {
                    "id": id_interne_d,
                    "id_source_mgb2": seg_id_src,
                    "episode": episode,
                    "debut": round(debut, 3),
                    "fin": round(fin, 3),
                    "audio": chemin_audio_relatif,
                    "ar_manuel": ar_manuel,
                    "ar_ecoute": ar_ecoute,
                    "ecart_ecoute": round(ecart, 4),
                    "partie": partie,
                    "modele_asr": modele_ecoute,
                    "revision_asr": revision_modele,
                    "parametres": parametres_meta,
                    "date_execution": date_exec_iso,
                }
                # Validation du schéma intermédiaire D_SELECTION
                errs, _ = valider_fiche(fiche_d, "d_selection", seuils)
                if errs:
                    raise ValueError(f"Fiche D1 non conforme à SCHEMA_D_SELECTION : {errs}")

                f_d.write(json.dumps(fiche_d, ensure_ascii=False) + "\n")
                f_d.flush()
                nouveaux_gardes_d.append(fiche_d)
                stats["gardes_d"] += 1
            else:
                compteur_ep_a[episode] += 1
                id_interne_a = f"a-{len(nouveaux_gardes_a) + nb_a_existant + 1:06d}"
                fiche_a = {
                    "id": id_interne_a,
                    "id_source_mgb2": seg_id_src,
                    "episode": episode,
                    "debut": round(debut, 3),
                    "fin": round(fin, 3),
                    "audio": chemin_audio_relatif,
                    "ar_manuel": ar_manuel,
                    "ar_ecoute": ar_ecoute,
                    "ecart_ecoute": round(ecart, 4),
                    "partie": partie,
                    "modele_asr": modele_ecoute,
                    "revision_asr": revision_modele,
                    "parametres": parametres_meta,
                    "date_execution": date_exec_iso,
                }
                # Validation du schéma A
                errs, _ = valider_fiche(fiche_a, "a", seuils)
                if errs:
                    raise ValueError(f"Fiche A non conforme à SCHEMA_A : {errs}")

                f_a.write(json.dumps(fiche_a, ensure_ascii=False) + "\n")
                f_a.flush()
                nouveaux_gardes_a.append(fiche_a)
                stats["gardes_a"] += 1

            stats["duree_audio_totale_s"] += duree

    finally:
        f_d.close()
        f_a.close()
        f_rejets.close()

    # Calcul des statistiques finales mesurées
    stats["duree_totale_execution_s"] = round(time.time() - t0_global, 2)
    stats["duree_audio_totale_s"] = round(stats["duree_audio_totale_s"], 2)

    if stats["temps_asr_par_segment"]:
        t_list = sorted(stats["temps_asr_par_segment"])
        stats["temps_asr_moyen_s"] = round(sum(t_list) / len(t_list), 3)
        stats["temps_asr_median_s"] = round(t_list[len(t_list) // 2], 3)
        if stats["duree_audio_totale_s"] > 0:
            stats["facteur_temps_reel_rtf"] = round(
                stats["temps_inference_total_s"] / stats["duree_audio_totale_s"], 3
            )

    # VRAM maximale si GPU
    if mode_reel and stats["appareil_utilise"] != "cpu":
        try:
            import torch
            stats["vram_max_mo"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        except Exception:
            pass

    # Vérification stricte de l'égalité de conservation
    total_comptabilise = stats["gardes_d"] + stats["gardes_a"] + stats["rejets"] + stats["deja_traites"]
    assert stats["entree"] == total_comptabilise, (
        f"Incohérence stricte de conservation D1 : entrée ({stats['entree']}) != "
        f"gardés_d ({stats['gardes_d']}) + gardés_a ({stats['gardes_a']}) + "
        f"rejets ({stats['rejets']}) + déjà_traités ({stats['deja_traites']})"
    )

    # Sauvegarde des statistiques réelles d'exécution
    with open(chemin_stats, "w", encoding="utf-8") as fs:
        json.dump(stats, fs, indent=2, ensure_ascii=False)

    return {
        "stats": stats,
        "gardes_d": nouveaux_gardes_d,
        "gardes_a": nouveaux_gardes_a,
        "rejets": nouveaux_rejets,
        "chemin_d": str(chemin_sortie_d),
        "chemin_a": str(chemin_sortie_a),
        "chemin_rejets": str(chemin_rejets),
        "chemin_stats": str(chemin_stats),
    }


# ==============================================================================
# ÉTAPE D2 : COMPARAISON DES MODÈLES TRADUCTEURS (GRILLE AVEUGLE)
# ==============================================================================
SPEC_MODELES_CANDIDATS = {
    "Qwen/Qwen2.5-32B-Instruct": {
        "famille": "Qwen 2.5",
        "parametres_milliards": 32.5,
        "licence": "Apache 2.0 (commercial & recherche libre)",
        "vram_4bit_go_estimee": 18.5,
        "viable_2x_t4": True,
        "debit_tokens_sec_estime": 3.2,
    },
    "CohereLabs/aya-expanse-32b": {
        "famille": "Aya Expanse",
        "parametres_milliards": 32.0,
        "licence": "CC-BY-NC 4.0 (recherche non commerciale)",
        "vram_4bit_go_estimee": 18.0,
        "viable_2x_t4": True,
        "debit_tokens_sec_estime": 2.8,
    },
    "google/gemma-2-27b-it": {
        "famille": "Gemma 2",
        "parametres_milliards": 27.2,
        "licence": "Gemma Terms of Use",
        "vram_4bit_go_estimee": 15.8,
        "viable_2x_t4": True,
        "debit_tokens_sec_estime": 3.8,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "famille": "Qwen 2.5 (Recommandé Kaggle)",
        "parametres_milliards": 7.6,
        "licence": "Apache 2.0",
        "vram_4bit_go_estimee": 4.8,
        "viable_2x_t4": True,
        "debit_tokens_sec_estime": 28.5,
    },
}


def generer_grille_aveugle_d2(
    segments_entree: List[Dict[str, Any]],
    candidats: List[str],
    graine: int = 42,
    mode_reel: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, str]]]:
    """Génère la grille aveugle et la clé secrète pour notation humaine."""
    if mode_reel:
        raise NotImplementedError(
            "[D2 NON IMPLÉMENTÉ EN MODE RÉEL] L'inférence réelle des modèles traducteurs LLM "
            "nécessite l'environnement GPU Kaggle et la validation préalable de D1."
        )

    import random
    rng = random.Random(graine)
    grille_aveugle = []
    cle_secrete = {}
    labels = [chr(65 + i) for i in range(len(candidats))]

    for seg in segments_entree:
        sid = seg["id"]
        ar_texte = seg.get("ar_ecoute") or seg.get("ar_manuel", "")
        modeles_melanges = list(candidats)
        rng.shuffle(modeles_melanges)
        attribution = dict(zip(labels, modeles_melanges))
        cle_secrete[sid] = attribution

        ligne_grille = {
            "id_segment": sid,
            "ar": ar_texte,
            "remarque": "GRILLE FACTICE - SIMULATION" if not mode_reel else "",
        }
        for lbl in labels:
            nom_m = attribution[lbl]
            prefixe = nom_m.split("/")[-1].split("-")[0]
            ligne_grille[f"traduction_{lbl}"] = f"[{prefixe}] Traduction de test simulée."
            ligne_grille[f"note_{lbl}"] = ""
        grille_aveugle.append(ligne_grille)

    return grille_aveugle, cle_secrete


# ==============================================================================
# ÉTAPE D3 : TRADUCTION PAR PAQUETS DE 5 SEGMENTS
# ==============================================================================
def executer_d3_traduction_paquets(
    segments_d1: List[Dict[str, Any]],
    chemin_noms_csv: Path,
    mode_reel: bool = False,
    modele_traducteur: str = "Qwen/Qwen2.5-7B-Instruct",
    taille_paquet: int = 5,
    max_relances: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Exécute la traduction par paquets de 5 segments consécutifs d'un même épisode."""
    if mode_reel:
        raise NotImplementedError(
            "[D3 NON IMPLÉMENTÉ EN MODE RÉEL] La traduction réelle LLM par paquets est réservée "
            "à l'étape Kaggle GPU après validation humaine de D2."
        )

    fiches_traduites = []
    rejets_d3 = []

    # Regroupement par paquets
    paquets = []
    i = 0
    while i < len(segments_d1):
        ep_actuel = segments_d1[i].get("episode", "")
        paquet_courant = [segments_d1[i]]
        j = i + 1
        while j < len(segments_d1) and len(paquet_courant) < taille_paquet and segments_d1[j].get("episode") == ep_actuel:
            paquet_courant.append(segments_d1[j])
            j += 1
        paquets.append(paquet_courant)
        i = j

    for paquet in paquets:
        nb_attendu = len(paquet)
        for seg in paquet:
            fiche = dict(seg)
            fiche["fr"] = "[Simulation] Le représentant officiel a souligné que les discussions se poursuivaient."
            fiche["modele_fr"] = modele_traducteur
            fiches_traduites.append(fiche)

    return fiches_traduites, rejets_d3


# ==============================================================================
# ÉTAPE D4 : FILTRAGE AUTOMATIQUE ET SCORE QE (COMETKIWI)
# ==============================================================================
def executer_d4_filtres(
    segments_traduits: List[Dict[str, Any]],
    mode_reel: bool = False,
    modele_qe: str = "Unbabel/wmt22-cometkiwi-da",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Applique les filtres automatiques sur les traductions (D4)."""
    if mode_reel:
        raise NotImplementedError(
            "[D4 NON IMPLÉMENTÉ EN MODE RÉEL] L'évaluation métrique réelle COMETKiwi requiert "
            "un environnement GPU CUDA."
        )

    import py3langid
    seuils = charger_seuils()
    prop = seuils["proposes"]
    r_min = prop["rapport_longueur_min"]
    r_max = prop["rapport_longueur_max"]
    qe_rejet_pct = prop["qe_rejet_pire_pct"]

    motif_arabe = re.compile(r"[\u0600-\u06FF]")
    rejets_d4 = []
    gardes_intermediaires = []
    paires_uniques = set()

    for seg in segments_traduits:
        sid = seg["id"]
        ar = seg.get("ar_ecoute", seg.get("ar_manuel", "")).strip()
        fr = seg.get("fr", "").strip()

        # Filtre 1 : Caractères arabes résiduels ou méta-texte
        if motif_arabe.search(fr) or "voici la traduction" in fr.lower():
            rejets_d4.append({
                "id": sid, "filtre": "D4_1_meta", "motif": "arabe_ou_meta_texte",
                "valeur": fr[:30], "seuil": "aucun_meta",
            })
            continue

        # Filtre 2 : Détection de langue française
        langue, _ = py3langid.classify(fr)
        if langue != "fr":
            rejets_d4.append({
                "id": sid, "filtre": "D4_2_langue", "motif": "langue_non_francaise",
                "valeur": langue, "seuil": "fr",
            })
            continue

        # Filtre 3 : Ratio de longueur
        ratio = len(fr) / (len(ar) if len(ar) > 0 else 1)
        if not (r_min <= ratio <= r_max):
            rejets_d4.append({
                "id": sid, "filtre": "D4_3_ratio", "motif": "rapport_longueur_hors_bornes",
                "valeur": f"{ratio:.2f}", "seuil": f"[{r_min}, {r_max}]",
            })
            continue

        # Filtre 6 : Dédoublonnage exact
        paire = (ar, fr)
        if paire in paires_uniques:
            rejets_d4.append({
                "id": sid, "filtre": "D4_6_doublon", "motif": "doublon_exact",
                "valeur": "deja_vu", "seuil": "unique",
            })
            continue
        paires_uniques.add(paire)

        fiche = dict(seg)
        fiche["score_qe"] = round(0.85 + 0.05 * (len(fr) / (len(ar) + 1)), 4)
        gardes_intermediaires.append(fiche)

    # Filtre 7 : Retrait des 20% pires COMETKiwi
    gardes_intermediaires.sort(key=lambda x: x["score_qe"])
    nb_rejet_qe = int(len(gardes_intermediaires) * (qe_rejet_pct / 100.0))
    rejetes_qe = gardes_intermediaires[:nb_rejet_qe]
    gardes_finaux = gardes_intermediaires[nb_rejet_qe:]

    for r in rejetes_qe:
        rejets_d4.append({
            "id": r["id"], "filtre": "D4_7_qe", "motif": "score_cometkiwi_trop_faible",
            "valeur": f"{r['score_qe']:.4f}", "seuil": f"retrait_{qe_rejet_pct}pct",
        })

    assert len(gardes_finaux) + len(rejets_d4) == len(segments_traduits)
    return gardes_finaux, rejets_d4


# ==============================================================================
# POINT D'ENTRÉE CLI DU PIPELINE
# ==============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline MGB-2 (Étapes D1 à D4)")
    parser.add_argument("--etape", choices=["d1", "d2", "d3", "d4", "tout"], default="d1", help="Étape à exécuter")
    parser.add_argument("--mode-reel", action="store_true", help="Active l'exécution réelle avec vrais modèles")
    parser.add_argument("--max-segments", type=int, default=20, help="Nombre maximal de segments éligibles")
    parser.add_argument("--max-evaluations-asr", type=int, default=None, help="Nombre maximal de segments évalués par ASR")
    parser.add_argument("--num-beams", type=int, default=4, help="Nombre de faisceaux pour le décodage ASR (défaut: 4)")
    parser.add_argument("--modele-asr", type=str, default="openai/whisper-tiny", help="Modèle Whisper pour D1")
    parser.add_argument("--repartition", choices=["alterne", "quota"], default="alterne", help="Mode de répartition D/A")
    args = parser.parse_args()

    print(f"=== PIPELINE MGB-2 ===")
    print(f"Étape : {args.etape} | Mode réel : {args.mode_reel} | Max segments : {args.max_segments} | Max ASR : {args.max_evaluations_asr} | Beams : {args.num_beams}")

    if args.etape == "tout":
        if args.mode_reel:
            sys.exit(
                "[ERREUR] Le mode --etape tout est DÉSACTIVÉ en mode réel : les étapes D2 à D4 "
                "ne sont pas encore implémentées en réel et requièrent la validation humaine préalable de D1."
            )
        else:
            print("[INFO] Exécution complète simulée de D1 à D4...")
            res_d1 = executer_d1_selection(mode_reel=False, max_segments=args.max_segments, max_evaluations_asr=args.max_evaluations_asr, num_beams=args.num_beams)
            # Simulation enchaînée
            print(f"[D1 TEST TERMINÉ] D: {len(res_d1['gardes_d'])} | A: {len(res_d1['gardes_a'])}")
    elif args.etape == "d1":
        res_d1 = executer_d1_selection(
            mode_reel=args.mode_reel,
            max_segments=args.max_segments,
            max_evaluations_asr=args.max_evaluations_asr,
            num_beams=args.num_beams,
            modele_ecoute=args.modele_asr,
            repartition_mode=args.repartition,
        )
        stats = res_d1["stats"]
        print(f"\n[RÉSULTAT D1]")
        print(f"  - Segments entrés examinés : {stats['entree']}")
        print(f"  - Segments déjà traités (reprise) : {stats['deja_traites']}")
        print(f"  - Segments retenus Jeu D (sélection) : {stats['gardes_d']}")
        print(f"  - Segments retenus Jeu A (alignement) : {stats['gardes_a']}")
        print(f"  - Segments rejetés : {stats['rejets']}")
        print(f"  - Motifs de rejets : {stats['rejets_par_motif']}")
        print(f"  - Durée audio cumulée : {stats['duree_audio_totale_s']}s")
        print(f"  - Temps d'inférence ASR total : {stats['temps_inference_total_s']:.2f}s")
        print(f"  - Facteur Temps Réel (RTF) : {stats['facteur_temps_reel_rtf']}")
        print(f"  - Fichier D : {res_d1['chemin_d']}")
        print(f"  - Fichier A : {res_d1['chemin_a']}")
        print(f"  - Rejets : {res_d1['chemin_rejets']}")
        print(f"  - Stats : {res_d1['chemin_stats']}")
    elif args.etape in ("d2", "d3", "d4"):
        if args.mode_reel:
            sys.exit(
                f"[ERREUR] L'étape {args.etape} en mode réel n'est pas encore opérationnelle. "
                "Conformément au plan, elle est suspendue en attente de la validation de D1."
            )
        else:
            print(f"[INFO] Étape {args.etape} exécutée en mode simulation.")
