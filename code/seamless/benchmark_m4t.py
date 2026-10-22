#!/usr/bin/env python3
"""Benchmark SeamlessM4T sur le même extrait MGB-2 pour tout le groupe.

Le modèle lit le son et écrit le français, sans texte arabe intermédiaire.

Chemin principal : les 60 segments gelés du benchmark de traduction
(``resultats/traduction/segments_benchmark.jsonl``, lus depuis ``donnees/*.xml``).
``preparer_extraits_xml`` découpe leur son dans le wav de l'émission, et
``TraducteurSeamless`` écrit ``traductions_seamless-m4t-v2.jsonl`` au même format
que les traducteurs texte, ce qui l'ajoute à la grille aveugle.

Chemin plus ancien (dev MGB-2 au format jsonl) : les ``nb_max`` premiers identifiants triés dont la
durée connue ne dépasse pas 30 secondes. La liste est écrite dans
``resultats/seamless_m4t_ids.txt`` : S0, S2 et S3 doivent reprendre ces ids.

Le dossier ``data/mgb2`` du dépôt
https://github.com/nwujczak/nlp_project_mlp_cross_attention
est privé. ``resoudre_dossier_mgb2`` le clone en sparse checkout quand
les identifiants GitHub de la machine le permettent, sinon il lit une copie
déjà présente.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

RACINE_PROJET = Path(__file__).resolve().parents[2]
DOSSIER_MESURES = RACINE_PROJET / "code" / "mesures"
if str(DOSSIER_MESURES) not in sys.path:
    sys.path.insert(0, str(DOSSIER_MESURES))

from scores import calculer_chrf_bleu, calculer_rtf  # noqa: E402
from tableau_resultats import evaluer_sorties_systeme, lire_resultats  # noqa: E402

URL_DEPOT = "https://github.com/nwujczak/nlp_project_mlp_cross_attention.git"
MODELE_ID = "facebook/seamless-m4t-v2-large"
SYSTEME = "SEAMLESS"
LANGUE_CIBLE = "fra"
DUREE_MAX_S = 30.0
NB_SEGMENTS_DEFAUT = 20
SUFFIXES_AUDIO = {".wav", ".flac", ".ogg", ".mp3"}

CLES_ID = ("id", "segment_id", "utt_id", "utt", "id_source", "id_source_mgb2")
CLES_AUDIO = ("audio", "path", "wav", "fichier", "audio_path", "file", "chemin_audio")
CLES_FR = ("fr", "french", "translation", "target", "fr_traduit", "tgt", "francais")
CLES_AR = ("ar", "ar_manuel", "text_ar", "transcript", "transcription", "arabic", "src")
CLES_DUREE = ("duree_s", "duree", "duration")
CLES_PROGRAMME = ("program", "episode", "recording_id")


def apercu_arborescence(dossier: Path, limite: int = 40) -> str:
    """Liste les premiers fichiers pour diagnostiquer un format inattendu."""
    lignes: List[str] = []
    for chemin in sorted(dossier.rglob("*")):
        if ".git" in chemin.parts or not chemin.is_file():
            continue
        lignes.append(str(chemin.relative_to(dossier)))
        if len(lignes) >= limite:
            lignes.append("...")
            break
    return "\n".join(lignes) if lignes else "(dossier vide)"


def chemins_candidats(racine: Path) -> List[Path]:
    """Emplacements où le groupe peut avoir déjà déposé ``data/mgb2``."""
    return [
        racine / "donnees" / "externe" / "nlp_project_mlp_cross_attention" / "data" / "mgb2",
        racine.parent / "nlp_project_mlp_cross_attention" / "data" / "mgb2",
        racine / "data" / "mgb2",
    ]


def _dossier_a_des_fichiers(dossier: Path) -> bool:
    if not dossier.exists():
        return False
    for chemin in dossier.rglob("*"):
        if chemin.is_file() and ".git" not in chemin.parts:
            return True
    return False


def cloner_donnees_mgb2(destination: Path, url: str = URL_DEPOT) -> Path:
    """Clone uniquement ``data/mgb2``. Réutilise le dossier s'il contient déjà des fichiers."""
    destination = Path(destination)
    deja = destination / "data" / "mgb2"
    if _dossier_a_des_fichiers(deja):
        return deja

    destination.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    if not (destination / ".git").exists():
        commande = [
            "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
            url, str(destination),
        ]
        resultat = subprocess.run(commande, capture_output=True, text=True, env=env)
        if resultat.returncode != 0:
            detail = (resultat.stderr or resultat.stdout or "").strip()
            raise PermissionError(
                "Impossible de cloner le dépôt MGB-2. Il est privé : GitHub refuse "
                "l'accès sans identifiants sur cette machine.\n"
                f"{detail}\n"
                "Dans un terminal, après connexion à GitHub :\n"
                f"  git clone --depth 1 --filter=blob:none --sparse {url} {destination}\n"
                f"  git -C {destination} sparse-checkout set data/mgb2\n"
                "Puis relance le notebook. Le dossier attendu est "
                f"{deja}."
            )

    commande_sparse = ["git", "-C", str(destination), "sparse-checkout", "set", "data/mgb2"]
    resultat_sparse = subprocess.run(commande_sparse, capture_output=True, text=True, env=env)
    if resultat_sparse.returncode != 0:
        detail = (resultat_sparse.stderr or resultat_sparse.stdout or "").strip()
        raise RuntimeError(f"Le sparse checkout de data/mgb2 a échoué.\n{detail}")
    if not _dossier_a_des_fichiers(deja):
        raise FileNotFoundError(
            f"Le clone est terminé mais {deja} ne contient aucun fichier.\n"
            + apercu_arborescence(destination)
        )
    return deja


def resoudre_dossier_mgb2(
    racine: Path,
    dossier: Optional[str] = None,
    cloner: bool = True,
) -> Path:
    """Trouve ``data/mgb2`` localement, ou le clone dans ``donnees/externe``."""
    if dossier:
        cible = Path(dossier).expanduser()
        if not cible.exists():
            raise FileNotFoundError(f"Dossier MGB-2 introuvable : {cible}")
        return cible
    variable = os.environ.get("MGB2_DIR", "").strip()
    if variable:
        cible = Path(variable).expanduser()
        if not cible.exists():
            raise FileNotFoundError(f"MGB2_DIR introuvable : {cible}")
        return cible
    for candidat in chemins_candidats(racine):
        if _dossier_a_des_fichiers(candidat):
            return candidat
    if not cloner:
        raise FileNotFoundError(
            "Aucun data/mgb2 local. Définis MGB2_DIR ou clone le dépôt du groupe."
        )
    destination = racine / "donnees" / "externe" / "nlp_project_mlp_cross_attention"
    return cloner_donnees_mgb2(destination)


def _est_arabe(texte: str) -> bool:
    return any("\u0600" <= caractere <= "\u06FF" for caractere in texte)


def _normaliser_cle(cle: str) -> str:
    return str(cle).strip().lower().replace(" ", "_")


def _lire_flottant(ligne: Dict[str, Any], cles: Sequence[str]) -> Optional[float]:
    texte = _premiere_valeur(ligne, cles)
    if not texte:
        return None
    try:
        return float(texte.replace(",", "."))
    except ValueError:
        return None


def _premiere_valeur(ligne: Dict[str, Any], cles: Sequence[str]) -> str:
    index = {_normaliser_cle(cle): valeur for cle, valeur in ligne.items()}
    for cle in cles:
        valeur = index.get(cle)
        if valeur is None:
            continue
        if isinstance(valeur, dict):
            for sous_cle in ("path", "fichier", "file"):
                if valeur.get(sous_cle):
                    return str(valeur[sous_cle]).strip()
            continue
        texte = str(valeur).strip()
        if texte and texte.lower() not in {"none", "null", "nan"}:
            return texte
    return ""


def _indexer_audio(dossier: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for chemin in dossier.rglob("*"):
        if not chemin.is_file() or ".git" in chemin.parts:
            continue
        if chemin.suffix.lower() not in SUFFIXES_AUDIO:
            continue
        index.setdefault(chemin.name, chemin)
        index.setdefault(chemin.stem, chemin)
    return index


def _resoudre_audio(valeur: str, identifiant: str, index_audio: Dict[str, Path], racines: Sequence[Path]) -> Optional[Path]:
    if valeur:
        chemin = Path(valeur)
        candidats = [chemin]
        if not chemin.is_absolute():
            for racine in racines:
                candidats.append(racine / chemin)
                candidats.append(racine / "wav" / chemin.name)
                candidats.append(racine / "audio" / chemin.name)
        for candidat in candidats:
            if candidat.is_file():
                return candidat.resolve()
        if chemin.name in index_audio:
            return index_audio[chemin.name]
        if chemin.stem in index_audio:
            return index_audio[chemin.stem]
    for cle in (identifiant, Path(identifiant).name if identifiant else "", Path(identifiant).stem if identifiant else ""):
        if cle and cle in index_audio:
            return index_audio[cle]
    return None


def _duree_rapide(chemin: Path) -> Optional[float]:
    try:
        if chemin.suffix.lower() == ".wav":
            with wave.open(str(chemin), "rb") as fichier:
                taux = fichier.getframerate()
                if taux <= 0:
                    return None
                return fichier.getnframes() / float(taux)
        import soundfile as sf
        return float(sf.info(str(chemin)).duration)
    except Exception:
        return None


def lire_audio_mono_16k(chemin: Path, sr_cible: int = 16000) -> Tuple[Any, int, float]:
    """Lit wav/flac/ogg et renvoie un signal mono float32 à 16 kHz, plus la durée source."""
    import numpy as np

    if chemin.suffix.lower() == ".wav":
        with wave.open(str(chemin), "rb") as fichier:
            taux = fichier.getframerate()
            canaux = fichier.getnchannels()
            largeur = fichier.getsampwidth()
            brut = fichier.readframes(fichier.getnframes())
        if largeur == 2:
            signal = np.frombuffer(brut, dtype="<i2").astype(np.float32) / 32768.0
        elif largeur == 4:
            signal = np.frombuffer(brut, dtype="<i4").astype(np.float32) / 2147483648.0
        elif largeur == 1:
            signal = (np.frombuffer(brut, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"Largeur d'échantillon wav non gérée ({largeur} octets) : {chemin}")
        if canaux > 1:
            signal = signal.reshape(-1, canaux).mean(axis=1)
    else:
        import soundfile as sf
        audio, taux = sf.read(str(chemin), dtype="float32", always_2d=False)
        signal = np.asarray(audio, dtype=np.float32)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
    if signal.size == 0 or taux <= 0:
        raise ValueError(f"Audio vide : {chemin}")
    duree = float(signal.shape[0]) / float(taux)
    if taux != sr_cible:
        nb_cible = max(1, int(round(duree * sr_cible)))
        axe_source = np.linspace(0.0, 1.0, num=signal.shape[0], endpoint=False)
        axe_cible = np.linspace(0.0, 1.0, num=nb_cible, endpoint=False)
        signal = np.interp(axe_cible, axe_source, signal).astype(np.float32)
    return signal, sr_cible, duree


def _enregistrements_depuis_objet(donnees: Any) -> List[Dict[str, Any]]:
    if isinstance(donnees, list):
        return [ligne for ligne in donnees if isinstance(ligne, dict)]
    if isinstance(donnees, dict):
        for cle in ("data", "segments", "rows", "examples"):
            if isinstance(donnees.get(cle), list):
                return [ligne for ligne in donnees[cle] if isinstance(ligne, dict)]
        if any(_normaliser_cle(cle) in CLES_ID + CLES_AUDIO + CLES_FR + CLES_AR for cle in donnees):
            return [donnees]
    return []


def _lire_table(chemin: Path) -> List[Dict[str, Any]]:
    suffixe = chemin.suffix.lower()
    if suffixe in {".jsonl", ".json"}:
        texte = chemin.read_text(encoding="utf-8-sig").strip()
        if not texte:
            return []
        if texte[0] in "[{":
            try:
                return _enregistrements_depuis_objet(json.loads(texte))
            except json.JSONDecodeError:
                if suffixe == ".json":
                    return []
        lignes = []
        for ligne in texte.splitlines():
            ligne = ligne.strip()
            if ligne:
                lignes.append(json.loads(ligne))
        return _enregistrements_depuis_objet(lignes)
    if suffixe == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                f"{chemin.name} est un parquet. Installe pyarrow pour le lire."
            ) from exc
        table = pq.read_table(chemin)
        return _enregistrements_depuis_objet(table.to_pylist())
    with open(chemin, "r", encoding="utf-8-sig", newline="") as fichier:
        echantillon = fichier.read(4096)
        fichier.seek(0)
        try:
            dialecte = csv.Sniffer().sniff(echantillon, delimiters=",;\t|")
        except csv.Error:
            dialecte = csv.excel
        return [dict(ligne) for ligne in csv.DictReader(fichier, dialect=dialecte)]


def _fichiers_tables(dossier: Path) -> List[Path]:
    fichiers = []
    for chemin in dossier.rglob("*"):
        if not chemin.is_file() or ".git" in chemin.parts:
            continue
        if chemin.name in {"dataset_info.json", "state.json", "dataset_dict.json"}:
            continue
        if chemin.suffix.lower() in {".jsonl", ".json", ".csv", ".tsv", ".parquet"}:
            fichiers.append(chemin)
    return sorted(fichiers)


def _segment_depuis_ligne(
    ligne: Dict[str, Any],
    index_audio: Dict[str, Path],
    racines: Sequence[Path],
    repli_id: str,
) -> Optional[Dict[str, Any]]:
    identifiant = _premiere_valeur(ligne, CLES_ID) or repli_id
    audio_texte = _premiere_valeur(ligne, CLES_AUDIO)
    francais = _premiere_valeur(ligne, CLES_FR)
    arabe = _premiere_valeur(ligne, CLES_AR)
    if not arabe:
        texte = _premiere_valeur(ligne, ("text", "sentence", "raw_text"))
        if texte and _est_arabe(texte):
            arabe = texte
        elif texte and not francais:
            francais = texte
    chemin_audio = _resoudre_audio(audio_texte, identifiant, index_audio, racines)
    duree_texte = _premiere_valeur(ligne, CLES_DUREE)
    duree = None
    if duree_texte:
        try:
            duree = float(duree_texte.replace(",", "."))
        except ValueError:
            duree = None
    debut = _lire_flottant(ligne, ("start", "debut"))
    fin = _lire_flottant(ligne, ("end", "fin"))
    if duree is None and debut is not None and fin is not None and fin > debut:
        duree = fin - debut
    programme = _premiere_valeur(ligne, CLES_PROGRAMME)
    if chemin_audio is None and not francais and not arabe and not programme:
        return None
    return {
        "id": identifiant,
        "chemin_audio": str(chemin_audio) if chemin_audio else "",
        "fr": francais,
        "ar": arabe,
        "duree_s": duree,
        "program": programme,
        "debut": debut,
        "fin": fin,
    }


def _lire_kaldi(dossier: Path, index_audio: Dict[str, Path]) -> List[Dict[str, Any]]:
    textes: Dict[str, Dict[str, str]] = {}
    wavs: Dict[str, str] = {}
    for chemin in dossier.rglob("*"):
        if not chemin.is_file() or ".git" in chemin.parts:
            continue
        nom = chemin.name
        if nom in {"wav.scp"}:
            for ligne in chemin.read_text(encoding="utf-8", errors="replace").splitlines():
                morceaux = ligne.split(maxsplit=1)
                if len(morceaux) == 2 and not morceaux[1].endswith("|"):
                    wavs[morceaux[0]] = morceaux[1].strip()
        if nom in {"text", "text.ar", "text.non_overlap_speech", "text.fr", "text.french"}:
            champ = "fr" if nom in {"text.fr", "text.french"} else "ar"
            for ligne in chemin.read_text(encoding="utf-8", errors="replace").splitlines():
                morceaux = ligne.split(maxsplit=1)
                if len(morceaux) != 2:
                    continue
                textes.setdefault(morceaux[0], {})[champ] = morceaux[1].strip()
    if not textes and not wavs:
        return []
    identifiants = sorted(set(textes) | set(wavs))
    segments = []
    for identifiant in identifiants:
        champs = textes.get(identifiant, {})
        arabe = champs.get("ar", "")
        francais = champs.get("fr", "")
        if arabe and not _est_arabe(arabe) and not francais:
            francais, arabe = arabe, ""
        chemin_audio = _resoudre_audio(wavs.get(identifiant, ""), identifiant, index_audio, [dossier])
        segments.append({
            "id": identifiant,
            "chemin_audio": str(chemin_audio) if chemin_audio else "",
            "fr": francais,
            "ar": arabe,
            "duree_s": None,
        })
    return segments


def _lire_hf_disk(dossier: Path) -> List[Dict[str, Any]]:
    marqueurs = [dossier / "dataset_dict.json", dossier / "state.json"]
    if not any(chemin.exists() for chemin in marqueurs):
        return []
    from datasets import load_from_disk

    jeu = load_from_disk(str(dossier))
    if hasattr(jeu, "keys"):
        for nom in ("dev", "validation", "valid", "test", "train"):
            if nom in jeu:
                jeu = jeu[nom]
                break
        else:
            premiere = next(iter(jeu.keys()))
            jeu = jeu[premiere]
    segments = []
    for indice, ligne in enumerate(jeu):
        audio = ligne.get("audio")
        chemin = ""
        if isinstance(audio, dict) and audio.get("path"):
            chemin = str(audio["path"])
        segment = _segment_depuis_ligne(dict(ligne), {}, [dossier], f"hf-{indice:06d}")
        if segment is None:
            continue
        if chemin and not segment["chemin_audio"]:
            resolu = _resoudre_audio(chemin, segment["id"], {}, [dossier])
            segment["chemin_audio"] = str(resolu) if resolu else chemin
        segments.append(segment)
    return segments


def lister_segments(dossier: Path, split: Optional[str] = "dev") -> List[Dict[str, Any]]:
    """Lit jsonl, csv, parquet, Kaldi ou un dataset Hugging Face déjà sur disque.

    ``split="dev"`` ne lit que ``dev/`` quand ce dossier existe, pour que les
    20 identifiants restent ceux du développement MGB-2 et non du test.
    """
    dossier = Path(dossier)
    if not dossier.exists():
        raise FileNotFoundError(f"Dossier introuvable : {dossier}")
    zone = dossier / split if split and (dossier / split).is_dir() else dossier
    index_audio = _indexer_audio(zone)
    racines = [zone, dossier]
    segments: List[Dict[str, Any]] = []
    vus = set()

    def _ajouter(candidats: Iterable[Dict[str, Any]]) -> None:
        for segment in candidats:
            identifiant = str(segment.get("id") or "").strip()
            if not identifiant or identifiant in vus:
                continue
            vus.add(identifiant)
            segments.append(segment)

    try:
        _ajouter(_lire_hf_disk(zone))
    except Exception:
        pass
    for fichier in _fichiers_tables(zone):
        try:
            lignes = _lire_table(fichier)
        except ImportError:
            raise
        except Exception:
            continue
        construits = []
        for indice, ligne in enumerate(lignes):
            segment = _segment_depuis_ligne(
                ligne, index_audio, racines + [fichier.parent], f"{fichier.stem}-{indice:06d}"
            )
            if segment is not None:
                segment["dossier_split"] = str(fichier.parent)
                segment["split"] = fichier.parent.name
                construits.append(segment)
        _ajouter(construits)
    if not segments:
        _ajouter(_lire_kaldi(zone, index_audio))
    if not segments:
        for identifiant, chemin in sorted(index_audio.items()):
            if identifiant != chemin.stem:
                continue
            _ajouter([{
                "id": chemin.stem,
                "chemin_audio": str(chemin),
                "fr": "",
                "ar": "",
                "duree_s": None,
            }])
    if not segments:
        raise FileNotFoundError(
            f"Aucun segment lisible dans {zone}.\n" + apercu_arborescence(zone)
        )
    return segments


def selectionner_segments(
    segments: Sequence[Dict[str, Any]],
    nb_max: int = NB_SEGMENTS_DEFAUT,
    duree_max_s: float = DUREE_MAX_S,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Garde les ``nb_max`` premiers ids triés, en écartant l'audio sans fichier ou trop long."""
    choisis: List[Dict[str, Any]] = []
    ignores_longs = 0
    sans_audio = 0
    ids_sans_audio: List[str] = []
    for brut in sorted(segments, key=lambda s: str(s["id"])):
        if len(choisis) >= nb_max:
            break
        segment = dict(brut)
        chemin = str(segment.get("chemin_audio") or "")
        a_un_fichier = bool(chemin) and Path(chemin).is_file()
        peut_decouper = (
            bool(segment.get("program"))
            and segment.get("debut") is not None
            and segment.get("fin") is not None
        )
        if not a_un_fichier and not peut_decouper:
            sans_audio += 1
            if len(ids_sans_audio) < 8:
                ids_sans_audio.append(str(segment.get("id")))
            continue
        if not a_un_fichier:
            duree = segment.get("duree_s")
            if duree is not None and float(duree) > duree_max_s:
                ignores_longs += 1
                continue
            choisis.append(segment)
            continue
        duree = segment.get("duree_s")
        if duree is None:
            duree = _duree_rapide(Path(chemin))
            segment["duree_s"] = duree
        if duree is not None and float(duree) > duree_max_s:
            ignores_longs += 1
            continue
        choisis.append(segment)
    return choisis, {
        "retenus": len(choisis),
        "ignores_longs": ignores_longs,
        "sans_audio": sans_audio,
        "ids_sans_audio": ids_sans_audio,
    }


def _nom_extrait(identifiant: str) -> str:
    propre = "".join(caractere if caractere.isalnum() or caractere in "._-" else "_" for caractere in identifiant)
    return (propre[:180] or "segment") + ".wav"


def _archive_du_split(dossier_split: Path) -> Optional[Path]:
    nom = dossier_split.name
    for candidat in (
        dossier_split / f"{nom}.tar.gz",
        dossier_split / f"{nom}.tar",
        dossier_split.parent / f"{nom}.tar.gz",
    ):
        if candidat.is_file():
            return candidat
    return None


def _message_audio_manquant(dossier_split: Path) -> str:
    archive = dossier_split / f"{dossier_split.name}.tar.gz"
    return (
        "Les transcriptions sont lisibles, l'audio des émissions n'est pas dans le dépôt "
        "(git ignore les archives).\n"
        f"Fichier attendu : {archive}\n"
        "Depuis le dossier data/mgb2 du dépôt nlp_project_mlp_cross_attention, avec un "
        "jeton Hugging Face et la licence QCRI acceptée :\n"
        "  from huggingface_hub import snapshot_download\n"
        "  snapshot_download('QCRI/mgb2', repo_type='dataset',\n"
        f"      allow_patterns=['{dossier_split.name}/{dossier_split.name}.tar.gz'],\n"
        f"      local_dir='{dossier_split.parent}')"
    )


def _indexer_archive_wav(archive: Path) -> Dict[str, str]:
    import tarfile

    index: Dict[str, str] = {}
    with tarfile.open(archive, "r:gz" if archive.suffix == ".gz" else "r:") as flux:
        for membre in flux:
            if membre.isfile() and membre.name.endswith(".wav"):
                index.setdefault(Path(membre.name).stem, membre.name)
    return index


def _extraire_emission(archive: Path, nom_membre: str, destination: Path) -> Path:
    import tarfile

    if destination.is_file() and destination.stat().st_size > 44:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporaire = destination.with_suffix(".wav.part")
    with tarfile.open(archive, "r:gz" if archive.suffix == ".gz" else "r:") as flux:
        source = flux.extractfile(nom_membre)
        if source is None:
            raise FileNotFoundError(f"{nom_membre} absent de {archive}")
        with open(temporaire, "wb") as sortie:
            while True:
                bloc = source.read(1024 * 1024)
                if not bloc:
                    break
                sortie.write(bloc)
    temporaire.replace(destination)
    return destination


def decouper_wav(source: Path, debut_s: float, fin_s: float, cible: Path) -> Path:
    """Découpe [debut_s, fin_s] d'un wav PCM et l'écrit dans ``cible``."""
    with wave.open(str(source), "rb") as entree:
        taux = entree.getframerate()
        if taux <= 0:
            raise ValueError(f"Fréquence invalide dans {source}")
        debut = max(0, int(debut_s * taux))
        fin = min(entree.getnframes(), int(fin_s * taux))
        if fin <= debut:
            raise ValueError(f"Tranche vide {debut_s}-{fin_s} dans {source}")
        entree.setpos(debut)
        trames = entree.readframes(fin - debut)
        parametres = entree.getparams()
    cible.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(cible), "wb") as sortie:
        sortie.setparams(parametres)
        sortie.writeframes(trames)
    return cible


def _wav_emission(
    dossier_split: Path,
    programme: str,
    cache: Path,
    index_archives: Dict[str, Dict[str, str]],
) -> Path:
    for candidat in (
        dossier_split / f"{programme}.wav",
        dossier_split / "wav" / f"{programme}.wav",
        cache / "emissions" / f"{programme}.wav",
    ):
        if candidat.is_file() and candidat.stat().st_size > 44:
            return candidat
    archive = _archive_du_split(dossier_split)
    if archive is None:
        raise FileNotFoundError(_message_audio_manquant(dossier_split))
    cle_archive = str(archive)
    if cle_archive not in index_archives:
        index_archives[cle_archive] = _indexer_archive_wav(archive)
    nom_membre = index_archives[cle_archive].get(programme)
    if nom_membre is None:
        raise FileNotFoundError(
            f"L'émission {programme}.wav n'est pas dans {archive}."
        )
    return _extraire_emission(archive, nom_membre, cache / "emissions" / f"{programme}.wav")


def attacher_extraits(segments: Sequence[Dict[str, Any]], cache: Path) -> List[Dict[str, Any]]:
    """Découpe le wav de chaque segment. Réutilise un fichier déjà présent."""
    cache = Path(cache)
    index_archives: Dict[str, Dict[str, str]] = {}
    sorties: List[Dict[str, Any]] = []
    for brut in segments:
        segment = dict(brut)
        chemin = str(segment.get("chemin_audio") or "")
        if chemin and Path(chemin).is_file():
            sorties.append(segment)
            continue
        programme = str(segment.get("program") or "")
        debut = segment.get("debut")
        fin = segment.get("fin")
        dossier_split = Path(str(segment.get("dossier_split") or ""))
        if not programme or debut is None or fin is None or not dossier_split:
            raise FileNotFoundError(
                f"Segment {segment.get('id')} sans fichier audio ni horodatage d'émission."
            )
        source = _wav_emission(dossier_split, programme, cache, index_archives)
        cible = cache / "extraits" / _nom_extrait(str(segment["id"]))
        if not cible.is_file():
            decouper_wav(source, float(debut), float(fin), cible)
        segment["chemin_audio"] = str(cible)
        sorties.append(segment)
    return sorties


def extraire_ids_tokens(sortie: Any) -> List[int]:
    """Aplati la sortie ``generate`` de SeamlessM4T jusqu'à une liste d'identifiants."""
    if hasattr(sortie, "sequences"):
        sortie = sortie.sequences
    if isinstance(sortie, (tuple, list)) and sortie:
        sortie = sortie[0]
    brut = sortie.tolist() if hasattr(sortie, "tolist") else sortie
    while isinstance(brut, (list, tuple)) and brut and isinstance(brut[0], (list, tuple)):
        brut = brut[0]
    return [int(token) for token in brut]


def decoder_texte(processor: Any, sortie: Any) -> str:
    identifiants = extraire_ids_tokens(sortie)
    try:
        texte = processor.decode(identifiants, skip_special_tokens=True)
    except Exception:
        texte = processor.batch_decode([identifiants], skip_special_tokens=True)[0]
    return str(texte).strip()


def charger_modele_m4t(modele_id: str = MODELE_ID):
    """Charge SeamlessM4T v2 (ou v1 si l'identifiant ne contient pas ``v2``)."""
    try:
        import torch
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "Paquets manquants. À partir de requirements-fige.txt, installe aussi "
            "sentencepiece, soundfile et sacrebleu."
        ) from exc
    if "v2" in modele_id:
        from transformers import SeamlessM4Tv2Model as Classe
    else:
        from transformers import SeamlessM4TModel as Classe
    if torch.cuda.is_available():
        appareil, dtype = "cuda", torch.float16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        appareil, dtype = "mps", torch.float32
    else:
        appareil, dtype = "cpu", torch.float32
    processor = AutoProcessor.from_pretrained(modele_id)
    try:
        modele = Classe.from_pretrained(modele_id, dtype=dtype)
    except TypeError:
        modele = Classe.from_pretrained(modele_id, torch_dtype=dtype)
    modele.to(appareil)
    modele.eval()
    return modele, processor, appareil


def fabriquer_traducteur(
    modele_id: str = MODELE_ID,
    nb_faisceaux: int = 5,
) -> Callable[[Any, int], Tuple[str, float]]:
    """Renvoie une fonction (audio 16 kHz, taux) -> (français, secondes de calcul)."""
    import torch

    modele, processor, appareil = charger_modele_m4t(modele_id)
    dtype_modele = next(modele.parameters()).dtype

    def preparer(audio: Any, taux: int) -> Dict[str, Any]:
        try:
            return processor(audio=audio, sampling_rate=taux, return_tensors="pt")
        except (TypeError, ValueError):
            # transformers < 4.47 n'accepte que ``audios``
            return processor(audios=audio, sampling_rate=taux, return_tensors="pt")

    def vers_modele(valeur: Any) -> Any:
        if not hasattr(valeur, "to"):
            return valeur
        if torch.is_floating_point(valeur):
            return valeur.to(appareil, dtype=dtype_modele)
        return valeur.to(appareil)

    def traduire(audio: Any, taux: int) -> Tuple[str, float]:
        depart = time.perf_counter()
        entrees = {cle: vers_modele(valeur) for cle, valeur in preparer(audio, taux).items()}
        arguments = {
            "tgt_lang": LANGUE_CIBLE,
            "generate_speech": False,
            "text_num_beams": nb_faisceaux,
        }
        with torch.inference_mode():
            try:
                sortie = modele.generate(**entrees, **arguments)
            except (TypeError, ValueError):
                arguments.pop("text_num_beams")
                sortie = modele.generate(**entrees, **arguments)
        return decoder_texte(processor, sortie), time.perf_counter() - depart

    return traduire


CANDIDAT_SEAMLESS = {"nom": "seamless-m4t-v2", "modele_id": MODELE_ID, "type": "audio", "licence": "CC-BY-NC 4.0"}


def trouver_audio_episode(episode: str, dossiers: Sequence[Path]) -> Optional[Path]:
    """Cherche ``<episode>.wav`` (ou flac, mp3, ogg) dans les dossiers donnés et leurs sous-dossiers."""
    for dossier in dossiers:
        dossier = Path(dossier)
        if not dossier.is_dir():
            continue
        for suffixe in (".wav", ".flac", ".mp3", ".ogg"):
            for chemin in dossier.rglob(f"{episode}{suffixe}"):
                if chemin.is_file() and chemin.stat().st_size > 44:
                    return chemin
    return None


def decouper_extrait(source: Path, debut_s: float, fin_s: float, cible: Path) -> Path:
    """Tranche [debut_s, fin_s] d'un fichier audio, écrite en wav."""
    if source.suffix.lower() == ".wav":
        try:
            return decouper_wav(source, debut_s, fin_s, cible)
        except wave.Error:
            pass
    import soundfile as sf

    taux = sf.info(str(source)).samplerate
    audio, taux = sf.read(str(source), start=int(debut_s * taux), stop=int(fin_s * taux), dtype="float32", always_2d=False)
    if len(audio) == 0:
        raise ValueError(f"Tranche vide {debut_s}-{fin_s} dans {source}")
    cible.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(cible), audio, taux, subtype="PCM_16")
    return cible


def trouver_archives(dossiers: Sequence[Path]) -> List[Path]:
    """Archives MGB-2 (``test.tar.gz``, ``dev.tar.gz``…) posées dans les dossiers audio."""
    archives: List[Path] = []
    for dossier in dossiers:
        dossier = Path(dossier)
        if not dossier.is_dir():
            continue
        for motif in ("*.tar.gz", "*.tgz", "*.tar"):
            archives.extend(chemin for chemin in sorted(dossier.rglob(motif)) if chemin not in archives)
    return archives


def extraire_episodes_archive(
    archive: Path,
    episodes: Optional[Iterable[str]],
    destination: Path,
    nb_max: Optional[int] = None,
) -> Tuple[Dict[str, Path], List[str]]:
    """Lit l'archive une seule fois et copie ``<episode>.wav`` dans ``destination``.

    ``episodes=None`` prend les ``nb_max`` premières émissions de l'archive et s'arrête là.
    Renvoie les émissions extraites et la liste des émissions vues dans l'archive.
    """
    import tarfile

    voulus = None if episodes is None else set(episodes)
    trouves: Dict[str, Path] = {}
    contenu: List[str] = []
    destination = Path(destination)
    with tarfile.open(archive, "r:*") as flux:
        for membre in flux:
            if nb_max is not None and len(trouves) >= nb_max:
                break
            if not (membre.isfile() and membre.name.endswith(".wav")):
                continue
            episode = Path(membre.name).stem
            contenu.append(episode)
            if (voulus is not None and episode not in voulus) or episode in trouves:
                continue
            cible = destination / f"{episode}.wav"
            if not (cible.is_file() and cible.stat().st_size > 44):
                source = flux.extractfile(membre)
                if source is None:
                    continue
                cible.parent.mkdir(parents=True, exist_ok=True)
                temporaire = cible.with_suffix(".wav.part")
                with open(temporaire, "wb") as sortie:
                    while True:
                        bloc = source.read(1024 * 1024)
                        if not bloc:
                            break
                        sortie.write(bloc)
                temporaire.replace(cible)
            trouves[episode] = cible
    return trouves, sorted(contenu)


def decouper_emission(
    source: Path,
    nb: int,
    duree_s: float,
    dossier: Path,
    marge_s: float = 60.0,
    rayon_s: float = 2.0,
) -> List[Dict[str, Any]]:
    """Tire ``nb`` extraits d'environ ``duree_s`` secondes, répartis sur l'émission.

    Chaque bord est déplacé sur la trame de 50 ms la plus silencieuse à ``rayon_s`` près,
    pour ne pas couper un mot. Le début et la fin de l'émission (génériques) sont évités.
    """
    import numpy as np

    source = Path(source)
    audio, taux, duree = lire_audio_mono_16k(source)
    pas_trame = 0.05
    taille = int(pas_trame * taux)
    nb_trames = len(audio) // taille
    energie = np.sqrt(np.mean(np.square(audio[: nb_trames * taille].reshape(nb_trames, taille)), axis=1))

    def pause(t: float) -> float:
        i0 = max(0, int((t - rayon_s) / pas_trame))
        i1 = min(nb_trames, int((t + rayon_s) / pas_trame) + 1)
        if i1 <= i0:
            return min(max(t, 0.0), duree)
        return (i0 + int(np.argmin(energie[i0:i1]))) * pas_trame

    zone_debut = min(marge_s, duree / 4)
    zone_fin = max(duree - marge_s, 3 * duree / 4)
    ecart = max(0.0, zone_fin - zone_debut - duree_s) / max(1, nb - 1)
    extraits = []
    for k in range(nb):
        debut = pause(zone_debut + k * ecart)
        fin = min(pause(debut + duree_s), debut + DUREE_MAX_S, duree)
        if fin - debut < duree_s / 2:
            fin = min(debut + duree_s, duree)
        identifiant = f"{source.stem}_{int(debut * 100):07d}_{int(fin * 100):07d}"
        cible = Path(dossier) / _nom_extrait(identifiant)
        if not cible.is_file():
            decouper_extrait(source, debut, fin, cible)
        extraits.append({
            "id": identifiant,
            "episode": source.stem,
            "debut": round(debut, 2),
            "fin": round(fin, 2),
            "duree_s": round(fin - debut, 2),
            "chemin_audio": str(cible),
        })
    return extraits


def extraits_depuis_archive(
    archive: Path,
    cache: Path,
    nb_emissions: int = 3,
    nb_par_emission: int = 10,
    duree_s: float = 15.0,
) -> List[Dict[str, Any]]:
    """Prend les premières émissions d'une archive MGB-2 et en tire des extraits, sans transcription."""
    cache = Path(cache)
    emissions = sorted((cache / "emissions").glob("*.wav"))
    if len(emissions) < nb_emissions:
        trouves, _ = extraire_episodes_archive(archive, None, cache / "emissions", nb_max=nb_emissions)
        emissions = sorted(trouves.values())
    if not emissions:
        raise FileNotFoundError(f"Aucun wav dans {archive}")
    extraits: List[Dict[str, Any]] = []
    for source in emissions[:nb_emissions]:
        extraits.extend(decouper_emission(source, nb_par_emission, duree_s, cache / "extraits"))
    return extraits


def message_audio_xml_manquant(
    episodes: Sequence[str],
    dossiers: Sequence[Path],
    archives_lues: Optional[Dict[Path, List[str]]] = None,
) -> str:
    noms = "\n".join(f"  {e}.wav" for e in episodes)
    lieux = "\n".join(f"  {d}" for d in dossiers)
    premier = episodes[0]
    texte = (
        "Le XML donne le texte et les horodatages, pas le son. Fichier(s) audio manquant(s) :\n"
        f"{noms}\n"
        f"Dépose-les (ou l'archive MGB-2 qui les contient) dans l'un de ces dossiers :\n{lieux}\n"
    )
    if archives_lues:
        texte += "Archive(s) lue(s), sans ces émissions :\n"
        for archive, contenu in archives_lues.items():
            apercu = ", ".join(contenu[:5]) + (" …" if len(contenu) > 5 else "")
            texte += f"  {archive.name} : {len(contenu)} émissions ({apercu})\n"
        texte += (
            "Les émissions d'un split ne sont pas dans les autres : le XML doit venir du même "
            "split que l'archive. Pour tester avec cette archive, mets dans donnees/ le XML "
            "d'une de ses émissions et supprime segments_benchmark.jsonl pour retirer le jeu.\n"
        )
    return texte + (
        "Ils sont dans l'archive MGB-2 de l'équipe (train/wav/). Pour extraire une seule émission :\n"
        f"  tar -xzf train.tar.gz --wildcards '*{premier}.wav'\n"
        "L'archive se lit en entier pour trouver le fichier : demande-le plutôt à la personne "
        "qui a déjà décompressé les 128 Go."
    )


def preparer_extraits_xml(
    segments: Sequence[Dict[str, Any]],
    dossiers_audio: Sequence[Path],
    cache: Path,
    marge_s: float = 0.0,
) -> List[Dict[str, Any]]:
    """Ajoute ``chemin_audio`` à chaque segment XML, découpé depuis le son de son émission.

    Le wav est cherché tel quel, puis dans les archives MGB-2 des mêmes dossiers.
    ``marge_s`` élargit chaque tranche des deux côtés (horodatages arrondis à la seconde).
    """
    episodes = sorted({s["episode"] for s in segments})
    dossiers = list(dossiers_audio) + [Path(cache) / "emissions"]
    sources = {e: trouver_audio_episode(e, dossiers) for e in episodes}
    archives_lues: Dict[Path, List[str]] = {}
    for archive in trouver_archives(dossiers_audio):
        manquants = [e for e, chemin in sources.items() if chemin is None]
        if not manquants:
            break
        trouves, archives_lues[archive] = extraire_episodes_archive(archive, manquants, Path(cache) / "emissions")
        sources.update(trouves)
    manquants = [e for e, chemin in sources.items() if chemin is None]
    if manquants:
        raise FileNotFoundError(message_audio_xml_manquant(manquants, dossiers_audio, archives_lues))
    sorties = []
    for segment in segments:
        cible = Path(cache) / "extraits_xml" / _nom_extrait(str(segment["id"]))
        if not cible.is_file():
            debut = max(0.0, float(segment["debut"]) - marge_s)
            decouper_extrait(sources[segment["episode"]], debut, float(segment["fin"]) + marge_s, cible)
        sorties.append({**segment, "chemin_audio": str(cible)})
    return sorties


class TraducteurSeamless:
    """Même interface que les traducteurs texte de ``code/traduction`` : liste de segments -> (français, secondes)."""

    def __init__(self, modele_id: str = MODELE_ID, nb_faisceaux: int = 5, duree_max_s: float = DUREE_MAX_S):
        self._traduire = fabriquer_traducteur(modele_id, nb_faisceaux=nb_faisceaux)
        self.duree_max_s = duree_max_s

    def traduire(self, segments: Sequence[Dict[str, Any]]) -> List[Tuple[str, float]]:
        sorties = []
        for segment in segments:
            audio, taux, _ = lire_audio_mono_16k(Path(segment["chemin_audio"]))
            audio = audio[: int(self.duree_max_s * taux)]
            sorties.append(self._traduire(audio, taux))
        return sorties

    def fermer(self) -> None:
        import gc

        self._traduire = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def executer_benchmark(
    segments: Sequence[Dict[str, Any]],
    traduire: Callable[[Any, int], Tuple[str, float]],
    dossier_sortie: Path,
    jeu: str = "mgb2",
    modele_id: str = MODELE_ID,
    configuration: str = "tgt_lang=fra",
    duree_max_s: float = DUREE_MAX_S,
) -> Dict[str, Any]:
    """Traduit chaque segment, mesure chrF++, BLEU et le facteur temps réel, écrit le CSV du projet."""
    dossier_sortie = Path(dossier_sortie)
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    references = []
    hypotheses = []
    details = []
    for segment in segments:
        chemin = Path(segment["chemin_audio"])
        audio, taux, duree_source = lire_audio_mono_16k(chemin)
        nb_max = int(duree_max_s * taux)
        if audio.shape[0] > nb_max:
            audio = audio[:nb_max]
        duree = float(audio.shape[0]) / float(taux)
        francais, temps = traduire(audio, taux)
        references.append({
            "id": segment["id"],
            "fr": segment.get("fr") or "",
            "ar": segment.get("ar") or "",
            "duree_s": duree,
        })
        hypotheses.append({
            "id": segment["id"],
            "fr": francais,
            "fr_traduit": francais,
            "temps_calcul_s": round(temps, 4),
        })
        details.append({
            "id": segment["id"],
            "fr": francais,
            "fr_ref": segment.get("fr") or "",
            "ar": segment.get("ar") or "",
            "temps_calcul_s": round(temps, 4),
            "duree_s": round(duree, 3),
            "duree_source_s": round(duree_source, 3),
            "rtf": calculer_rtf(temps, duree),
            "chemin_audio": str(chemin),
        })

    avec_francais = [i for i, ref in enumerate(references) if ref["fr"]]
    mesures: Dict[str, Any] = {}
    chemin_csv = dossier_sortie / "resultats.csv"
    if avec_francais:
        refs_fr = [references[i] for i in avec_francais]
        hyps_fr = [hypotheses[i] for i in avec_francais]
        lignes = evaluer_sorties_systeme(
            fiches_reference=refs_fr,
            sorties_systeme=hyps_fr,
            systeme=SYSTEME,
            version_modele=modele_id,
            jeu=jeu,
            configuration=configuration,
            chemin_csv=chemin_csv,
            evaluer_par_type=False,
        )
        for ligne in lignes:
            mesures[ligne["mesure"]] = ligne["valeur"]
    else:
        temps_total = sum(float(h["temps_calcul_s"]) for h in hypotheses)
        duree_totale = sum(float(r["duree_s"]) for r in references)
        mesures["RTF"] = calculer_rtf(temps_total, duree_totale)
        mesures["chrF++"] = "non mesuré"
        mesures["BLEU"] = "non mesuré"

    chemin_jsonl = dossier_sortie / "seamless_m4t.jsonl"
    with open(chemin_jsonl, "w", encoding="utf-8") as fichier:
        for ligne in details:
            fichier.write(json.dumps(ligne, ensure_ascii=False) + "\n")
    chemin_ids = dossier_sortie / "seamless_m4t_ids.txt"
    chemin_ids.write_text("\n".join(segment["id"] for segment in segments) + "\n", encoding="utf-8")

    return {
        "jeu": jeu,
        "nb_segments": len(details),
        "nb_avec_reference_fr": len(avec_francais),
        "mesures": mesures,
        "details": details,
        "chemin_csv": str(chemin_csv),
        "chemin_jsonl": str(chemin_jsonl),
        "chemin_ids": str(chemin_ids),
        "lignes_csv": lire_resultats(chemin_csv),
    }


def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(description="Benchmark SeamlessM4T sur data/mgb2")
    parseur.add_argument("--dossier", default="", help="Dossier data/mgb2. Vide : recherche puis clone.")
    parseur.add_argument("--nb", type=int, default=NB_SEGMENTS_DEFAUT)
    parseur.add_argument("--essai", action="store_true", help="Traducteur factice, sans télécharger le modèle.")
    parseur.add_argument("--sans-clone", action="store_true")
    parseur.add_argument("--sortie", default=str(RACINE_PROJET / "resultats"))
    return parseur


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = construire_parseur().parse_args(argv)
    dossier = resoudre_dossier_mgb2(
        RACINE_PROJET,
        dossier=arguments.dossier or None,
        cloner=not arguments.sans_clone,
    )
    print(f"Données : {dossier}")
    choisis, info = selectionner_segments(lister_segments(dossier, split="dev"), nb_max=arguments.nb)
    print(
        f"Segments retenus : {info['retenus']} "
        f"(trop longs écartés pendant la sélection : {info['ignores_longs']}, "
        f"sans audio : {info['sans_audio']})"
    )
    if not choisis:
        raise SystemExit("Aucun segment audio de 30 secondes ou moins.")
    choisis = attacher_extraits(choisis, RACINE_PROJET / "donnees" / "externe" / "cache_mgb2")
    if arguments.essai:
        def traduire(audio: Any, taux: int) -> Tuple[str, float]:
            return "sous-titre factice", 0.01
    else:
        print(f"Chargement de {MODELE_ID}")
        traduire = fabriquer_traducteur(MODELE_ID)
    bilan = executer_benchmark(choisis, traduire, Path(arguments.sortie))
    print(json.dumps({cle: bilan[cle] for cle in ("nb_segments", "nb_avec_reference_fr", "mesures", "chemin_ids")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
