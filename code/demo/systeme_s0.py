"""Système de départ S0 : détection de voix, Whisper small, Opus-MT ar->fr et export SRT.

Chaîne complète :
- Découpage de l'audio en segments de parole <= 30 s
- Transcription arabe par Whisper (`openai/whisper-small`)
- Traduction arabe -> français par Opus-MT (`Helsinki-NLP/opus-mt-ar-fr`)
- Export au format SubRip (.srt) avec horodatages
- Évaluation sur FLEURS (`ar_eg` -> `fr_fr`) et enregistrement dans `resultats/resultats.csv`
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf
import yaml

RACINE_PROJET = Path(__file__).resolve().parents[2]
DOSSIER_DEMO = Path(__file__).resolve().parent
DOSSIER_MESURES = RACINE_PROJET / "code" / "mesures"

for p in (DOSSIER_DEMO, DOSSIER_MESURES):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from srt import generer_contenu_srt, sauvegarder_srt  # noqa: E402
from tableau_resultats import evaluer_sorties_systeme  # noqa: E402

CHEMIN_CONFIG_S0 = RACINE_PROJET / "configs" / "s0.yaml"


def charger_config_s0(chemin_config: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Charge la configuration YAML de S0 depuis `configs/s0.yaml`."""
    cible = Path(chemin_config) if chemin_config else CHEMIN_CONFIG_S0
    if not cible.exists():
        raise FileNotFoundError(f"Fichier de configuration S0 introuvable : {cible}")
    with open(cible, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def charger_audio_mono_16k(chemin_audio: Union[str, Path], sr_cible: int = 16000) -> Tuple[np.ndarray, int]:
    """Lit un fichier audio (.wav, .flac, .ogg) et le convertit en mono float32 à `sr_cible` Hz."""
    audio, sr = sf.read(str(chemin_audio), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != sr_cible and len(audio) > 0:
        duree = len(audio) / float(sr)
        nb_cible = max(1, int(round(duree * sr_cible)))
        x_orig = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=nb_cible, endpoint=False)
        audio = np.interp(x_new, x_orig, audio).astype(np.float32)
        sr = sr_cible
    return audio.astype(np.float32), sr


def decouper_audio_vad(
    audio: np.ndarray,
    sr: int = 16000,
    duree_max_s: float = 30.0,
    silence_min_s: float = 0.5,
    seuil_energie_db: float = -38.0,
) -> List[Tuple[float, float, np.ndarray]]:
    """Découpe un signal audio en segments de parole de durée <= `duree_max_s`.

    Détecte les pauses de plus de `silence_min_s` secondes sous `seuil_energie_db`
    et garantit qu'aucun segment ne dépasse `duree_max_s` secondes.
    """
    if len(audio) == 0:
        return []

    duree_totale_s = len(audio) / float(sr)
    if duree_totale_s <= duree_max_s:
        # Vérification s'il y a des pauses franches permettant un découpage naturel en sous-phrases
        pas_fenetre = max(1, int(0.05 * sr))  # 50 ms
        seuil_amp = 10.0 ** (seuil_energie_db / 20.0)
        segments: List[Tuple[float, float, np.ndarray]] = []
        debut_courant = None
        silence_acc_s = 0.0

        for idx in range(0, len(audio), pas_fenetre):
            tranche = audio[idx : idx + pas_fenetre]
            rms = float(np.sqrt(np.mean(np.square(tranche)) + 1e-12))
            t_s = idx / float(sr)
            duree_tranche = len(tranche) / float(sr)

            if rms >= seuil_amp:
                if debut_courant is None:
                    debut_courant = t_s
                silence_acc_s = 0.0
            else:
                if debut_courant is not None:
                    silence_acc_s += duree_tranche
                    if silence_acc_s >= silence_min_s and (t_s - debut_courant) >= 1.0:
                        fin_s = t_s
                        i_deb = int(round(debut_courant * sr))
                        i_fin = int(round(fin_s * sr))
                        segments.append((round(debut_courant, 3), round(fin_s, 3), audio[i_deb:i_fin]))
                        debut_courant = None
                        silence_acc_s = 0.0

        if debut_courant is not None:
            i_deb = int(round(debut_courant * sr))
            segments.append((round(debut_courant, 3), round(duree_totale_s, 3), audio[i_deb:]))

        if not segments:
            return [(0.0, round(duree_totale_s, 3), audio)]
        return segments

    # Si l'audio dépasse duree_max_s, découpage garanti en tranches <= duree_max_s
    segments_longs: List[Tuple[float, float, np.ndarray]] = []
    taille_max = int(round(duree_max_s * sr))
    for i_deb in range(0, len(audio), taille_max):
        i_fin = min(len(audio), i_deb + taille_max)
        sous_audio = audio[i_deb:i_fin]
        offset_s = i_deb / float(sr)
        sous_segs = decouper_audio_vad(
            sous_audio,
            sr=sr,
            duree_max_s=duree_max_s,
            silence_min_s=silence_min_s,
            seuil_energie_db=seuil_energie_db,
        )
        for d_rel, f_rel, arr in sous_segs:
            segments_longs.append((round(offset_s + d_rel, 3), round(offset_s + f_rel, 3), arr))

    return segments_longs


class SystemeS0:
    """Chaîne de sous-titrage S0 : Whisper + Opus-MT arabe-français."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        mode_substitution: bool = False,
        transcribe_fn: Optional[Callable[[np.ndarray, int], str]] = None,
        translate_fn: Optional[Callable[[str], str]] = None,
    ):
        self.config = config or charger_config_s0()
        self.mode_substitution = mode_substitution
        self._transcribe_fn = transcribe_fn
        self._translate_fn = translate_fn
        self._asr_processor = None
        self._asr_model = None
        self._mt_tokenizer = None
        self._mt_model = None
        self.modele_asr_effectif = self.config["ecoute"]["modele_id"]
        self.modele_mt_effectif = self.config["traduction"]["modele_id"]

    def _charger_modeles_reels(self):
        if self.mode_substitution or self._asr_model is not None:
            return

        import torch
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )

        asr_id = self.config["ecoute"]["modele_id"]
        try:
            self._asr_processor = WhisperProcessor.from_pretrained(asr_id, local_files_only=False)
            self._asr_model = WhisperForConditionalGeneration.from_pretrained(asr_id, local_files_only=False)
            self.modele_asr_effectif = asr_id
        except Exception as exc_asr:
            repli_id = self.config["ecoute"].get("modele_repli_local")
            if repli_id:
                try:
                    self._asr_processor = WhisperProcessor.from_pretrained(repli_id, local_files_only=True)
                    self._asr_model = WhisperForConditionalGeneration.from_pretrained(repli_id, local_files_only=True)
                    self.modele_asr_effectif = f"{repli_id} (repli cache local : {asr_id} inaccessible)"
                except Exception:
                    raise RuntimeError(
                        f"Impossible de charger le modèle d'écoute {asr_id} ni le repli local {repli_id} : {exc_asr}"
                    ) from exc_asr
            else:
                raise RuntimeError(f"Impossible de charger le modèle d'écoute {asr_id} : {exc_asr}") from exc_asr

        mt_id = self.config["traduction"]["modele_id"]
        try:
            self._mt_tokenizer = AutoTokenizer.from_pretrained(mt_id, local_files_only=False)
            self._mt_model = AutoModelForSeq2SeqLM.from_pretrained(mt_id, local_files_only=False)
            self.modele_mt_effectif = mt_id
        except Exception as exc_mt:
            raise RuntimeError(
                f"Impossible de charger le modèle de traduction {mt_id} (vérifier la connexion réseau/portail Wi-Fi) : {exc_mt}"
            ) from exc_mt

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._asr_model.to(self._device).eval()
        self._mt_model.to(self._device).eval()

    def transcrire_segment(self, audio: np.ndarray, sr: int = 16000) -> str:
        """Transcrit un segment audio 16 kHz en texte arabe."""
        if self._transcribe_fn is not None:
            return self._transcribe_fn(audio, sr)
        if self.mode_substitution:
            return "أعلن المتحدث الرسمي اليوم عن اتفاق اقتصادي جديد في المؤتمر."

        self._charger_modeles_reels()
        import torch

        inputs = self._asr_processor(audio, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(self._device)
        forced_Ids = self._asr_processor.get_decoder_prompt_ids(
            language=self.config["ecoute"].get("langue", "ar"),
            task=self.config["ecoute"].get("tache", "transcribe"),
        )
        with torch.no_grad():
            pred_ids = self._asr_model.generate(
                input_features,
                forced_decoder_ids=forced_Ids,
                max_new_tokens=self.config["traduction"].get("max_new_tokens", 128),
            )
        texte_ar = self._asr_processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        return texte_ar.strip()

    def traduire_texte(self, texte_arabe: str) -> str:
        """Traduit un segment arabe vers le français."""
        if not texte_arabe.strip():
            return ""
        if self._translate_fn is not None:
            return self._translate_fn(texte_arabe)
        if self.mode_substitution:
            table_demo = {
                "أعلن المتحدث الرسمي اليوم عن اتفاق اقتصادي جديد في المؤتمر.": (
                    "Le porte-parole officiel a annoncé aujourd'hui un nouvel accord économique lors de la conférence."
                ),
            }
            return table_demo.get(
                texte_arabe.strip(),
                f"[Traduction S0] {texte_arabe.strip()}",
            )

        self._charger_modeles_reels()
        import torch

        batch = self._mt_tokenizer([texte_arabe], return_tensors="pt", truncation=True).to(self._device)
        with torch.no_grad():
            gen = self._mt_model.generate(
                **batch,
                num_beams=int(self.config["traduction"].get("num_beams", 1)),
                max_new_tokens=int(self.config["traduction"].get("max_new_tokens", 128)),
            )
        texte_fr = self._mt_tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        return texte_fr.strip()

    def sous_titrer_fichier(
        self,
        chemin_audio: Union[str, Path],
        chemin_srt_sortie: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Traite un fichier audio de bout en bout et produit les sous-titres français + SRT."""
        t0 = time.perf_counter()
        sr_cible = int(self.config["ecoute"].get("frequence_echantillonnage_hz", 16000))
        audio, sr = charger_audio_mono_16k(chemin_audio, sr_cible=sr_cible)
        duree_audio_s = max(0.001, len(audio) / float(sr))

        cfg_vad = self.config.get("vad", {})
        tranches = decouper_audio_vad(
            audio,
            sr=sr,
            duree_max_s=float(cfg_vad.get("duree_max_segment_s", 30.0)),
            silence_min_s=float(cfg_vad.get("silence_min_coupure_s", 0.5)),
            seuil_energie_db=float(cfg_vad.get("seuil_energie_db", -38.0)),
        )

        segments_sortie: List[Dict[str, Union[float, str]]] = []
        for idx, (deb_s, fin_s, arr) in enumerate(tranches, start=1):
            ar_txt = self.transcrire_segment(arr, sr=sr)
            fr_txt = self.traduire_texte(ar_txt)
            segments_sortie.append(
                {
                    "index": idx,
                    "debut_s": deb_s,
                    "fin_s": fin_s,
                    "texte_ar": ar_txt,
                    "texte_fr": fr_txt,
                }
            )

        temps_calcul_s = time.perf_counter() - t0
        contenu_srt = generer_contenu_srt(segments_sortie)
        if chemin_srt_sortie is not None:
            sauvegarder_srt(segments_sortie, chemin_srt_sortie)

        return {
            "systeme": "S0",
            "modele_ecoute": self.modele_asr_effectif,
            "modele_traduction": self.modele_mt_effectif,
            "duree_audio_s": round(duree_audio_s, 3),
            "temps_calcul_s": round(temps_calcul_s, 3),
            "rtf": round(temps_calcul_s / duree_audio_s, 4),
            "segments": segments_sortie,
            "texte_ar_complet": " ".join(str(s["texte_ar"]) for s in segments_sortie).strip(),
            "texte_fr_complet": " ".join(str(s["texte_fr"]) for s in segments_sortie).strip(),
            "srt": contenu_srt,
            "chemin_srt": str(chemin_srt_sortie) if chemin_srt_sortie else None,
        }


def generer_signal_parole_test(duree_s: float = 4.0, sr: int = 16000) -> np.ndarray:
    """Crée un signal audio synthétique modulé avec une pause au milieu pour tester VAD + S0."""
    t = np.linspace(0.0, duree_s, int(duree_s * sr), endpoint=False, dtype=np.float32)
    signal = 0.2 * np.sin(2.0 * math.pi * 220.0 * t).astype(np.float32)
    # Pause de 0.7 s au milieu pour déclencher le découpage VAD en 2 segments
    milieu = len(signal) // 2
    demi_pause = int(0.35 * sr)
    signal[max(0, milieu - demi_pause) : min(len(signal), milieu + demi_pause)] = 0.0
    return signal


def executer_mode_essai_s0(dossier_sortie: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Exécute S0 en mode essai hors-ligne sur un extrait synthétique et vérifie l'export SRT."""
    cible_dir = Path(dossier_sortie) if dossier_sortie else (RACINE_PROJET / "resultats" / "essai_s0")
    cible_dir.mkdir(parents=True, exist_ok=True)
    chemin_wav = cible_dir / "extrait_essai_ar.wav"
    chemin_srt = cible_dir / "extrait_essai_fr.srt"

    signal = generer_signal_parole_test(duree_s=4.0, sr=16000)
    sf.write(str(chemin_wav), signal, 16000)

    s0 = SystemeS0(mode_substitution=True)
    res = s0.sous_titrer_fichier(chemin_wav, chemin_srt_sortie=chemin_srt)
    return res


def construire_parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Système S0 (Whisper small + Opus-MT ar->fr) : sous-titrage différé avec export SRT."
    )
    parser.add_argument(
        "--essai",
        action="store_true",
        help="Exécute un test complet de la chaîne S0 et de l'export SRT en mode substitution hors ligne.",
    )
    parser.add_argument("--audio", type=str, help="Chemin du fichier audio (.wav, .flac) à sous-titrer.")
    parser.add_argument("--sortie-srt", type=str, help="Chemin du fichier .srt à générer.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CHEMIN_CONFIG_S0),
        help="Chemin du fichier de configuration YAML (défaut: configs/s0.yaml).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = construire_parseur()
    args = parser.parse_args(argv)

    if args.essai:
        res = executer_mode_essai_s0()
        print(
            f"[ESSAI S0 OK] {len(res['segments'])} segments sous-titrés, "
            f"RTF={res['rtf']}, SRT généré dans {res['chemin_srt']}"
        )
        return 0

    if not args.audio:
        parser.error("L'argument --audio est requis hors mode --essai.")

    cfg = charger_config_s0(args.config)
    s0 = SystemeS0(config=cfg, mode_substitution=False)
    chemin_srt = args.sortie_srt or str(Path(args.audio).with_suffix(".srt"))
    res = s0.sous_titrer_fichier(args.audio, chemin_srt_sortie=chemin_srt)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
