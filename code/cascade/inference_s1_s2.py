"""Inférence et mode différé pour les systèmes S1 (sans LoRA) et S2 (avec LoRA) (sections 2, 4.1 et 5).

Implémente :
- Décodage glouton (température 0, do_sample=False), plafonné à 128 tokens par segment.
- Mode différé :
  1. Découpage par détection de voix (segments <= 30 secondes).
  2. Transcription arabe par le modèle d'écoute retenu (`openai/whisper-large-v3`).
  3. Détection des noms propres présents dans le segment via `donnees/noms/noms_v1.csv`.
  4. Traduction française avec les 2 phrases précédentes déjà traduites en contexte et les noms propres.
  5. Export SubRip (`.srt`) avec horodatages exacts.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import soundfile as sf

RACINE_PROJET = Path(__file__).resolve().parents[2]
DOSSIER_CASCADE = Path(__file__).resolve().parent
DOSSIER_DEMO = RACINE_PROJET / "code" / "demo"

for p in (DOSSIER_CASCADE, DOSSIER_DEMO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from preparation_exemples import (  # noqa: E402
    CHEMIN_CONFIG_S2,
    charger_config_s2,
    charger_lexique_noms,
    construire_messages_cascade,
    detecter_noms_propres_segment,
    formater_chat_template_cascade,
)
from srt import generer_contenu_srt, sauvegarder_srt  # noqa: E402
from systeme_s0 import charger_audio_mono_16k, decouper_audio_vad, generer_signal_parole_test  # noqa: E402


class SystemeCascadeS1S2:
    """Chaîne de sous-titrage en cascade : S1 (LLM sans LoRA) ou S2 (LLM avec LoRA)."""

    def __init__(
        self,
        code_systeme: str = "S2",
        config: Optional[Dict[str, Any]] = None,
        chemin_adaptateur_lora: Optional[Union[str, Path]] = None,
        mode_substitution: bool = False,
        transcribe_fn: Optional[Callable[[np.ndarray, int], str]] = None,
        translate_fn: Optional[Callable[[str, Sequence[str], Sequence[Dict[str, str]]], str]] = None,
    ):
        if code_systeme not in ("S1", "S2"):
            raise ValueError(f"code_systeme doit être 'S1' ou 'S2' (reçu : {code_systeme})")
        self.code_systeme = code_systeme
        self.config = config or charger_config_s2()
        self.chemin_adaptateur_lora = Path(chemin_adaptateur_lora) if chemin_adaptateur_lora else None
        self.mode_substitution = mode_substitution
        self._transcribe_fn = transcribe_fn
        self._translate_fn = translate_fn

        chemin_lex = RACINE_PROJET / self.config.get("briques", {}).get("lexique_noms_csv", "donnees/noms/noms_v1.csv")
        self.lexique_noms = charger_lexique_noms(chemin_lex)
        self._asr_proc = None
        self._asr_model = None
        self._llm_tok = None
        self._llm_model = None

    def _charger_modeles_reels(self) -> None:
        if self.mode_substitution or self._asr_model is not None:
            return

        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        asr_id = self.config["briques"]["modele_ecoute_id"]
        self._asr_proc = AutoProcessor.from_pretrained(asr_id)
        self._asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(asr_id, torch_dtype=dtype).to(device).eval()

        llm_id = self.config["briques"]["modele_llm_id"]
        self._llm_tok = AutoTokenizer.from_pretrained(llm_id)
        kwargs_llm: Dict[str, Any] = {"device_map": "auto"}
        if device == "cuda" and self.config.get("entrainement", {}).get("quantification_4bits", True):
            kwargs_llm["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs_llm["torch_dtype"] = torch.float16
        else:
            kwargs_llm["torch_dtype"] = dtype

        base_llm = AutoModelForCausalLM.from_pretrained(llm_id, **kwargs_llm)
        if self.code_systeme == "S2" and self.chemin_adaptateur_lora and self.chemin_adaptateur_lora.exists():
            from peft import PeftModel

            base_llm = PeftModel.from_pretrained(base_llm, str(self.chemin_adaptateur_lora))
        self._llm_model = base_llm.eval()

    def transcrire_segment(self, audio: np.ndarray, sr: int = 16000) -> str:
        """Transcrit un segment audio 16 kHz en texte arabe."""
        if self._transcribe_fn is not None:
            return self._transcribe_fn(audio, sr)
        if self.mode_substitution:
            return "التقى الرئيس الأسبق بيل كلينتون بممثلي الأمم المتحدة في المؤتمر."

        self._charger_modeles_reels()
        import torch

        device = next(self._asr_model.parameters()).device
        inputs = self._asr_proc(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        max_tok = int(self.config.get("inference", {}).get("max_new_tokens", 128))
        with torch.no_grad():
            ids = self._asr_model.generate(**inputs, max_new_tokens=max_tok)
        return self._asr_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def traduire_segment_avec_contexte(
        self,
        texte_arabe: str,
        phrases_precedentes_fr: Sequence[str],
    ) -> Dict[str, Any]:
        """Traduit un segment arabe en injectant les 2 phrases précédentes et les noms propres."""
        noms_trouves = detecter_noms_propres_segment(texte_arabe, self.lexique_noms)
        contexte_2 = [p.strip() for p in phrases_precedentes_fr if p and p.strip()][-2:]

        messages = construire_messages_cascade(
            texte_arabe=texte_arabe,
            traduction_fr=None,
            phrases_precedentes_fr=contexte_2,
            noms_propres=noms_trouves,
            consigne_systeme=self.config.get("consigne_systeme"),
        )
        prompt_chat = formater_chat_template_cascade(
            messages,
            tokenizer=self._llm_tok,
            add_generation_prompt=True,
        )

        if self._translate_fn is not None:
            texte_fr = self._translate_fn(texte_arabe, contexte_2, noms_trouves)
        elif self.mode_substitution:
            noms_str = ", ".join(n["forme_francaise"] for n in noms_trouves)
            if noms_str:
                texte_fr = f"L'ancien président Bill Clinton a rencontré les représentants lors de la conférence ({noms_str})."
            else:
                texte_fr = f"[Traduction {self.code_systeme}] {texte_arabe.strip()}"
        else:
            self._charger_modeles_reels()
            import torch

            inputs = self._llm_tok(prompt_chat, return_tensors="pt").to(self._llm_model.device)
            max_tok = int(self.config.get("inference", {}).get("max_new_tokens", 128))
            with torch.no_grad():
                out = self._llm_model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_tok,
                )
            gen_ids = out[0][inputs["input_ids"].shape[1] :]
            texte_fr = self._llm_tok.decode(gen_ids, skip_special_tokens=True).strip()

        return {
            "texte_fr": texte_fr,
            "noms_propres_detectes": noms_trouves,
            "contexte_utilise": contexte_2,
            "prompt_chat_template": prompt_chat,
        }

    def sous_titrer_fichier_differe(
        self,
        chemin_audio: Union[str, Path],
        chemin_srt_sortie: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Mode différé complet : VAD <= 30 s, contexte glissant de 2 phrases, export SRT."""
        t0 = time.perf_counter()
        audio, sr = charger_audio_mono_16k(chemin_audio, sr_cible=16000)
        duree_audio_s = max(0.001, len(audio) / float(sr))

        cfg_inf = self.config.get("inference", {})
        tranches = decouper_audio_vad(
            audio,
            sr=sr,
            duree_max_s=float(cfg_inf.get("duree_max_segment_s", 30.0)),
            silence_min_s=float(cfg_inf.get("silence_min_coupure_s", 0.5)),
            seuil_energie_db=float(cfg_inf.get("seuil_energie_db", -38.0)),
        )

        historique_fr: List[str] = []
        segments_sortie: List[Dict[str, Any]] = []

        for idx, (deb_s, fin_s, arr) in enumerate(tranches, start=1):
            ar_txt = self.transcrire_segment(arr, sr=sr)
            trad_info = self.traduire_segment_avec_contexte(
                texte_arabe=ar_txt,
                phrases_precedentes_fr=historique_fr[-2:],
            )
            fr_txt = trad_info["texte_fr"]
            if fr_txt:
                historique_fr.append(fr_txt)

            segments_sortie.append(
                {
                    "index": idx,
                    "debut_s": deb_s,
                    "fin_s": fin_s,
                    "texte_ar": ar_txt,
                    "texte_fr": fr_txt,
                    "contexte_utilise": trad_info["contexte_utilise"],
                    "noms_propres_detectes": trad_info["noms_propres_detectes"],
                }
            )

        temps_calcul_s = time.perf_counter() - t0
        contenu_srt = generer_contenu_srt(segments_sortie)
        if chemin_srt_sortie is not None:
            sauvegarder_srt(segments_sortie, chemin_srt_sortie)

        return {
            "systeme": self.code_systeme,
            "duree_audio_s": round(duree_audio_s, 3),
            "temps_calcul_s": round(temps_calcul_s, 4),
            "rtf": round(temps_calcul_s / duree_audio_s, 4),
            "segments": segments_sortie,
            "srt": contenu_srt,
            "chemin_srt": str(chemin_srt_sortie) if chemin_srt_sortie else None,
        }


def executer_mode_essai_inference(dossier_sortie: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Vérifie le mode différé S1/S2 hors-ligne avec contexte glissant et export SRT."""
    cible_dir = Path(dossier_sortie) if dossier_sortie else (RACINE_PROJET / "resultats" / "essai_s2_inference")
    cible_dir.mkdir(parents=True, exist_ok=True)
    chemin_wav = cible_dir / "extrait_essai_s2.wav"
    chemin_srt = cible_dir / "extrait_essai_s2.srt"

    signal = generer_signal_parole_test(duree_s=4.0, sr=16000)
    sf.write(str(chemin_wav), signal, 16000)

    cascade = SystemeCascadeS1S2(code_systeme="S2", mode_substitution=True)
    return cascade.sous_titrer_fichier_differe(chemin_wav, chemin_srt_sortie=chemin_srt)


def construire_parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inférence et mode différé pour la cascade S1 (sans LoRA) et S2 (avec LoRA)."
    )
    parser.add_argument(
        "--essai",
        action="store_true",
        help="Exécute un test du mode différé S2 en substitution hors ligne.",
    )
    parser.add_argument("--systeme", choices=["S1", "S2"], default="S2", help="Système à exécuter (S1 ou S2).")
    parser.add_argument("--audio", type=str, help="Chemin du fichier audio à sous-titrer.")
    parser.add_argument("--sortie-srt", type=str, help="Chemin du fichier .srt de sortie.")
    parser.add_argument("--lora", type=str, help="Chemin du dossier/fichier d'adaptateur LoRA (pour S2).")
    parser.add_argument("--config", type=str, default=str(CHEMIN_CONFIG_S2), help="Chemin de la config YAML.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = construire_parseur()
    args = parser.parse_args(argv)

    if args.essai:
        res = executer_mode_essai_inference()
        print(
            f"[ESSAI INFERENCE {res['systeme']} OK] {len(res['segments'])} segments, "
            f"contexte seg 2={len(res['segments'][-1]['contexte_utilise'])} phrase(s), "
            f"SRT={res['chemin_srt']}"
        )
        return 0

    if not args.audio:
        parser.error("L'argument --audio est requis hors mode --essai.")

    cfg = charger_config_s2(args.config)
    cascade = SystemeCascadeS1S2(
        code_systeme=args.systeme,
        config=cfg,
        chemin_adaptateur_lora=args.lora,
        mode_substitution=False,
    )
    chemin_srt = args.sortie_srt or str(Path(args.audio).with_suffix(".srt"))
    res = cascade.sous_titrer_fichier_differe(args.audio, chemin_srt_sortie=chemin_srt)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
