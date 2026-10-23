"""Moteurs du mode direct : le vrai (S3 + Opus-MT, adaptateur LoRA facultatif) et un moteur factice pour les tests.

Un moteur de direct offre trois méthodes (voir code/direct/flux_incremental.py) :
    transcrire(signal) -> str                       arabe, par S3
    traduire_provisoire(ar) -> str                  traduction rapide (texte gris)
    traduire_definitif(ar, contexte) -> str         traduction soignée avec les phrases arabes précédentes (texte blanc)

Le réel garde S3 et Opus-MT chargés ensemble (groupe « direct » du Pool de lazy_engines.py).
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import numpy as np

OPUS_ID = "Helsinki-NLP/opus-mt-ar-fr"
SR = 16000


# ---------------------------------------------------------------- réel
class OpusMT:
    """Helsinki-NLP/opus-mt-ar-fr : traducteur rapide (quelques dizaines de ms) pour le texte provisoire."""

    def __init__(self, model_id: str = OPUS_ID, device: str | None = None):
        import torch
        from transformers import MarianMTModel, MarianTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = MarianTokenizer.from_pretrained(model_id)
        modele = MarianMTModel.from_pretrained(model_id)
        if self.device == "cuda":
            modele = modele.half()
        self.model = modele.to(self.device).eval()

    def traduire(self, ar: str) -> str:
        import torch
        if not ar.strip():
            return ""
        entree = self.tok([ar], return_tensors="pt", truncation=True, max_length=256).to(self.device)
        with torch.inference_mode():
            sortie = self.model.generate(**entree, num_beams=1, max_new_tokens=160)
        return self.tok.decode(sortie[0], skip_special_tokens=True).strip()


class MoteurDirectS3:
    """S3 (transcription arabe et traduction soignée) + Opus-MT (traduction provisoire).

    ``lora_dir`` : adaptateur PEFT facultatif posé sur le Qwen de S3. Il est actif pour TRADUIRE et désactivé
    pour TRANSCRIRE : le projecteur a été entraîné sur le Qwen de base, l'adaptateur le dérouterait.
    """
    nom = "S3 + Opus-MT"

    def __init__(self, transcripteur, opus, lora_dir: str | Path | None = None):
        self.t = transcripteur
        self.opus = opus
        self.lora_actif = False
        if lora_dir:
            self.lora_actif = self._poser_lora(Path(lora_dir))

    def _poser_lora(self, dossier: Path) -> bool:
        if not (dossier / "adapter_config.json").is_file():
            print(f"[live] PORTEPAROLE_LORA={dossier} : pas d'adapter_config.json, adaptateur ignoré", file=sys.stderr)
            return False
        from peft import PeftModel
        self.t.modele.llm = PeftModel.from_pretrained(self.t.modele.llm, str(dossier)).eval()
        print(f"[live] adaptateur LoRA chargé : {dossier}", file=sys.stderr)
        return True

    def _sans_lora(self):
        return self.t.modele.llm.disable_adapter() if self.lora_actif else contextlib.nullcontext()

    def transcrire(self, signal: np.ndarray) -> str:
        with self._sans_lora():
            morceaux = self.t.transcrire_signal(signal)
        return " ".join(m["texte"] for m in morceaux).strip()

    def traduire_provisoire(self, ar: str) -> str:
        return self.opus.traduire(ar)

    def traduire_definitif(self, ar: str, contexte: list[str]) -> str:
        return self.t.traduire_avec_contexte(ar, contexte)


def charger_moteur_direct(nlp_repo: str, s3_ckpt: str | None = None, lora_dir: str | None = None) -> MoteurDirectS3:
    """Charge S3 (HF_TOKEN requis pour l'encodeur Cohere) et Opus-MT."""
    code = Path(nlp_repo) / "code" / "projecteur"
    if not (code / "transcrire_s3.py").is_file():
        raise FileNotFoundError(f"{code}/transcrire_s3.py introuvable : donne la racine du dépôt de l'équipe")
    if str(code) not in sys.path:
        sys.path.append(str(code))
    import transcrire_s3
    transcripteur = transcrire_s3.TranscripteurS3.charger("temps_1_50k", Path(s3_ckpt) if s3_ckpt else None)
    lora_dir = lora_dir if lora_dir is not None else (os.environ.get("PORTEPAROLE_LORA") or None)
    return MoteurDirectS3(transcripteur, OpusMT(), lora_dir)


# ---------------------------------------------------------------- factice
# (mots arabes, glose mot à mot = traduction « rapide », traduction soignée)
PHRASES = [
    ("أهلا بكم في نشرة الأخبار لهذا المساء".split(),
     "bienvenue à vous dans bulletin des nouvelles de ce soir".split(),
     "Bonsoir et bienvenue dans ce journal."),
    ("تواصلت المحادثات بين الوفدين حتى ساعة متأخرة من الليل".split(),
     "continuèrent les pourparlers entre les délégations jusqu'à heure tardive de la nuit".split(),
     "Les pourparlers entre les deux délégations se sont poursuivis tard dans la nuit."),
    ("وأعلن المتحدث الرسمي أن الاجتماع المقبل سيعقد الأسبوع القادم".split(),
     "et annonça le porte-parole officiel que la réunion prochaine sera tenue la semaine prochaine".split(),
     "Le porte-parole officiel a annoncé que la prochaine réunion aurait lieu la semaine prochaine."),
    ("ويرى المراقبون أن الأسواق ستظل حذرة في الأيام المقبلة".split(),
     "et estiment les observateurs que les marchés resteront prudents dans les jours prochains".split(),
     "Les observateurs estiment que les marchés resteront prudents dans les jours à venir."),
]
FREQ_BASE, FREQ_PAS = 300.0, 120.0
MOTS_PAR_SECONDE = 2.0


def signal_phrase(indice: int, duree_s: float, amplitude: float = 0.2) -> np.ndarray:
    """Sinus dont la fréquence identifie la phrase ``indice`` (le faux ASR la reconnaît)."""
    t = np.arange(int(duree_s * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * (FREQ_BASE + FREQ_PAS * indice) * t)).astype(np.float32)


def signal_demo(phrases: list[int], duree_phrase_s: float = 6.0, pause_s: float = 1.2) -> np.ndarray:
    """Phrases factices séparées de pauses silencieuses : de quoi rejouer un journal complet sans modèle."""
    pause = np.zeros(int(pause_s * SR), dtype=np.float32)
    morceaux = [pause]
    for i in phrases:
        morceaux += [signal_phrase(i, duree_phrase_s), pause]
    return np.concatenate(morceaux)


class MoteurFactice:
    """ASR et traducteurs simulés, déterministes. Le faux ASR devine la phrase à la fréquence dominante du
    signal et la lit à MOTS_PAR_SECONDE : une passe partielle donne le début de la phrase, une passe sur le
    segment complet donne toute la phrase. Sert aux tests et à la démonstration sans GPU."""
    nom = "moteur factice (démonstration)"
    lora_actif = False

    def __init__(self, delai_s: float = 0.0):
        self.delai_s = delai_s
        self.appels_definitif: list[tuple[str, list[str]]] = []
        self.nb_transcriptions = 0

    def _phrase(self, signal: np.ndarray):
        coeffs = np.abs(np.fft.rfft(signal[: min(len(signal), SR)]))
        freq = int(np.argmax(coeffs)) * SR / max(1, min(len(signal), SR))
        indice = int(round((freq - FREQ_BASE) / FREQ_PAS)) % len(PHRASES)
        return PHRASES[indice]

    def _attendre(self):
        if self.delai_s:
            import time
            time.sleep(self.delai_s)

    def transcrire(self, signal: np.ndarray) -> str:
        self.nb_transcriptions += 1
        self._attendre()
        if len(signal) < SR // 2 or float(np.abs(signal).max()) < 1e-3:
            return ""
        mots_ar, _, _ = self._phrase(signal)
        n = min(len(mots_ar), max(1, int(len(signal) / SR * MOTS_PAR_SECONDE)))
        return " ".join(mots_ar[:n])

    def _indice(self, ar: str):
        n = len(ar.split())
        for mots_ar, gloses, final in PHRASES:
            if ar.split() == mots_ar[:n]:
                return n, mots_ar, gloses, final
        return n, None, None, None

    def traduire_provisoire(self, ar: str) -> str:
        n, mots_ar, gloses, _ = self._indice(ar)
        return " ".join(gloses[:n]) if gloses else ar

    def traduire_definitif(self, ar: str, contexte: list[str]) -> str:
        self.appels_definitif.append((ar, list(contexte)))
        n, mots_ar, gloses, final = self._indice(ar)
        if final is None:
            return ar
        return final if n == len(mots_ar) else " ".join(gloses[:n]).capitalize() + "…"
