"""Traduction de TEXTE arabe -> français (titres et résumés d'articles) avec NLLB-200.

Ce n'est pas le modèle de parole : il traite de l'audio, pas du texte écrit. Les traductions sont mises en cache
sur disque (clé = empreinte du texte), donc un titre n'est traduit qu'une fois.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path

CACHE_FILE = Path(__file__).parent / ".cache" / "translations.json"
MODEL = os.environ.get("PORTEPAROLE_NLLB", "facebook/nllb-200-distilled-600M")


def chunk_text(text: str, max_chars: int = 350) -> list[str]:
    """Découpe un paragraphe en morceaux d'au plus `max_chars` caractères, à une fin de phrase si possible
    (NLLB tronque au-delà de 200 tokens). Aucun mot n'est perdu."""
    parts, cur = [], ""
    for sent in re.findall(r"[^.؟?!…]+[.؟?!…]*\s*", text) or [text]:
        sent = sent.strip()
        if not sent:
            continue
        while len(sent) > max_chars:                       # phrase trop longue : coupe à la dernière virgule/espace
            cut = max(sent.rfind(c, 0, max_chars) for c in "،, ")
            cut = cut if cut > max_chars // 2 else max_chars
            head, sent = sent[:cut + 1].strip(), sent[cut + 1:].strip()
            if cur:
                parts.append(cur); cur = ""
            parts.append(head)
        if cur and len(cur) + 1 + len(sent) > max_chars:
            parts.append(cur); cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    return parts


def key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class Translator:
    """Cache + traduction paresseuse. `backend(list[str]) -> list[str]` fait le vrai travail (None = désactivé)."""

    def __init__(self, backend=None, cache_file: Path | None = CACHE_FILE):
        self.backend, self.cache_file = backend, cache_file
        self._lock = threading.Lock()  # un seul calcul à la fois (le modèle est partagé entre fils)
        self.cache: dict[str, str] = {}
        if cache_file and cache_file.exists():
            try:
                self.cache = json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                pass

    @property
    def enabled(self) -> bool:
        return self.backend is not None

    def get(self, text: str) -> str | None:
        return self.cache.get(key(text))

    def translate(self, texts: list[str]) -> list[str | None]:
        """Traduit ce qui manque dans le cache ; renvoie None pour les textes non traduits (désactivé)."""
        with self._lock:
            todo = list(dict.fromkeys(t for t in texts if t and self.get(t) is None))
            if todo and self.backend:
                for t, fr in zip(todo, self.backend(todo)):
                    self.cache[key(t)] = fr
                self._save()
            return [self.get(t) if t else "" for t in texts]

    def translate_paragraph(self, text: str) -> str | None:
        """Traduit un paragraphe long en le découpant en phrases ; None si le traducteur est désactivé."""
        if not text:
            return ""
        out = self.translate(chunk_text(text))
        return None if any(o is None for o in out) else " ".join(out)

    def _save(self) -> None:
        if not self.cache_file:
            return
        try:
            self.cache_file.parent.mkdir(exist_ok=True)
            self.cache_file.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass


def nllb_backend(model_name: str = MODEL, batch: int = 8, beams: int = 4, device: str | None = None):
    """Charge NLLB et renvoie une fonction de traduction par lots (ordre préservé, textes triés par longueur)."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name, src_lang="arb_Arab")
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=torch.float16 if device == "cuda" else torch.float32)
    except TypeError:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model = model.to(device).eval()
    fr = tok.convert_tokens_to_ids("fra_Latn")

    def run(texts: list[str]) -> list[str]:
        out = [""] * len(texts)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for k in range(0, len(order), batch):
            idx = order[k:k + batch]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=200).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, forced_bos_token_id=fr, num_beams=beams, max_new_tokens=220)
            for i, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
                out[i] = t.strip()
        return out

    return run


class BackgroundTranslator:
    """Traduit en arrière-plan, par petits lots, pour que l'interface ne bloque jamais : elle affiche ce qui est déjà
    en cache et se met à jour quand de nouvelles traductions arrivent."""

    def __init__(self, translator: Translator, batch: int = 4):
        self.tr, self.batch = translator, batch
        self.queue: list[str] = []
        self.failed: set[str] = set()
        self.running = False
        self.inflight = 0
        self._lock = threading.Lock()

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self.queue) + self.inflight

    def enqueue(self, texts: list[str]) -> None:
        if not self.tr.enabled:
            return
        with self._lock:
            known = set(self.queue) | self.failed
            self.queue += [t for t in dict.fromkeys(texts) if t and t not in known and self.tr.get(t) is None]
            if self.queue and not self.running:
                self.running = True
                threading.Thread(target=self._work, daemon=True).start()

    def _work(self) -> None:
        while True:
            with self._lock:
                if not self.queue:
                    self.running = False
                    return
                batch, self.queue = self.queue[:self.batch], self.queue[self.batch:]
                self.inflight = len(batch)
            try:
                self.tr.translate(batch)
            except Exception as exc:  # modèle absent, mémoire saturée : on n'insiste pas sur ces textes
                print(f"traduction en arrière-plan impossible : {type(exc).__name__}: {exc}", flush=True)
                with self._lock:
                    self.failed.update(batch)
            finally:
                with self._lock:
                    self.inflight = 0


def default_translator() -> Translator:
    """NLLB si PORTEPAROLE_MT != "off" et que transformers + torch sont installés ; sinon traducteur désactivé."""
    if os.environ.get("PORTEPAROLE_MT", "nllb") == "off":
        return Translator(None)
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return Translator(None)
    holder = {}

    def lazy(texts):
        if "run" not in holder:
            holder["run"] = nllb_backend()
        return holder["run"](texts)

    return Translator(lazy)
