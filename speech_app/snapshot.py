"""Édition figée : tout ce qu'il faut pour que le site s'ouvre sans réseau ni modèle.

Dossier produit :
  edition.json          date, articles et vidéos par journal, avec leur traduction française
  articles/<uid>.json   texte complet (arabe + français) des articles retenus
  videos/<id>.json      sous-titres français produits par le modèle pour une vidéo
  logs/                 log.jsonl et val.jsonl de l'entraînement (facultatif)

Préparation (sur une machine avec NLLB, idéalement Colab GPU) :
    python snapshot.py build --out snapshot --per-source 5 --articles-per-source 2 [--logs dossier]
Les sous-titres de vidéos s'ajoutent ensuite avec `save_video_result` (voir le notebook Colab).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import feeds
import reader
from engines import Result, Segment
from feeds import Item


class FrozenTranslator:
    """Même interface que Translator pour render.press : lecture seule dans l'archive."""
    enabled = False

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def get(self, text: str) -> str | None:
        return self.mapping.get(text)


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class Snapshot:
    def __init__(self, root: Path, meta: dict):
        self.root, self.meta = Path(root), meta
        self.articles_by_source = {k: [Item(**d) for d in v] for k, v in meta["articles"].items()}
        self.videos_by_source = {k: [Item(**d) for d in v] for k, v in meta["videos"].items()}
        self.fr = meta["fr"]                      # texte arabe -> français (titres et résumés)
        self.translator = FrozenTranslator(self.fr)

    @classmethod
    def load(cls, root: str | Path) -> "Snapshot":
        root = Path(root)
        return cls(root, _read(root / "edition.json"))

    @property
    def created(self) -> dt.datetime:
        return dt.datetime.fromisoformat(self.meta["created"])

    @property
    def logs_dir(self) -> Path | None:
        d = self.root / "logs"
        return d if d.exists() else None

    def article(self, uid: str):
        """(Article, titre français, paragraphes français) ou None si le texte complet n'est pas dans l'archive."""
        p = self.root / "articles" / f"{uid}.json"
        if not p.exists():
            return None
        d = _read(p)
        item = next((i for its in self.articles_by_source.values() for i in its if i.uid == uid), None)
        if item is None:
            return None
        return reader.Article(item, d["paragraphs"], d["origin"], d.get("error")), d.get("fr_title"), d.get("fr_paragraphs", [])

    def find_item(self, uid: str) -> Item | None:
        return next((i for its in self.articles_by_source.values() for i in its if i.uid == uid), None)

    def video_result(self, video_id: str) -> Result | None:
        p = self.root / "videos" / f"{video_id}.json"
        if not p.exists():
            return None
        d = _read(p)
        segs = [Segment(s["start"], s["end"], fr=s.get("fr"), ar=s.get("ar")) for s in d["segments"]]
        return Result(engine=d["engine"], label=d["label"], task="st", duration=d["duration"], segments=segs,
                      seconds=d.get("seconds"), warning=d.get("warning"), produces_arabic=d.get("produces_arabic", False), is_demo=False)

    def translated_video_ids(self) -> set[str]:
        d = self.root / "videos"
        return {p.stem for p in d.glob("*.json")} if d.exists() else set()


def build(out_dir: str | Path, translator, per_source: int = 5, articles_per_source: int = 2, videos_per_source: int = 8,
          logs_dir: str | Path | None = None, fulltext: bool = True, getter=feeds.http_get, extractor=reader.extract_page) -> Snapshot:
    """Prépare l'archive. `translator` est un textmt.Translator (désactivé = pas de français, avec avertissement)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    articles, videos, fr = {}, {}, {}
    for src in feeds.SOURCES:
        items, err = feeds.fetch(src, "article", per_source, force=True, getter=getter)
        vids, verr = feeds.fetch(src, "video", videos_per_source, force=True, getter=getter)
        for e in (err, verr):
            if e:
                print(f"{src.id}: {e}", flush=True)
        articles[src.id], videos[src.id] = items, vids
    all_items = [i for its in articles.values() for i in its]
    texts = [i.title for i in all_items] + [i.summary for i in all_items if i.summary]
    if translator.enabled:
        translator.translate(texts)
        fr = {t: translator.get(t) for t in texts if translator.get(t)}
    else:
        print("Traduction désactivée : l'archive n'aura pas de français.", flush=True)

    if fulltext:
        for its in articles.values():
            for it in its[:articles_per_source]:
                art = reader.fetch_article(it, getter=getter, extractor=extractor)
                fr_title = fr.get(it.title)
                fr_paras = [translator.translate_paragraph(p) for p in art.paragraphs] if translator.enabled else []
                _write(out / "articles" / f"{it.uid}.json",
                       {"paragraphs": art.paragraphs, "origin": art.origin, "error": art.error,
                        "fr_title": fr_title, "fr_paragraphs": fr_paras})

    meta = {"created": dt.datetime.now().isoformat(timespec="seconds"),
            "articles": {k: [{**asdict(i), "body": None} for i in v] for k, v in articles.items()},
            "videos": {k: [asdict(i) for i in v] for k, v in videos.items()}, "fr": fr}
    _write(out / "edition.json", meta)
    if logs_dir:
        dest = out / "logs"
        dest.mkdir(exist_ok=True)
        for name in ("log.jsonl", "val.jsonl"):
            if (Path(logs_dir) / name).exists():
                shutil.copy(Path(logs_dir) / name, dest / name)
    return Snapshot(out, meta)


def save_video_result(out_dir: str | Path, item: Item, res: Result) -> Path:
    """Enregistre les sous-titres français d'une vidéo (produits par le modèle) dans l'archive."""
    d = {"video_id": item.video_id, "title": item.title, "source": item.source, "engine": res.engine, "label": res.label,
         "duration": res.duration, "seconds": res.seconds, "warning": res.warning, "produces_arabic": res.produces_arabic,
         "generated": dt.datetime.now().isoformat(timespec="seconds"),
         "segments": [{"start": s.start, "end": s.end, "fr": s.fr, "ar": s.ar} for s in res.segments]}
    path = Path(out_dir) / "videos" / f"{item.video_id}.json"
    _write(path, d)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Prépare une édition figée du Porte-Parole")
    p.add_argument("command", choices=["build"])
    p.add_argument("--out", default="snapshot")
    p.add_argument("--per-source", type=int, default=5)
    p.add_argument("--articles-per-source", type=int, default=2)
    p.add_argument("--logs")
    p.add_argument("--no-fulltext", action="store_true", help="n'embarque aucun texte complet d'article")
    a = p.parse_args()
    from textmt import default_translator
    snap = build(a.out, default_translator(), a.per_source, a.articles_per_source, logs_dir=a.logs, fulltext=not a.no_fulltext)
    n = sum(len(v) for v in snap.articles_by_source.values())
    print(f"édition figée écrite dans {a.out} : {n} articles, {sum(len(v) for v in snap.videos_by_source.values())} vidéos, "
          f"{len(snap.fr)} textes traduits")


if __name__ == "__main__":
    main()
