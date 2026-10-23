"""Texte complet d'un article, à la demande (un clic = une requête).

Ordre : (1) le texte que le flux publie déjà (Hespress, An-Nahar) ; (2) sinon la page de l'article, dont on extrait le
texte principal avec trafilatura ; (3) sinon le résumé du flux, signalé comme partiel. Le texte reste en mémoire :
rien n'est écrit sur disque. Les droits appartiennent au journal, chaque lecture renvoie vers sa page.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

import feeds
from feeds import BY_ID, Item

MAX_PARAGRAPHS = 60
MAX_CHARS = 15000


@dataclass
class Article:
    item: Item
    paragraphs: list[str]
    origin: str            # "flux", "page" ou "résumé"
    error: str | None = None


_cache: dict[str, Article] = {}


def allowed_link(item: Item) -> bool:
    """Le lien doit être en http(s) et appartenir au domaine du journal (les flux sont des données non fiables)."""
    u = urlparse(item.link)
    dom = BY_ID[item.source].domain
    host = (u.hostname or "").lower()
    return u.scheme in ("http", "https") and bool(dom) and (host == dom or host.endswith("." + dom))


def trim(paragraphs: list[str], title: str = "") -> list[str]:
    out, total = [], 0
    for p in paragraphs:
        if p.strip() == title.strip():  # les extracteurs reprennent souvent le titre en tête de texte
            continue
        if len(out) >= MAX_PARAGRAPHS or total + len(p) > MAX_CHARS:
            break
        out.append(p)
        total += len(p)
    return out


def extract_page(html: bytes | str) -> list[str]:
    """Texte principal d'une page d'article (trafilatura), en paragraphes."""
    import trafilatura
    text = trafilatura.extract(html, include_comments=False, include_tables=False, favor_recall=True,
                               output_format="txt")
    return [l.strip() for l in (text or "").split("\n") if len(l.strip()) >= 2]


def fetch_article(item: Item, getter=feeds.http_get, extractor=extract_page) -> Article:
    if item.uid in _cache:
        return _cache[item.uid]
    summary = [item.summary] if item.summary else [item.title]
    if item.body:
        art = Article(item, trim(item.body, item.title), "flux")
    elif not allowed_link(item):
        art = Article(item, summary, "résumé", "lien hors du domaine du journal")
    else:
        try:
            paras = extractor(getter(item.link, 20))
            art = Article(item, trim(paras, item.title), "page") if paras else Article(item, summary, "résumé", "texte introuvable dans la page")
        except Exception as exc:  # site bloqué, page absente, trafilatura manquant
            art = Article(item, summary, "résumé", f"{type(exc).__name__}: {exc}")
    # un texte "complet" plus court que le résumé du flux n'en est pas un
    if art.origin != "résumé" and sum(len(p) + 1 for p in art.paragraphs) + 2 < len(item.summary):  # +1 : l'espace entre paragraphes
        art = Article(item, summary, "résumé", "texte extrait trop court")
    _cache[item.uid] = art
    return art
