"""Presse et vidéos en direct : flux RSS des journaux et flux Atom des chaînes YouTube officielles.

On ne garde que ce que les flux publient eux-mêmes (titre, résumé, lien, vignette). Pas de récupération du texte
complet des articles : chaque entrée renvoie vers la page du journal.
"""
from __future__ import annotations

import datetime as dt
import email.utils
import hashlib
import html
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path

UA = "Mozilla/5.0 (compatible; PorteParole/0.1; recherche non commerciale)"
TTL = 600  # secondes avant de redemander un flux
CACHE_FILE = Path(__file__).parent / ".cache" / "feeds.json"

NS = {"media": "http://search.yahoo.com/mrss/", "yt": "http://www.youtube.com/xml/schemas/2015",
      "atom": "http://www.w3.org/2005/Atom", "content": "http://purl.org/rss/1.0/modules/content/",
      "dc": "http://purl.org/dc/elements/1.1/"}


@dataclass(frozen=True)
class Source:
    id: str
    name_ar: str
    name_fr: str
    country: str
    rss: str
    channel_id: str  # chaîne YouTube officielle
    domain: str = ""        # les liens d'articles doivent appartenir à ce domaine
    feed_full: bool = False  # le flux contient-il le texte complet de l'article ?


SOURCES = [
    Source("hespress", "هسبريس", "Hespress", "Maroc", "https://www.hespress.com/feed", "UCvyivffb2WwNKonKiLwVAEQ", "hespress.com", True),
    Source("echorouk", "الشروق", "Echorouk", "Algérie", "https://www.echoroukonline.com/feed", "UCR8l6OQCNzsAA26AEcCKcKw", "echoroukonline.com", False),
    Source("annahar", "النهار", "An-Nahar", "Liban", "https://www.annahar.com/rss", "UCnviqxeOMEZttcEwm9d410w", "annahar.com", True),
    Source("aawsat", "الشرق الأوسط", "Asharq Al-Awsat", "Arabie saoudite", "https://aawsat.com/feed", "UCRJUVYt9gjg8MnlGacySUdg", "aawsat.com", False),
    Source("aljazeera", "الجزيرة", "Al Jazeera", "Qatar", "https://www.aljazeera.net/rss", "UCfiwzLy-8yKzIbsmZTzxDgw", "aljazeera.net", False),
]
BY_ID = {s.id: s for s in SOURCES}


@dataclass
class Item:
    source: str
    kind: str                 # "article" ou "video"
    title: str
    link: str
    summary: str = ""
    published: str | None = None   # ISO 8601
    image: str | None = None
    video_id: str | None = None
    body: list[str] | None = None  # texte complet fourni par le flux (gardé en mémoire seulement)

    @property
    def uid(self) -> str:
        return hashlib.sha1(self.link.encode("utf-8")).hexdigest()[:12]


# pied de page ajouté par WordPress à chaque résumé : « The post <titre> appeared first on <site> ... »
BOILERPLATE = re.compile(r"\s*The post\b.*?\bappeared first on\b.*$", re.I | re.S)


def clean(text: str | None, limit: int | None = None) -> str:
    """Texte brut : balises retirées, entités décodées, espaces réduits, coupé à une fin de phrase si trop long."""
    t = re.sub(r"<[^>]+>", " ", html.unescape(text or ""))
    t = re.sub(r"\s+", " ", html.unescape(t)).strip()
    t = BOILERPLATE.sub("", t).strip()
    if limit and len(t) > limit:
        cut = t[:limit]
        end = max(cut.rfind(c) for c in ".؟?!…،")
        t = cut[: end + 1] if end > limit * 0.5 else cut.rsplit(" ", 1)[0] + "…"
    return t


def html_to_paragraphs(raw: str | None) -> list[str]:
    """Paragraphes de texte brut à partir d'un fragment HTML (sauts de paragraphe, <br>, titres, listes)."""
    t = html.unescape(raw or "")
    t = re.sub(r"(?is)<(script|style)[^>]*>.*?</(script|style)>", " ", t)
    t = re.sub(r"(?i)</(p|div|h[1-6]|li|blockquote)>|<br\s*/?>", "\n", t)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    out = []
    for line in t.split("\n"):
        line = BOILERPLATE.sub("", re.sub(r"\s+", " ", line)).strip()
        if len(line) >= 2 and (not out or out[-1] != line):
            out.append(line)
    return out


def parse_date(s: str | None) -> str | None:
    if not s:
        return None
    for fn in (lambda x: email.utils.parsedate_to_datetime(x), lambda x: dt.datetime.fromisoformat(x.replace("Z", "+00:00"))):
        try:
            d = fn(s.strip())
            if d.tzinfo is None:
                d = d.replace(tzinfo=dt.timezone.utc)
            return d.astimezone(dt.timezone.utc).isoformat()
        except Exception:
            continue
    return None


def _text(el, path):
    found = el.find(path, NS)
    return found.text if found is not None and found.text else ""


def parse_rss(xml: bytes, source: str, limit: int = 8) -> list[Item]:
    root = ET.fromstring(xml)
    items = []
    for it in root.iter("item"):
        title, link = clean(_text(it, "title")), _text(it, "link").strip()
        if not title or not link:
            continue
        thumb = it.find("media:thumbnail", NS) if it.find("media:thumbnail", NS) is not None else it.find("media:content", NS)
        enc = it.find("enclosure")
        image = (thumb.get("url") if thumb is not None else None) or (
            enc.get("url") if enc is not None and (enc.get("type") or "").startswith("image") else None)
        if not image:
            m = re.search(r'<img[^>]+src="([^"]+)"', _text(it, "content:encoded") or _text(it, "description"))
            image = m.group(1) if m else None
        body = None
        if BY_ID[source].feed_full:
            body = html_to_paragraphs(_text(it, "content:encoded") or _text(it, "description")) or None
        items.append(Item(source, "article", title, link, clean(_text(it, "description"), 260),
                          parse_date(_text(it, "pubDate")), image, body=body))
        if len(items) >= limit:
            break
    return items


def parse_youtube(xml: bytes, source: str, limit: int = 8) -> list[Item]:
    root = ET.fromstring(xml)
    items = []
    for e in root.findall("atom:entry", NS):
        vid = _text(e, "yt:videoId")
        if not vid:
            continue
        thumb = e.find("media:group/media:thumbnail", NS)
        items.append(Item(source, "video", clean(_text(e, "atom:title")), f"https://www.youtube.com/watch?v={vid}",
                          clean(_text(e, "media:group/media:description"), 200), parse_date(_text(e, "atom:published")),
                          thumb.get("url") if thumb is not None else f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg", vid))
        if len(items) >= limit:
            break
    return items


def http_get(url: str, timeout: float = 15) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/rss+xml, application/xml, */*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


_mem: dict[str, tuple[float, list[Item]]] = {}


def _load_disk() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_disk(key: str, items: list[Item]) -> None:
    try:
        CACHE_FILE.parent.mkdir(exist_ok=True)
        data = _load_disk()
        data[key] = {"t": time.time(), "items": [{**asdict(i), "body": None} for i in items]}  # pas de texte complet sur disque
        CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def fetch(source: Source, kind: str = "article", limit: int = 8, force: bool = False, getter=http_get) -> tuple[list[Item], str | None]:
    """(entrées, erreur). En cas d'échec réseau, renvoie la dernière copie disque s'il y en a une, avec l'erreur."""
    key = f"{source.id}:{kind}"
    now = time.time()
    if not force and key in _mem and now - _mem[key][0] < TTL:
        return _mem[key][1][:limit], None
    url = source.rss if kind == "article" else f"https://www.youtube.com/feeds/videos.xml?channel_id={source.channel_id}"
    try:
        items = (parse_rss if kind == "article" else parse_youtube)(getter(url), source.id, limit)
        _mem[key] = (now, items)
        _save_disk(key, items)
        return items, None
    except Exception as exc:  # réseau, XML invalide, blocage du site
        old = _load_disk().get(key)
        stale = [Item(**d) for d in old["items"]][:limit] if old else []
        return stale, f"{type(exc).__name__}: {exc}"


def fetch_all(kind: str = "article", limit: int = 4, force: bool = False, getter=http_get) -> dict[str, tuple[list[Item], str | None]]:
    return {s.id: fetch(s, kind, limit, force, getter) for s in SOURCES}


def find(uid: str) -> Item | None:
    """Retrouve une entrée d'article déjà chargée (mémoire, puis copie disque) à partir de son identifiant."""
    for _, items in _mem.values():
        for it in items:
            if it.uid == uid:
                return it
    for entry in _load_disk().values():
        for d in entry.get("items", []):
            it = Item(**d)
            if it.uid == uid:
                return it
    return None


def ago(iso: str | None, now: dt.datetime | None = None) -> str:
    """« il y a 3 h » (ou date courte au-delà de deux jours)."""
    if not iso:
        return ""
    d = dt.datetime.fromisoformat(iso)
    now = now or dt.datetime.now(dt.timezone.utc)
    s = (now - d).total_seconds()
    if s < 90:
        return "à l'instant"
    if s < 3600:
        return f"il y a {int(s // 60)} min"
    if s < 86400:
        return f"il y a {int(s // 3600)} h"
    if s < 2 * 86400:
        return "hier"
    return d.strftime("%d/%m")
