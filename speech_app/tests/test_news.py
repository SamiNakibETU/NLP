import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import clip
import feeds
import textmt
from model_engine import subtitle_lines, windows

RSS = """<?xml version="1.0"?><rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/"><channel>
<item><title>عنوان &amp; أول</title><link>https://ex.org/a</link><pubDate>Thu, 22 Oct 2026 10:00:00 +0000</pubDate>
<description>&lt;p&gt;نص الملخص الأول. جملة ثانية.&lt;/p&gt;</description><media:thumbnail url="https://ex.org/a.jpg"/></item>
<item><title></title><link>https://ex.org/b</link></item>
<item><title>ثان</title><link>https://ex.org/c</link><description>&lt;img src="https://ex.org/c.png"&gt; نص</description></item>
</channel></rss>""".encode()

ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:yt="http://www.youtube.com/xml/schemas/2015"
xmlns:media="http://search.yahoo.com/mrss/"><entry><yt:videoId>abcdefghijk</yt:videoId><title>فيديو</title>
<published>2026-10-22T09:00:00+00:00</published><media:group><media:thumbnail url="https://i.ytimg.com/x.jpg"/>
<media:description>وصف</media:description></media:group></entry></feed>""".encode()


def test_parse_rss_cleans_and_skips_empty_titles():
    items = feeds.parse_rss(RSS, "hespress")
    assert [i.title for i in items] == ["عنوان & أول", "ثان"]
    assert items[0].summary == "نص الملخص الأول. جملة ثانية." and items[0].image == "https://ex.org/a.jpg"
    assert items[0].published.startswith("2026-10-22T10:00:00") and items[1].image == "https://ex.org/c.png"


def test_parse_youtube():
    (v,) = feeds.parse_youtube(ATOM, "aljazeera")
    assert v.video_id == "abcdefghijk" and v.kind == "video" and v.link.endswith("v=abcdefghijk")


def test_fetch_uses_memory_cache_and_falls_back_on_error(tmp_path, monkeypatch):
    monkeypatch.setattr(feeds, "CACHE_FILE", tmp_path / "feeds.json")
    feeds._mem.clear()
    calls = []
    getter = lambda url: calls.append(url) or RSS
    src = feeds.SOURCES[0]
    assert feeds.fetch(src, getter=getter)[1] is None and feeds.fetch(src, getter=getter)[1] is None
    assert len(calls) == 1                                    # deuxième appel : cache mémoire
    feeds._mem.clear()
    def boom(url): raise OSError("réseau coupé")
    items, err = feeds.fetch(src, force=True, getter=boom)
    assert len(items) == 2 and "réseau coupé" in err          # copie disque servie malgré l'erreur


def test_clean_truncates_at_sentence_end_and_ago():
    t = feeds.clean("<b>" + "كلمة " * 30 + ".</b> " + "أخرى " * 40, 200)
    assert len(t) <= 201
    import datetime as dt
    now = dt.datetime(2026, 10, 22, 12, 0, tzinfo=dt.timezone.utc)
    assert feeds.ago("2026-10-22T09:00:00+00:00", now) == "il y a 3 h" and feeds.ago(None) == ""


def test_translator_caches_and_disabled_mode(tmp_path):
    seen = []
    tr = textmt.Translator(lambda ts: seen.append(list(ts)) or [f"fr:{t}" for t in ts], tmp_path / "t.json")
    assert tr.translate(["a", "b", "a"]) == ["fr:a", "fr:b", "fr:a"] and seen == [["a", "b"]]
    assert tr.translate(["a"]) == ["fr:a"] and len(seen) == 1                       # tout vient du cache
    assert textmt.Translator(None, tmp_path / "t.json").translate(["a", "z"]) == ["fr:a", None]  # relit le disque
    assert json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))


def test_subtitle_lines_and_windows():
    lines = subtitle_lines("un deux trois quatre cinq six sept huit neuf dix", 10.0, 20.0, max_chars=15)
    assert all(len(l) <= 15 for _, _, l in lines) and lines[0][0] == 10.0 and abs(lines[-1][1] - 20.0) < 1e-9
    assert " ".join(l for _, _, l in lines) == "un deux trois quatre cinq six sept huit neuf dix"
    assert windows(60.0, 25.0) == [(0.0, 25.0), (25.0, 50.0), (50.0, 60.0)] and windows(25.2, 25.0) == [(0.0, 25.0)]


def test_clip_rejects_bad_ids_and_reports_tools(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        clip.fetch_clip("x; rm -rf /", tmp_path)
    assert isinstance(clip.missing_tools(), list)


# ---------------------------------------------------------------- texte complet
import reader  # noqa: E402
import render  # noqa: E402


def _item(source="echorouk", link="https://www.echoroukonline.com/a", summary="ملخص", body=None):
    return feeds.Item(source, "article", "عنوان", link, summary, None, None, body=body)


def test_html_to_paragraphs_and_chunking():
    html = "<div><p>فقرة &amp; أولى</p><p>فقرة   ثانية<br>سطر ثالث</p><script>x()</script></div>"
    assert feeds.html_to_paragraphs(html) == ["فقرة & أولى", "فقرة ثانية", "سطر ثالث"]
    text = "جملة أولى طويلة. " * 30 + "جملة بلا نقطة " * 80
    chunks = textmt.chunk_text(text, 350)
    assert max(map(len, chunks)) <= 350 and " ".join(chunks).split() == text.split()


def test_translate_paragraph_joins_chunks_and_disabled_returns_none(tmp_path):
    tr = textmt.Translator(lambda ts: [f"<{t}>" for t in ts], tmp_path / "t.json")
    assert tr.translate_paragraph("أ. ب.") == "<أ. ب.>"
    assert textmt.Translator(None, None).translate_paragraph("نص") is None


def test_fetch_article_prefers_feed_body_then_page_then_summary():
    reader._cache.clear()
    it = _item("hespress", "https://www.hespress.com/a", body=["فقرة من الفلو " * 5] * 3)
    assert reader.fetch_article(it, getter=lambda *a: 1 / 0).origin == "flux"          # aucune requête réseau

    reader._cache.clear()
    page = _item(link="https://www.echoroukonline.com/b")
    art = reader.fetch_article(page, getter=lambda url, t: b"<html/>", extractor=lambda h: ["نص كامل طويل " * 10] * 4)
    assert art.origin == "page" and len(art.paragraphs) == 4

    reader._cache.clear()
    art = reader.fetch_article(_item(link="https://www.echoroukonline.com/c"), getter=lambda *a: 1 / 0)
    assert art.origin == "résumé" and "ZeroDivisionError" in art.error and art.paragraphs == ["ملخص"]


def test_fetch_article_refuses_foreign_domain_and_short_extraction():
    reader._cache.clear()
    called = []
    evil = _item(link="https://evil.example/echoroukonline.com")
    art = reader.fetch_article(evil, getter=lambda *a: called.append(1))
    assert not called and art.origin == "résumé" and "domaine" in art.error
    assert not reader.allowed_link(_item(link="file:///etc/passwd"))
    assert reader.allowed_link(_item("aljazeera", "https://www.aljazeera.net/x"))
    reader._cache.clear()
    short = reader.fetch_article(_item(summary="ملخص طويل " * 20, link="https://www.echoroukonline.com/d"),
                                 getter=lambda *a: b"", extractor=lambda h: ["قصير"])
    assert short.origin == "résumé" and "trop court" in short.error


def test_reader_page_escapes_and_shows_pending():
    reader._cache.clear()
    it = _item(link="https://www.echoroukonline.com/e")
    art = reader.Article(it, ["<b>x</b>", "ثاني"], "page")
    html = render.reader_page(art, "T <i>", [None, "Second"], "pending")
    assert "<b>x</b>" not in html and "&lt;b&gt;x&lt;/b&gt;" in html and "T &lt;i&gt;" in html
    assert "Traduction en cours" in html and "<p>Second</p>" in html and "Les droits appartiennent au journal" in html
    assert "Traduction non disponible" in render.reader_page(art, None, [], "off")


def test_story_titles_are_buttons_with_uid():
    it = _item("hespress", "https://www.hespress.com/a")
    html = render.story(it, "Titre FR", "Résumé FR")
    assert html.count(f'data-open="{it.uid}"') == 3 and "Lire l&#x27;article complet" in html


def test_title_repeated_in_extracted_text_is_dropped():
    reader._cache.clear()
    it = _item(link="https://www.echoroukonline.com/f")
    art = reader.fetch_article(it, getter=lambda *a: b"", extractor=lambda h: ["عنوان"] + ["نص طويل جدا " * 10] * 2)
    assert art.origin == "page" and "عنوان" not in art.paragraphs and len(art.paragraphs) == 2


def test_background_translator_fills_cache_without_blocking(tmp_path):
    import threading, time
    gate = threading.Event()
    def backend(texts):
        gate.wait(5)
        return [f"fr:{t}" for t in texts]
    tr = textmt.Translator(backend, tmp_path / "t.json")
    bg = textmt.BackgroundTranslator(tr, batch=2)
    t0 = time.time()
    bg.enqueue(["a", "b", "c", "a", ""])
    assert time.time() - t0 < 1 and bg.pending == 3 and tr.get("a") is None   # rend la main aussitôt
    gate.set()
    for _ in range(100):
        if bg.pending == 0:
            break
        time.sleep(0.05)
    assert bg.pending == 0 and [tr.get(x) for x in "abc"] == ["fr:a", "fr:b", "fr:c"]


def test_background_translator_survives_backend_errors(tmp_path):
    import time
    def boom(texts): raise RuntimeError("pas de mémoire")
    bg = textmt.BackgroundTranslator(textmt.Translator(boom, None), batch=2)
    bg.enqueue(["a", "b"])
    for _ in range(100):
        if bg.pending == 0:
            break
        time.sleep(0.05)
    assert bg.pending == 0 and bg.failed == {"a", "b"}
    bg.enqueue(["a"])
    assert bg.pending == 0                                   # un texte en échec n'est pas relancé en boucle


def test_wordpress_footer_is_removed_from_summary_and_body():
    raw = "<p>نص المقال الأول.</p><p>The post عنوان appeared first on Hespress - جريدة إلكترونية مغربية.</p>"
    assert feeds.clean(raw) == "نص المقال الأول."
    assert feeds.html_to_paragraphs(raw) == ["نص المقال الأول."]
    assert feeds.clean("ملخص قصير The post X appeared first on Y") == "ملخص قصير"
