import math
import struct
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plots
import render
from engines import Segment, Result, audio_duration, build_engines, split_times, to_srt, to_vtt, write_subtitles


def make_wav(path, seconds=7.0, sr=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(i / 20))) for i in range(int(seconds * sr))))
    return str(path)


def test_split_times_is_contiguous_and_covers_duration():
    parts = split_times(13.0)
    assert parts[0][0] == 0 and abs(parts[-1][1] - 13.0) < 1e-9
    assert all(abs(parts[i][1] - parts[i + 1][0]) < 1e-9 for i in range(len(parts) - 1))


def test_duration_and_demo_engines(tmp_path):
    wav = make_wav(tmp_path / "a.wav", 7.0)
    assert abs(audio_duration(wav) - 7.0) < 0.01
    engines = build_engines()
    res = engines["projector"].run(wav, "st")
    assert res.is_demo and res.fr and not res.ar          # le projecteur ne produit pas d'arabe en traduction
    assert engines["cascade"].run(wav, "st").ar            # la chaîne classique, si
    assert engines["xattn"].run(wav, "asr").ar and not engines["xattn"].run(wav, "asr").fr


def test_srt_and_vtt_format(tmp_path):
    res = Result("x", "X", "st", 8.0, [Segment(0.0, 3.5, fr="Bonjour."), Segment(3.5, 8.0, fr="Merci.")])
    assert to_srt(res).startswith("1\n00:00:00,000 --> 00:00:03,500\nBonjour.")
    assert "00:00:03.500 --> 00:00:08.000" in to_vtt(res) and to_vtt(res).startswith("WEBVTT")
    files = write_subtitles(res, tmp_path)
    assert [Path(f).suffix for f in files] == [".srt", ".vtt"] and all(Path(f).exists() for f in files)


def test_render_escapes_model_text():
    res = Result("x", "X<b>", "st", 5.0, [Segment(0, 5, fr='<script>alert("x")</script> & fin')], is_demo=False)
    html = render.article(res) + render.timeline(res) + render.compare([res], ["X"])
    assert "<script>" not in html and "&lt;script&gt;" in html and "X<b>" not in html


def test_confidence_marks_low_words_only_when_asked():
    res = Result("x", "X", "st", 5.0, [Segment(0, 5, fr="un deux trois")], confidences=[0.4, 0.7, 0.99])
    assert 'class="w low"' in render.article(res, True) and 'class="w mid"' in render.article(res, True)
    assert 'class="w' not in render.article(res, False)


def test_stamp_only_for_demo_and_empty_state():
    assert "stamp" in render.article(Result("x", "X", "st", 1.0, [Segment(0, 1, fr="a")], is_demo=True))
    assert "stamp" not in render.article(Result("x", "X", "st", 1.0, [Segment(0, 1, fr="a")], is_demo=False))
    assert "ghost" in render.empty_state("t", "h")


def test_logs_fall_back_to_labelled_demo(tmp_path):
    train, val, demo = plots.load_logs(tmp_path)
    assert demo and train and val
    (tmp_path / "log.jsonl").write_text('{"step": 1, "loss": 4.0}\n{"step": 2, "loss": 3.5}\n', encoding="utf-8")
    train, val, demo = plots.load_logs(tmp_path)
    assert not demo and len(train) == 2 and val == []
    assert plots.loss_figure(train, val, demo) is not None


def test_press_escapes_filters_links_and_marks_missing_translation():
    from feeds import Item
    from textmt import Translator
    evil = Item("hespress", "article", "<img src=x onerror=alert(1)>", "javascript:alert(1)", "ملخص & نص")
    ok = Item("aljazeera", "article", "عنوان", "https://ex.org/a?x=1&y=2", "")
    data = {"hespress": ([evil], None), "aljazeera": ([ok], None), "echorouk": ([], "OSError: boom")}
    html = render.press(data, Translator(None, None), None, "statut")
    assert "<img" not in html and "&lt;img" in html            # titre du flux échappé
    assert 'href="javascript:' not in html and 'href="#"' in html   # schéma refusé
    assert 'href="https://ex.org/a?x=1&amp;y=2"' in html
    assert "Traduction non disponible" in html and "Flux indisponible (OSError)" in html
    only = render.press(data, Translator(None, None), "Qatar", "statut")
    assert "Al Jazeera" in only and "Hespress" not in only


def test_press_shows_translation_when_known():
    from feeds import Item
    from textmt import Translator
    tr = Translator(lambda ts: [f"FR {t}" for t in ts], None)
    it = Item("hespress", "article", "عنوان", "https://ex.org/a", "ملخص")
    tr.translate(["عنوان", "ملخص"])
    html = render.press({"hespress": ([it], None)}, tr, None, "s")
    assert "FR عنوان" in html and "FR ملخص" in html and 'class="story lead"' in html


def test_video_embed_escapes_id():
    assert "youtube-nocookie.com/embed/abc" in render.video_original("abc")
    assert '"><script' not in render.video_original('"><script>')


def test_refresh_once_queues_translations_and_prefetches_articles(monkeypatch):
    import app
    from feeds import Item
    items = [Item("hespress", "article", f"عنوان {i}", f"https://www.hespress.com/{i}", f"ملخص {i}") for i in range(3)]
    monkeypatch.setattr(app.feeds, "fetch_all", lambda *a, **k: {"hespress": (items, None)})
    queued, fetched = [], []
    monkeypatch.setattr(app.BG, "enqueue", lambda texts: queued.extend(texts))
    monkeypatch.setattr(app.reader, "fetch_article", lambda it: fetched.append(it.uid))
    monkeypatch.setattr(app, "PREFETCH", 2)
    assert app.refresh_once() == 3
    assert queued[:3] == ["عنوان 0", "عنوان 1", "عنوان 2"] and "ملخص 0" in queued    # titres d'abord
    assert fetched == [items[0].uid, items[1].uid]                                   # seulement les 2 premiers


def test_press_says_translation_in_progress_when_translator_is_enabled():
    from feeds import Item
    from textmt import Translator
    it = Item("hespress", "article", "عنوان", "https://www.hespress.com/a", "ملخص")
    html = render.press({"hespress": ([it], None)}, Translator(lambda ts: ts, None), None, "s")
    assert "Traduction en cours" in html and "non disponible" not in html
