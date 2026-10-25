import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import feeds
import reader
import snapshot
import textmt
from engines import Result, Segment
from test_news import ATOM


def rss_for(url):
    host = urlparse(url).hostname
    items = "".join(f"<item><title>عنوان {i}</title><link>https://{host}/a{i}</link><description>&lt;p&gt;فقرة طويلة {i} "
                    f"{'نص ' * 40}&lt;/p&gt;&lt;p&gt;فقرة ثانية&lt;/p&gt;</description></item>" for i in range(3))
    return f'<?xml version="1.0"?><rss version="2.0"><channel>{items}</channel></rss>'.encode()


def getter(url, timeout=15):
    return ATOM if "youtube.com" in url else rss_for(url)


def make(tmp_path, monkeypatch, translator=None, **kw):
    monkeypatch.setattr(feeds, "CACHE_FILE", tmp_path / "feeds.json")
    feeds._mem.clear(); reader._cache.clear()
    tr = translator or textmt.Translator(lambda ts: [f"FR[{t[:12]}]" for t in ts], None)
    return snapshot.build(tmp_path / "snap", tr, per_source=3, articles_per_source=1, getter=getter,
                          extractor=lambda h: ["نص من الصفحة " * 5] * 2, **kw), tr


def test_build_then_load_roundtrip(tmp_path, monkeypatch):
    snap, _ = make(tmp_path, monkeypatch)
    again = snapshot.Snapshot.load(tmp_path / "snap")
    assert set(again.articles_by_source) == {s.id for s in feeds.SOURCES} and len(again.articles_by_source["hespress"]) == 3
    it = again.articles_by_source["hespress"][0]
    assert again.translator.get(it.title).startswith("FR[") and again.translator.get("غير موجود") is None
    art, fr_title, fr_paras = again.article(it.uid)
    assert art.origin == "flux" and len(fr_paras) == len(art.paragraphs) and fr_title.startswith("FR[")
    assert again.article(again.articles_by_source["hespress"][1].uid) is None      # seuls 1 article/journal en texte complet
    assert again.videos_by_source["aljazeera"][0].video_id == "abcdefghijk"


def test_snapshot_without_translation_or_fulltext(tmp_path, monkeypatch):
    snap, _ = make(tmp_path, monkeypatch, translator=textmt.Translator(None, None), fulltext=False)
    assert snap.fr == {} and not (tmp_path / "snap" / "articles").exists()


def test_video_result_roundtrip(tmp_path, monkeypatch):
    snap, _ = make(tmp_path, monkeypatch)
    vid = snap.videos_by_source["aljazeera"][0]
    res = Result("proj", "Projecteur", "st", 30.0, [Segment(0, 5, fr="Bonjour", ar=None), Segment(5, 30, fr="Merci")], seconds=4.2)
    snapshot.save_video_result(tmp_path / "snap", vid, res)
    back = snapshot.Snapshot.load(tmp_path / "snap")
    assert back.translated_video_ids() == {"abcdefghijk"}
    got = back.video_result("abcdefghijk")
    assert got.label == "Projecteur" and [s.fr for s in got.segments] == ["Bonjour", "Merci"] and not got.is_demo
    assert back.video_result("zzzzzzzzzzz") is None


def test_logs_are_copied(tmp_path, monkeypatch):
    logs = tmp_path / "logs_in"; logs.mkdir()
    (logs / "log.jsonl").write_text('{"step": 1, "loss": 4.0}\n', encoding="utf-8")
    snap, _ = make(tmp_path, monkeypatch, logs_dir=logs)
    assert (snap.logs_dir / "log.jsonl").exists() and not (snap.logs_dir / "val.jsonl").exists()


import json
import subprocess
import textwrap

APP_DIR = Path(__file__).resolve().parent.parent
PY = sys.executable


def run_py(code, cwd, env_extra):
    import os
    env = {**os.environ, **env_extra, "PYTHONIOENCODING": "utf-8", "PORTEPAROLE_MT": "off"}
    out = subprocess.run([PY, "-I", "-c", textwrap.dedent(code)], cwd=cwd, env=env, capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-1500:]
    return out.stdout


def test_frozen_app_serves_press_article_and_video_without_network(tmp_path, monkeypatch):
    snap, _ = make(tmp_path, monkeypatch)
    vid = snap.videos_by_source["aljazeera"][0]
    snapshot.save_video_result(tmp_path / "snap", vid, Result("proj", "Projecteur", "st", 20.0,
                               [Segment(0, 10, fr="Bonsoir."), Segment(10, 20, fr="Merci.")]))
    art_item = snap.articles_by_source["hespress"][0]
    other = snap.articles_by_source["hespress"][1]
    code = f"""
        import sys, json
        sys.path.insert(0, {str(APP_DIR)!r})
        import app, render
        assert app.FROZEN and "Édition figée" in app.MODE
        html = app.on_press("Tous")
        assert "FR[" in html and "Édition figée" in html and "Actualiser" not in html
        assert "FR[" in app._frozen_article({art_item.uid!r}) and "Les droits appartiennent" in app._frozen_article({art_item.uid!r})
        assert "texte complet non inclus" in app._frozen_article({other.uid!r})
        gallery, items, *_ = app.on_channel("Qatar")
        assert gallery[0][1].startswith("[traduite] "), gallery
        vout, vtl, files = app.on_translate(items[0], None, 60)
        assert "Bonsoir." in vtl and "data-seek" in vtl and len(files) == 2
        other_vid = app.feeds.Item("hespress", "video", "t", "https://www.youtube.com/watch?v=zzzzzzzzzzz", video_id="zzzzzzzzzzz")
        _, notice, _ = app.on_translate(other_vid, None, 60)
        assert "non traduite" in notice
        assert "torch" not in sys.modules
        app.build_app()
        print("ok")
    """
    assert run_py(code, tmp_path, {"PORTEPAROLE_SNAPSHOT": str(tmp_path / "snap")}).strip().endswith("ok")


def test_space_folder_is_self_contained(tmp_path, monkeypatch):
    import make_space
    make(tmp_path, monkeypatch)
    out = make_space.make_space(tmp_path / "snap", tmp_path / "space")
    assert (out / "app.py").exists() and (out / "porteparole.py").exists() and (out / "snapshot" / "edition.json").exists()
    assert out.joinpath("README.md").read_text(encoding="utf-8").startswith("---\ntitle:")
    assert "torch" not in out.joinpath("requirements.txt").read_text(encoding="utf-8")
    code = """
        import sys, os
        sys.path.insert(0, os.getcwd())
        os.environ["PORTEPAROLE_SNAPSHOT"] = os.path.join(os.getcwd(), "snapshot")
        import porteparole
        assert porteparole.FROZEN and "torch" not in sys.modules
        porteparole.build_app()
        print("ok")
    """
    assert run_py(code, out, {}).strip().endswith("ok")
