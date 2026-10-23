import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s3_engine import S3Engine  # noqa: E402


class Stub:
    """Transcripteur factice qui honore l'interface de TranscripteurS3 (tests de la logique du moteur)."""
    def __init__(self):
        self.calls = []

    def transcrire_signal(self, signal):
        self.calls.append("transcrire")
        return [{"debut_s": 0.0, "fin_s": 10.0, "texte": "مرحبا بكم في نشرة الأخبار"},
                {"debut_s": 10.0, "fin_s": 20.0, "texte": "تواصلت المحادثات"}]

    def traduire_signal(self, signal):
        self.calls.append("direct")
        return [{"debut_s": 0.0, "fin_s": 10.0, "texte": "Bonsoir et bienvenue dans ce journal"},
                {"debut_s": 10.0, "fin_s": 20.0, "texte": "Les pourparlers se sont poursuivis"}]

    def traduire(self, textes):
        self.calls.append("traduire")
        return ["Bonsoir à tous", ""]


def engine(mode, stub=None):
    return S3Engine("s", "S", mode, stub or Stub(), loader=lambda p: np.zeros(20 * 16000, dtype=np.float32))


def test_direct_gives_french_without_arabic():
    stub = Stub()
    res = engine("direct", stub).run("x.wav", "st")
    assert stub.calls == ["direct"]                                # aucune transcription arabe
    assert res.fr and not res.ar and not res.produces_arabic and not res.is_demo
    assert abs(res.duration - 20.0) < 1e-9 and res.segments[0].start == 0.0 and abs(res.segments[-1].end - 20.0) < 1e-9


def test_cascade_goes_through_arabic_and_keeps_both():
    stub = Stub()
    res = engine("cascade", stub).run("x.wav", "st")
    assert stub.calls == ["transcrire", "traduire"] and res.produces_arabic
    assert "Bonsoir à tous" in res.fr and "مرحبا بكم" in res.ar
    # traduction vide du 2e morceau : l'arabe est gardé plutôt que perdu
    assert any(s.fr is None and s.ar == "تواصلت المحادثات" for s in res.segments)


def test_asr_task_transcribes_for_both_modes():
    for mode in ("direct", "cascade"):
        stub = Stub()
        res = engine(mode, stub).run("x.wav", "asr")
        assert stub.calls == ["transcrire"] and res.ar and not res.fr and res.task == "asr"


def test_progress_is_reported_and_mode_is_validated():
    seen = []
    engine("direct").run("x.wav", "st", progress=lambda f, desc="": seen.append(f))
    assert seen[0] < seen[-1] == 1.0
    with pytest.raises(AssertionError):
        S3Engine("s", "S", "inconnu", Stub())


BENCH = Path(__file__).resolve().parents[2] / "NLP_bench"


@pytest.mark.skipif(not (BENCH / "code" / "projecteur" / "transcrire_s3.py").exists(), reason="dépôt de l'équipe absent")
def test_smoke_with_team_tiny_models(tmp_path):
    """Chaîne réelle de l'équipe avec ses petits modèles aléatoires : on vérifie la plomberie, pas le sens du texte."""
    pytest.importorskip("torch"); pytest.importorskip("transformers"); pytest.importorskip("soundfile")
    import soundfile as sf
    sys.path.append(str(BENCH / "code" / "projecteur"))
    import transcrire_s3
    wav = tmp_path / "bruit.wav"
    sf.write(wav, (0.1 * np.random.default_rng(0).standard_normal(25 * 16000)).astype("float32"), 16000)
    t = transcrire_s3.transcripteur_factice()
    for mode in ("direct", "cascade"):
        res = S3Engine("s", "S", mode, t).run(str(wav), "st")
        assert abs(res.duration - 25.0) < 0.01 and all(0 <= s.start <= s.end <= 25.01 for s in res.segments)


class ArabicCopyStub(Stub):
    """Reproduit ce que leur notebook enregistre : la « traduction directe » du temps 1 recopie l'arabe."""
    def traduire_signal(self, signal):
        return [{"debut_s": 0.0, "fin_s": 10.0, "texte": "كلما زادت الزيادة زاد عملنا"}]


def test_direct_warns_when_the_model_copies_arabic_but_cascade_does_not():
    from s3_engine import arabic_share
    res = engine("direct", ArabicCopyStub()).run("x.wav", "st")
    assert res.warning and "recopié de l'arabe" in res.warning and "100%" in res.warning
    assert engine("direct").run("x.wav", "st").warning is None            # vrai français : pas d'alerte
    assert engine("cascade").run("x.wav", "st").warning is None
    assert engine("direct", ArabicCopyStub()).run("x.wav", "asr").warning is None   # en transcription, l'arabe est voulu
    assert arabic_share("Bonjour tout le monde") == 0.0 and arabic_share("مرحبا") == 1.0 and arabic_share("") == 0.0


def test_warning_is_rendered_escaped_and_survives_the_frozen_edition(tmp_path):
    import render, snapshot
    from engines import Result, Segment
    res = Result("s3", "S3", "st", 10.0, [Segment(0, 10, fr="x")], warning="<b>attention</b> arabe")
    for html in (render.article(res), render.timeline(res), render.compare([res], ["S3"])):
        assert "role=\"alert\"" in html and "<b>attention</b>" not in html and "&lt;b&gt;attention" in html
    from feeds import Item
    snapshot.save_video_result(tmp_path, Item("aljazeera", "video", "t", "https://www.youtube.com/watch?v=abcdefghijk", video_id="abcdefghijk"), res)
    (tmp_path / "edition.json").write_text('{"created": "2026-10-23T12:00:00", "articles": {}, "videos": {}, "fr": {}}', encoding="utf-8")
    assert snapshot.Snapshot.load(tmp_path).video_result("abcdefghijk").warning == "<b>attention</b> arabe"
