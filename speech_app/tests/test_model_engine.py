"""Moteur réel avec de faux modèles minuscules (nécessite torch + transformers, et le code de speech_xattn)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
CODE = Path(__file__).resolve().parents[2] / "speech_xattn"
if not CODE.exists():
    pytest.skip("dossier speech_xattn introuvable", allow_module_level=True)
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_engine import SpeechLLMEngine  # noqa: E402
from speech_llm.model import SpeechLLM  # noqa: E402
from speech_llm.tiny import FakeTokenizer, tiny_encoder, tiny_llm  # noqa: E402
from speech_llm.xattn import SpeechLLMXAttn  # noqa: E402


class FakeFE:
    def __call__(self, wavs, sampling_rate, return_tensors):
        return SimpleNamespace(input_features=torch.randn(len(wavs), 16, 100))


def engine(model):
    torch.manual_seed(0)
    return SpeechLLMEngine("t", "T", "", model, FakeFE(), window=0.5, batch=2, max_new_tokens=6,
                           loader=lambda p: torch.randn(int(2.2 * 16000)).numpy())


@pytest.mark.parametrize("make", [
    lambda: SpeechLLM(tiny_encoder(), tiny_llm(), FakeTokenizer()),
    lambda: SpeechLLMXAttn(tiny_encoder(), tiny_llm(), FakeTokenizer(), every=1),
])
def test_engine_runs_windows_and_reports_timing(make):
    calls = []
    res = engine(make()).run("x.wav", "st", progress=lambda f, desc="": calls.append(round(f, 2)))
    assert abs(res.duration - 2.2) < 1e-6 and not res.is_demo and res.audio_tokens == 27
    assert calls == [0.5, 1.0]                                  # 4 fenêtres de 0,5 s, par lots de 2
    assert all(0 <= s.start <= s.end <= 2.2 + 1e-9 for s in res.segments)
    assert all(s.ar is None for s in res.segments)               # tâche st : français seulement
    asr = engine(make()).run("x.wav", "asr")
    assert all(s.fr is None for s in asr.segments)
