"""Petits modèles aléatoires pour les tests et les modes --essai (aucun téléchargement).

Repris de ``projecteur_v1/speech_llm/tiny.py`` (dépôt MLP). Le petit
``WhisperEncoder`` y est remplacé par un petit Cohere Transcribe
(``CohereAsrForConditionalGeneration``, encodeur Conformer de type Parakeet),
avec le même préparateur de features que le vrai modèle, à 16 bandes Mel.
"""

from types import SimpleNamespace

import torch

NB_BANDES_MEL = 16
D_ENC = 32
D_LLM = 32
EOS = 1


class TokenizerFactice:
    """Tokenizer au niveau octet : id = octet + 2. 0 = rembourrage, 1 = EOS."""

    eos_token_id = EOS
    pad_token_id = 0

    def __call__(self, texte, add_special_tokens=False):
        return SimpleNamespace(input_ids=[b + 2 for b in texte.encode("utf-8")])

    def decode(self, ids, skip_special_tokens=True):
        octets = bytes(int(i) - 2 for i in ids if 2 <= int(i) < 258)  # ids hors octet ignorés
        return octets.decode("utf-8", errors="replace")


def petit_llm():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(vocab_size=300, hidden_size=D_LLM, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                         eos_token_id=EOS, pad_token_id=0, initializer_range=0.3)
    return Qwen2ForCausalLM(config)


def petit_modele_asr(graine: int = 0):
    """Petit Cohere Transcribe complet (encodeur + décodeur), comme le vrai modèle chargé."""
    from transformers import CohereAsrConfig, CohereAsrForConditionalGeneration

    torch.manual_seed(graine)
    config = CohereAsrConfig(
        encoder_config=dict(hidden_size=D_ENC, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                            num_mel_bins=NB_BANDES_MEL, subsampling_conv_channels=8),
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
        vocab_size=64, max_position_embeddings=64,
    )
    return CohereAsrForConditionalGeneration(config).eval()


def petit_preparateur():
    from transformers import CohereAsrFeatureExtractor

    return CohereAsrFeatureExtractor(feature_size=NB_BANDES_MEL)


def vecteurs_factices(longueurs=(30, 50), d_enc: int = D_ENC, graine: int = 0):
    """Vecteurs d'encodeur précalculés aléatoires : liste de (T_i, d_enc)."""
    generateur = torch.Generator().manual_seed(graine)
    return [torch.randn(n, d_enc, generator=generateur) for n in longueurs]
