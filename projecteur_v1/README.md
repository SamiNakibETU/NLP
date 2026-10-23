# projecteur_v1 : parole arabe vers français, variante « projecteur »

Chaîne : audio -> features Whisper -> encodeur whisper-large-v3 (gelé) -> projecteur -> Qwen2.5-1.5B-Instruct.
Version 1 : étapes A (environnement) et B (test miniature) faites. Rien n'a encore tourné avec le vrai Whisper ni sur GPU.

## Contenu
- `speech_llm/projector.py` : LayerNorm, 2 x Conv1d (noyau 3, pas 2, GELU), MLP 1280 -> 2 x dim_LLM -> dim_LLM (50 Hz -> 12,5 Hz).
- `speech_llm/model.py` : encodeur gelé, assemblage `[préfixe][audio][instruction][cible+EOS]`, perte, génération.
- `speech_llm/training.py` : étape 1 (projecteur seul), étape 2 (projecteur + LoRA r=16, alpha=32).
- `speech_llm/audio.py` : chargement audio (start/end, 30 s max) et features Whisper.
- `speech_llm/tiny.py` : faux modèles pour les tests. `tests/test_miniature.py` : 7 tests.
- `manifest_dev_train.jsonl` (5069 segments) et `manifest_dev_val.jsonl` (773) : audio, start, end, text_ar. Pas de text_fr pour l'instant.
- `colab_etapes_A_B.ipynb` : notebook Colab (GPU T4) ; `docs/colab.md`, `docs/journal.md` : ce qui est exécuté et ce qui reste non vérifié.

## Lancer les tests
    pip install -r requirements.txt
    python -m pytest -q

## Reste à faire
Étapes C (vrai Whisper), D (tokens Qwen), E (projecteur causal), `train.py`, F (contrôle d'entraînement), G (streaming).
Licence des données : MGB2 (QCRI-ALT), recherche non commerciale, non transférable.
