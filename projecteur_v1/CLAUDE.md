# Projet speech_llm
Parole arabe (MSA) -> texte français, sans transcription à l'inférence.
Chaîne : audio -> features Whisper -> encodeur whisper-large-v3 (gelé) -> projecteur -> Qwen2.5-1.5B-Instruct.
Seule la variante « projecteur » est faite ; la cross-attention viendra plus tard.
Règles : répondre en français ; ne jamais présenter un chiffre comme vérifié s'il ne l'est pas ;
noter tout dans docs/journal.md ; demander avant un changement d'architecture ; ne rien supprimer sans prévenir.
Tests : `python -m pytest -q` (venv : H:\mgb2\venv_mt, CPU). Entraînement prévu sur Colab T4 (fp16, pas de bf16).
