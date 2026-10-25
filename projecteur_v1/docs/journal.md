# Journal

## Étape A - environnement (exécutée)
- Machine locale : Intel i5-1335U, 15,6 Go RAM, aucun GPU CUDA (Intel UHD seulement). torch 2.14.1+cpu.
- Versions : transformers 5.20.2, peft 0.21.2, librosa 1.0.0, soundfile 0.14.0, sacrebleu 2.6.0, accelerate 1.15.0.
- Cible d'entraînement : Colab T4 (16 Go, fp16 ; le T4 ne gère pas bf16). NON vérifié : rien n'a encore tourné sur ce GPU.

## Étape B - test miniature (exécutée)
- Code : speech_llm/{projector,audio,model,training,tiny}.py ; tests/test_miniature.py.
- Modèles aléatoires : petit WhisperEncoder (d_model 32, 100 frames mel d'entrée), petit Qwen2 (hidden 32), faux tokenizer octet.
- Résultat : 7 tests passés (pytest, 12,5 s, CPU) :
  longueurs du projecteur (500 frames -> 125 tokens ; 1500 -> 375), rembourrage jeté,
  assemblage [préfixe][audio][instruction][cible+EOS], labels à -100 hors cible+EOS,
  étape 1 : seuls les paramètres du projecteur reçoivent un gradient ; étape 2 : projecteur + LoRA seulement,
  encodeur gelé et en eval, génération à partir de inputs_embeds avec rembourrage à gauche.
- NON vérifié : comportement avec le vrai Whisper-large-v3 (128 mels, 3000 frames) et le vrai tokenizer Qwen
  (gabarit ChatML écrit à la main : à confirmer à l'étape D) ; temps et mémoire sur T4.
