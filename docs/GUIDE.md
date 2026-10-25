# Guide d'utilisation

Ce guide permet de lancer Kalam et de reproduire les mesures sans aide extérieure. Les commandes sont à exécuter depuis la racine du dépôt, sauf mention contraire.

## Sommaire

1. [Prérequis](#1-prérequis)
2. [Lancer Kalam sur Google Colab](#2-lancer-kalam-sur-google-colab)
3. [Démonstrations à essayer](#3-démonstrations-à-essayer)
4. [Dépannage](#4-dépannage)
5. [Mode local sans GPU](#5-mode-local-sans-gpu)
6. [Docker](#6-docker)
7. [Reproduire les mesures sur Kaggle](#7-reproduire-les-mesures-sur-kaggle)
8. [Lancer les tests](#8-lancer-les-tests)

## 1. Prérequis

| Élément | Obligatoire | Usage |
|---|---|---|
| Compte Google | oui | exécuter le notebook sur Colab (GPU T4 gratuit) |
| Compte Hugging Face | oui pour S3 et le direct | télécharger les modèles |
| Accès au modèle [`CohereLabs/cohere-transcribe-arabic-07-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026) | oui pour S3 et le direct | encodeur audio de S3, à accès contrôlé |
| Jeton Hugging Face en lecture | oui pour S3 et le direct | secret Colab `HF_TOKEN` |
| Accès à [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) | non | diarisation du rapport ; sans lui, repli sur ECAPA |
| Clé de l'API ScaleDown | non | résumé de fin de session ; sans elle, résumé local |

Préparer le compte Hugging Face :

1. Se connecter sur <https://huggingface.co>, ouvrir la page du modèle Cohere et accepter ses conditions d'accès. L'accès est en général accordé tout de suite ; la page l'indique.
2. Settings > Access Tokens > Create new token, type **Read**. Copier le jeton (`hf_…`).

## 2. Lancer Kalam sur Google Colab

C'est la configuration dans laquelle l'équipe a fait ses démonstrations et ses mesures du direct.

1. Ouvrir le notebook : <https://colab.research.google.com/github/SamiNakibETU/NLP/blob/main/speech_app/colab_live.ipynb>.
2. Exécution > Modifier le type d'exécution > **GPU T4** > Enregistrer.
3. Ouvrir le panneau des secrets (icône de clé, barre de gauche) et ajouter :

   | Nom exact | Valeur | Accès au notebook |
   |---|---|---|
   | `HF_TOKEN` | jeton Hugging Face en lecture | activé |
   | `SCALEDOWN_API_KEY` | clé ScaleDown (facultatif) | activé |

4. Exécution > **Tout exécuter**. Colab peut afficher un avertissement sur l'origine du notebook (GitHub) : choisir « Exécuter quand même ».
5. Attendre la fin de la cellule 7 (« Lien et QR code ») : elle affiche un lien `https://….trycloudflare.com/live` et un QR code. Ouvrir le lien dans un navigateur, ou scanner le QR code avec un téléphone.

Ce que fait chaque cellule :

| Cellule | Rôle |
|---|---|
| 1. Vérifier le GPU et installer les dépendances | `ffmpeg`, `transformers`, `speechbrain`, `pyannote.audio`, `piper-tts`… |
| 2. Cloner le dépôt | branche `main` dans `/content/NLP` |
| 3. Lire les secrets et choisir les moteurs | lit `HF_TOKEN` et `SCALEDOWN_API_KEY`, vérifie l'accès au modèle Cohere |
| 4. Lancer le serveur | `uvicorn api:app` sur le port 7860 |
| 5. Ouvrir le tunnel public | lien `trycloudflare.com` (HTTPS, sans compte) |
| 6. Précharger les modèles du direct | S3, Opus-MT et NLLB-200 1,3B |
| 7. Lien et QR code | page à ouvrir |
| 8. Vérifier le serveur (facultatif) | test de bout en bout sur de la parole FLEURS, bilan PASS/FAIL |
| 9. Démarrer un direct depuis le notebook (facultatif) | exemple d'appel à l'API, en commentaire |
| 10. Mettre à jour sans changer de lien | `git pull` et relance du serveur, le tunnel reste ouvert |
| 11. Journal du serveur | 40 dernières lignes de `/content/server.log` |

Si `HF_TOKEN` manque ou si l'accès au modèle Cohere n'est pas accordé, la cellule 3 l'indique et le notebook continue sans S3. Le direct est alors indisponible ; la page `/video` fonctionne avec Seamless et S0, et la cellule 7 pointe vers elle.

## 3. Démonstrations à essayer

Les pages sont accessibles depuis le menu en haut de l'interface : `/live` (direct), `/video` (fichier), `/sessions` (sessions passées), `/rapport?sid=…` (rapport d'une session).

### 3.1 Direct sur une chaîne arabe

1. Sur `/live`, ouvrir le menu « Source du son » et choisir « Lien d'une chaîne ».
2. Le champ contient déjà le lien du direct YouTube d'Al Jazeera, <https://www.youtube.com/live/bNyUyrR0PHo> ; le remplacer au besoin par un autre direct arabe. Cliquer sur « Démarrer ».

Résultat attendu : une fois le lien résolu par `yt-dlp` sur le serveur, les phrases apparaissent en gris pendant que le présentateur parle, puis passent en blanc environ trois secondes après la fin de chaque phrase. Un nouveau paragraphe, marqué « Voix 2 », « Voix 3 »…, signale un changement de locuteur. Le menu « Options du direct » permet d'afficher l'arabe, d'agrandir le texte, d'activer la lecture vocale et de copier le lien de la session pour la suivre sur un autre écran (`/live?sid=…`).

### 3.2 Micro du téléphone

1. Ouvrir le lien public sur un téléphone (le QR code de la cellule 7).
2. Choisir « Micro », autoriser l'accès au micro, puis faire entendre de la parole arabe (une vidéo d'information lue sur un autre appareil, par exemple).

Résultat attendu : même affichage gris puis blanc. Or le navigateur n'autorise le micro qu'en HTTPS : c'est pourquoi le notebook ouvre un tunnel Cloudflare. La source « Onglet » (son d'un onglet de Chrome sur ordinateur) fonctionne de la même manière.

### 3.3 Vidéo déposée

1. Sur `/video`, déposer un fichier audio ou vidéo (mp4, mp3, wav…) d'un journal arabe, de préférence une à trois minutes.
2. Choisir le moteur dans le menu « Moteur » puis « Sous-titrer ».

| Moteur | Chaîne | Remarque |
|---|---|---|
| `s3` | S3 transcrit l'arabe, puis son Qwen traduit le texte | demande l'accès Cohere |
| `seamless` | SeamlessM4T v2 large, de la parole au français | meilleur score de bout en bout sur FLEURS |
| `s0` | Whisper small puis Opus-MT | référence de départ, la plus rapide |

Résultat attendu : la vidéo se lit avec ses sous-titres français synchronisés ; les boutons proposent les exports `.srt` et `.vtt`. Comme le T4 ne garde qu'un groupe de modèles en mémoire, changer de moteur décharge le précédent, et le premier passage de chaque moteur comprend son chargement. Pour une vidéo déposée, une minute de journal a demandé 26 s de calcul.

Le même fichier peut être rejoué en temps réel sur `/live`, source « Fichier vidéo » : la vidéo s'affiche au-dessus des sous-titres du direct.

### 3.4 Rapport de fin de session

1. Arrêter une session de direct (« Arrêter »), puis « Voir le rapport ». Toutes les sessions sont aussi listées sur `/sessions`.
2. Choisir les options (« Distinguer les voix », « Relire la traduction », « Joindre le texte arabe »), puis « Générer le rapport ».

Résultat attendu : la page affiche trois onglets une fois le rapport prêt (61 s pour dix minutes d'audio sur un T4). « Résumé » : sujet, points clés, personnes et chiffres cités, chaque point suivi d'un horodatage `[mm:ss]` cliquable qui place la lecture audio à cet instant. « Transcript » : le texte par locuteur. « Détails » : méthodes employées, durées, avertissements. Les fichiers `transcript.md`, `resume.md`, `sous_titres.srt`, `rapport.json` et `transcript.docx` sont téléchargeables.

Avec `SCALEDOWN_API_KEY`, le résumé est rédigé par l'API ScaleDown à partir de la transcription arabe (2,9 s sur une session de 60 segments). Sans clé, il est rédigé en local par Qwen2.5-7B-Instruct quantifié en 4 bits (environ 110 s).

### 3.5 Lecture vocale

Dans le rapport, « Écouter le résumé » lit le résumé en français. Dans le direct, l'option « Lecture vocale » lit chaque phrase définitive. La synthèse est faite par Piper (voix `fr_FR-siwis-medium`, sur CPU), avec edge-tts en repli si Piper est indisponible.

## 4. Dépannage

| Symptôme | Cause probable | Que faire |
|---|---|---|
| « yt-dlp n'a pas pu ouvrir ce lien » ou aucun audio reçu sur un lien YouTube | YouTube bloque parfois les adresses des serveurs Colab | télécharger la vidéo ailleurs et utiliser la source « Fichier vidéo » sur `/live`, ou la page `/video` |
| Le lien public ne répond plus | le tunnel Cloudflare est tombé | relancer la cellule 5 seule ; un **nouveau** lien s'affiche, puis relancer la cellule 7 pour le QR code |
| `CUDA out of memory` dans le journal | plusieurs modèles sont restés en mémoire | Exécution > Redémarrer la session, puis Tout exécuter |
| Le premier direct ou la première vidéo met plusieurs minutes à démarrer | téléchargement des modèles (plusieurs Go) | attendre ; la cellule 6 précharge le direct pour éviter cette attente pendant une démonstration |
| La cellule 3 signale « Accès au modèle Cohere refusé » | conditions non acceptées, ou jeton d'un autre compte | accepter les conditions sur la page du modèle avec le compte du jeton, relancer la cellule 3, puis la cellule 10 (qui relance le serveur avec ces réglages) et la cellule 7 |
| Le rapport indique « pyannote indisponible : repli ECAPA » | accès non accordé au modèle pyannote | sans conséquence : la diarisation passe par ECAPA |
| La page affiche une ancienne version | le code a changé sur GitHub après le lancement | cellule 10, « Mettre à jour sans changer de lien » |
| Toute autre erreur | | cellule 11 (journal du serveur) |

## 5. Mode local sans GPU

Le moteur factice simule l'ASR et les traducteurs avec quatre phrases arabes fixes : il permet de parcourir l'interface et de tester l'API sans télécharger de modèle. Les chiffres qu'il affiche (retard, réécriture) ne mesurent rien.

Installation (Python 3.11 ou plus récent ; testé avec Python 3.13 sous Windows) :

```bash
pip install torch
pip install -r speech_app/requirements.txt fastapi "uvicorn[standard]" python-multipart pyyaml
```

Le mode factice importe le code du direct, qui dépend de `torch` ; aucun modèle n'est téléchargé. Sous Linux sans GPU, la version CPU de `torch` est plus légère : `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

Lancement de l'application complète (pages, API, interface Gradio sur `/atelier`) :

```bash
cd speech_app
PORTEPAROLE_LIVE=factice python -m uvicorn api:app --port 8770
```

```powershell
cd speech_app
$env:PORTEPAROLE_LIVE = "factice"; python -m uvicorn api:app --port 8770
```

Ouvrir <http://127.0.0.1:8770/live>, choisir « Micro » (sur `localhost`, le navigateur accepte le micro sans HTTPS) : le moteur factice renvoie ses phrases quel que soit le son. Le rapport de fin fonctionne sans GPU, avec un résumé extractif signalé comme tel. Dans ce mode, `/video` propose des moteurs de démonstration au texte fictif, et la lecture vocale répond 503 si `piper-tts` n'est pas installé.

Serveur du direct seul, sans l'interface Gradio :

```bash
cd speech_app
python live.py --factice            # http://127.0.0.1:8765/live
```

Pour utiliser les vrais modèles sur une machine équipée d'un GPU NVIDIA, définir `PORTEPAROLE_NLP_REPO` (racine du dépôt) et `HF_TOKEN`, installer `speech_app/requirements-api.txt`, et lancer la même commande `uvicorn` sans `PORTEPAROLE_LIVE`.

## 6. Docker

`speech_app/Dockerfile` construit une image GPU (base `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`, port 7860) qui contient `speech_app/` mais ni le reste du dépôt ni les poids des modèles. `speech_app/docker-compose.yml` monte le dépôt dans `/nlp` et un cache Hugging Face dans `/cache`.

L'image n'a pas été construite ni testée dans l'état actuel du dépôt. La lecture des fichiers montre plusieurs limites :

- **Le direct ne démarre pas** dans l'image telle quelle : `live.py` cherche `code/direct/` dans le dossier parent de `speech_app/`, soit `/` dans le conteneur, au lieu de `/nlp`.
- `yt-dlp` n'est pas installé dans l'image : la source « Lien d'une chaîne » échouerait même après correction.
- Les sessions sont écrites dans `/app/sessions`, perdues à la suppression du conteneur.
- `pyannote.audio` 4 peut exiger une version de `torch` plus récente que celle de l'image et la réinstaller.

En revanche, la page `/video` (moteurs `s0`, `seamless`, `s3`) et l'API `/api/translate` n'ont pas ces dépendances et devraient fonctionner :

```bash
cd speech_app
docker build -t kalam .
docker run --gpus all -p 7860:7860 \
  -v /chemin/vers/NLP:/nlp:ro -v "$HOME/hf_cache:/cache" \
  -e HF_TOKEN=hf_... -e PORTEPAROLE_ENGINES=s0,seamless,s3 \
  kalam
```

Contournement non testé pour le direct : lancer le serveur depuis le dépôt monté, avec les sessions dans le volume de cache.

```bash
docker run --gpus all -p 7860:7860 \
  -v /chemin/vers/NLP:/nlp:ro -v "$HOME/hf_cache:/cache" \
  -e HF_TOKEN=hf_... -e PORTEPAROLE_SESSIONS=/cache/sessions -e PORTEPAROLE_CACHE=/cache/kalam \
  -w /nlp/speech_app kalam uvicorn api:app --host 0.0.0.0 --port 7860
```

Faute de test de l'image, Colab reste le seul chemin vérifié pour une démonstration.

## 7. Reproduire les mesures sur Kaggle

Les mesures sur FLEURS ont été calculées par les kernels de `kaggle/` sur un GPU T4 de Kaggle. Chaque kernel est écrit dans `kaggle/<nom>/source.py` (cellules `# %%`) et converti en notebook par `kaggle/construire_notebooks.py`. Les sorties récupérées sont dans `resultats/kaggle/<nom>/` ; les notebooks exécutés sont copiés dans `notebooks/kaggle_<nom>.ipynb`.

| Kernel | Mesure | Dépend de |
|---|---|---|
| `nlp-lora-traduction` | entraînement du LoRA de Qwen2.5-1.5B, BLEU/chrF++ avant et après | `nlp-code`, `nlp-opus` |
| `nlp-eval-systemes` | S0, SeamlessM4T v2, Whisper-large-v3 sur FLEURS ; S3 si `HF_TOKEN` est attaché | `nlp-code` |
| `nlp-eval-traducteurs` | six traducteurs de texte sur l'arabe de référence | `nlp-code`, sorties des deux kernels précédents |
| `nlp-eval-s3-lora` | cascades (transcription puis Qwen, avec et sans LoRA) | idem |
| `nlp-comet` | COMET (`Unbabel/wmt22-comet-da`) sur toutes les traductions | sorties des quatre kernels précédents |
| `nlp-eval-s3` | S3 sur FLEURS | `nlp-code`, `HF_TOKEN` ; **non exécuté** |
| `nlp-analyse-llm` | analyse du décodeur : tokens, logits, température, grille de 24 requêtes | `nlp-code`, `nlp-opus` |

Étapes, avec la CLI Kaggle configurée (`~/.kaggle/kaggle.json` ou variables `KAGGLE_USERNAME` et `KAGGLE_KEY`) :

```bash
pip install kaggle nbformat
# 1. Jeux de données privés : code du dépôt (avec le projecteur S3) et sous-ensemble OPUS
python kaggle/preparer_datasets.py --sortie ../kaggle_travail --utilisateur VOTRE_COMPTE
kaggle datasets create -p ../kaggle_travail/nlp-code
kaggle datasets create -p ../kaggle_travail/nlp-opus
# 2. Notebooks et métadonnées au nom de votre compte
python kaggle/construire_notebooks.py --utilisateur VOTRE_COMPTE
# 3. Exécution, dans l'ordre du tableau
kaggle kernels push -p kaggle/nlp-eval-systemes
kaggle kernels status VOTRE_COMPTE/nlp-eval-systemes
kaggle kernels output VOTRE_COMPTE/nlp-eval-systemes -p resultats/kaggle/nlp-eval-systemes
```

Remarques :

- **`HF_TOKEN` sur Kaggle** : les kernels qui chargent S3 (`nlp-eval-systemes`, `nlp-eval-s3`) lisent le secret Kaggle `HF_TOKEN`. La CLI ne sait pas l'attacher : ouvrir le kernel dans l'interface web, Add-ons > Secrets, ajouter `HF_TOKEN` et cocher le kernel, puis relancer. Sans ce secret, S3 est sauté (c'est ce qui s'est produit pour les résultats du dépôt).
- `nlp-opus` est construit à partir de `donnees/presse/p_v1.jsonl`, qui n'est pas versionné. Le régénérer avec `code/donnees/telecharger_opus.py` puis `code/donnees/filtrer_presse.py`, ou se limiter à `nlp-eval-systemes`, le seul kernel qui n'en dépend ni directement ni par un autre kernel. L'adaptateur LoRA obtenu est versionné dans `resultats/kaggle/nlp-lora-traduction/adaptateur_lora/`.
- FLEURS est téléchargé par les kernels eux-mêmes depuis Hugging Face (environ 400 Mo).
- `kaggle/lire_log.py <fichier.log>` affiche l'erreur d'un journal de kernel.

## 8. Lancer les tests

Les tests n'ont pas besoin de GPU : les modèles sont remplacés par des moteurs factices.

```bash
cd speech_app
python -m pytest tests -q --ignore=tests/test_snapshot.py
cd ..
python -m pytest code/direct/tests -q
```

Résultats obtenus le 24 octobre 2026 (Windows 11, Python 3.13, sans GPU) :

```text
speech_app/tests   92 passed, 2 skipped
code/direct/tests  30 passed
```

`tests/test_snapshot.py` concerne l'ancienne édition figée de l'interface (`app.py`) et reste hors de la commande de référence. `code/direct/tests/test_flux_s3.py::TestNotebook::test_notebook_en_mode_essai` exécute un notebook dans un noyau séparé (`nbclient`) qui recharge `torch` : sous Windows, il a échoué une fois par manque de mémoire virtuelle (`WinError 1455`) quand une autre suite tournait en parallèle, et passe seul. Les autres modules ont leurs propres tests (`python -m pytest code/<module>/tests -q`).
