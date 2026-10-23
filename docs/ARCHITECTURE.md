# Architecture

Ce document décrit les composants de Kalam, leurs échanges, l'API HTTP et WebSocket et les variables d'environnement. Il est établi à partir du code de la branche `integration`.

## Sommaire

1. [Vue d'ensemble](#1-vue-densemble)
2. [Moteurs de transcription et de traduction](#2-moteurs-de-transcription-et-de-traduction)
3. [Mode direct](#3-mode-direct)
4. [Sessions et rapport de fin](#4-sessions-et-rapport-de-fin)
5. [Lecture vocale](#5-lecture-vocale)
6. [Routes de l'API](#6-routes-de-lapi)
7. [Messages WebSocket](#7-messages-websocket)
8. [Variables d'environnement](#8-variables-denvironnement)
9. [Organisation des fichiers](#9-organisation-des-fichiers)

## 1. Vue d'ensemble

Kalam est un seul serveur FastAPI (`speech_app/api.py`) lancé par `uvicorn api:app`. Il sert les pages web (HTML et JavaScript sans framework, dans `speech_app/static/`), l'API, les WebSocket du direct et, sur `/atelier`, une ancienne interface Gradio de comparaison des moteurs.

```text
 navigateur ou téléphone                                serveur (speech_app/)
 ┌──────────────────────┐   PCM 16 kHz (WebSocket)   ┌───────────────────────────────────────────────────────────┐
 │ /live   micro,onglet │ ─────────────────────────► │ live.py ── SessionLive ── FluxIncremental (code/direct/)  │
 │         lien,fichier │ ◄───────────────────────── │    │            │                 │                       │
 │                      │  provisoire / definitif    │    │            │                 ▼                       │
 │                      │                            │    │            │       live_moteurs.MoteurDirectS3       │
 │                      │                            │    │            │       S3 + Opus-MT + NLLB + ECAPA       │
 │                      │                            │    │            ▼                                         │
 │                      │                            │    │      sessions.py : audio.wav, segments.jsonl         │
 │ /rapport             │ ◄── rapport.json, .md ──── │ session_api.py ── session_report.py                       │
 │                      │                            │    diarisation, relecture Qwen2.5-7B, résumé ScaleDown    │
 │ /video               │ ── fichier ──────────────► │ api.py /api/translate ── lazy_engines (s0, seamless, s3)  │
 └──────────────────────┘                            │ yt-dlp + ffmpeg : audio d'un lien YouTube ou HLS          │
                                                     └───────────────────────────────────────────────────────────┘
```

Tous les appels de modèle passent par un verrou unique, `lazy_engines.GPU_LOCK`, et par un registre `lazy_engines.POOL` qui ne garde qu'un groupe de modèles en mémoire (le T4 de Colab a 16 Go). Les groupes sont `direct` (S3, Opus-MT, NLLB), `s0`, `seamless` et `s3`. Demander un autre groupe décharge le précédent ; le rapport de fin libère le groupe chargé avant de charger son propre LLM.

## 2. Moteurs de transcription et de traduction

### S0 : référence sans entraînement

`code/demo/systeme_s0.py`, appelé par `lazy_engines.S0Engine`. Découpage par seuil d'énergie (segments de 30 s au plus), transcription par `openai/whisper-small`, traduction par `Helsinki-NLP/opus-mt-ar-fr`. Configuration : `configs/s0.yaml`.

### SeamlessM4T v2

`lazy_engines.SeamlessEngine`. `facebook/seamless-m4t-v2-large` traduit la parole en texte français sans texte arabe intermédiaire, par fenêtres fixes de 25 s, recherche en faisceau de largeur 5, fp16 sur GPU.

### S3 : encodeur gelé, projecteur entraîné, LLM gelé

Code : `code/projecteur/` (modèle, entraînement, transcription) et `code/ecoute/` (encodage audio). Réglages : `configs/s3.yaml` et `configs/encodage_s3.yaml`.

```text
 audio 16 kHz ──► encodeur de CohereLabs/cohere-transcribe-arabic-07-2026 (gelé, 1 895,3 M param.)
                  vecteurs de dimension 1 280 à 12,5 Hz
              ──► projecteur : LayerNorm, Linear 1280→3072, GELU, Linear 3072→1536 (entraîné, 8,66 M param.)
              ──► Qwen/Qwen2.5-1.5B-Instruct (gelé), vecteurs audio insérés dans inputs_embeds
              ──► texte arabe
```

**Entraînement (temps 1).** Seul le projecteur apprend, à transcrire l'arabe. Données : 50 000 segments de MGB-2 (144,9 h, 2 086 émissions), filtrés sur la durée (5 à 25 s), le nombre de mots (au moins six) et le WMER (au plus 15 %), séparés par émission pour qu'aucune émission du test ne serve à l'entraînement. Les vecteurs de l'encodeur sont précalculés une fois. AdamW, taux 10⁻³, lot effectif 16 (micro-lot de 4, accumulation de 4), bf16, sur un Apple M5 Pro. Entraînement arrêté au pas 1 730 ; le projecteur retenu est celui du pas 1 500 (`modeles/s3_temps1_50k/projecteur_meilleur.pt`). Journal détaillé : `code/projecteur/journal_s3.md`.

**Génération.** Décodage glouton, longueur limitée à `ceil(7 × durée) + 10` tokens, puis suppression des boucles (une suite de 1 à 8 mots répétée au moins quatre fois est ramenée à une occurrence). Un fichier long est découpé en morceaux de 20 s au plus, coupés au point le plus calme des 5 dernières secondes. Sur un T4, le projecteur entraîné en bf16 tourne en fp16.

**Temps 2 non réalisé.** Le second temps prévu (traduire directement depuis l'audio) n'a pas été entraîné : avec le seul projecteur du temps 1, Qwen recopie l'arabe quand on lui demande de traduire l'audio projeté. C'est pourquoi S3 sert seulement à transcrire, et un traducteur de texte produit le français. Le moteur `s3` de `/video` traduit avec le même Qwen2.5-1.5B (consigne de `configs/s3.yaml`, section `traduction_texte`) ; le direct traduit avec Opus-MT et NLLB.

### Traducteurs de texte du direct

| Rôle | Modèle | Choix |
|---|---|---|
| texte gris, provisoire | `Helsinki-NLP/opus-mt-ar-fr` | le plus rapide mesuré : 0,19 s par phrase (médiane), 0,17 Go |
| texte blanc, définitif | `facebook/nllb-200-distilled-1.3B` | meilleur BLEU, chrF++ et COMET mesurés : 0,82 s par phrase, 2,78 Go |

Les deux sont chargés en fp16 sur GPU, décodage glouton (`live_moteurs.OpusMT`, `live_moteurs.NLLB`). Si la traduction définitive contient plus de 30 % de lettres arabes, Opus-MT la remplace. Toute écriture arabe restante et les propositions répétées à la suite sont retirées du sous-titre (`live_moteurs.nettoyer_sous_titre`). `PORTEPAROLE_TRAD_BLANC=qwen` rétablit l'ancien traducteur définitif, le Qwen de S3, avec l'adaptateur LoRA de `PORTEPAROLE_LORA` s'il est fourni.

## 3. Mode direct

Code : `speech_app/live.py` (sessions, sources, WebSocket) et `code/direct/flux_incremental.py` (segmentation et double passe). Réglages : `configs/direct.yaml`, sections `direct`, `direct_s3` et `direct_live`.

### Sources audio

| Source | Chemin de l'audio |
|---|---|
| Micro, onglet | le navigateur capte le son (AudioWorklet `pcm-worklet.js`) et envoie du PCM 16 kHz mono int16 par paquets de 250 ms sur `/ws/live` |
| Lien d'une chaîne | le serveur résout l'URL avec `yt-dlp -g` puis lit le flux avec `ffmpeg` ; un direct interrompu est rouvert jusqu'à cinq fois |
| Fichier vidéo | le fichier est déposé puis rejoué par `ffmpeg -re`, au rythme de l'audio |

Pour un flux HLS ou un direct YouTube, `ffmpeg` est lancé sans `-re` : le flux avance déjà au rythme du direct, et `-re` le faisait dériver derrière la playlist.

### Segmentation et règles de clôture

Le flux est découpé en segments d'après l'énergie, par trames de 100 ms. Le niveau de parole est le 90ᵉ centile des trames de la dernière minute ; une trame est silencieuse sous ce niveau moins 20 dB, et toujours sous −60 dBFS. Un segment est clos :

- après 0,5 s de silence, s'il dure déjà au moins 2 s ;
- d'office à 10 s, au point le plus calme des 3 dernières secondes.

Le silence pur est jeté ; 0,2 s de silence est gardé avant le début de la parole.

### Double passe

```text
 parole ───────────────────────────────────────────►│ silence de 0,5 s : clôture
        │ passe grise, chaque seconde d'audio        │
        │ S3 sur le segment en cours                 │ S3 sur le segment entier (ou reprise de la
        │ Opus-MT ──► message "provisoire" (gris)    │ dernière passe grise si elle finit à moins de 0,6 s)
        │                                            │ NLLB ──► message "definitif" (blanc), même id_segment
        │                                            │ ECAPA ──► numéro de voix
```

- **Passe grise** : toutes les secondes d'audio, si le segment en cours dure au moins 1 s, S3 le transcrit et Opus-MT le traduit. Le message n'est envoyé que si l'arabe a changé. Le champ `fr_stable` donne le début commun aux deux dernières passes, que la page peut figer.
- **Clôture** : le segment entier est transcrit, sauf si la dernière passe grise couvre déjà le segment à 0,6 s près (son arabe est alors réutilisé). NLLB produit le texte définitif, qui remplace le gris du même `id_segment`.
- **Saut de la passe grise** : quand plus de quatre paquets (1 s d'audio) attendent dans la file, le calcul est en retard. Le serveur traite tout l'audio en attente d'un coup, clôt les segments possibles et saute la passe grise jusqu'à avoir rattrapé le direct.
- **Retard** : mesuré à l'horloge, de l'arrivée du paquet qui contient la fin de la parole jusqu'à l'émission du message définitif.

### Changement de locuteur

À la clôture, une empreinte `speechbrain/spkrec-ecapa-voxceleb` est calculée sur le segment (au moins 1 s). `SuiviLocuteurs` l'attribue à la voix connue la plus proche si la similarité cosinus atteint 0,45, sinon ouvre une nouvelle voix (huit au plus). Le message définitif porte alors `locuteur` et `changement_locuteur`. Si speechbrain manque, le direct continue sans locuteurs.

### Spectateurs

Chaque session a une salle de diffusion (`live.Diffusion`). Un écran qui rejoint la session par `/live?sid=…` reçoit l'état, les segments définitifs déjà émis et le gris en cours, puis le direct. Une session terminée est rejouée depuis le disque.

## 4. Sessions et rapport de fin

### Session

`speech_app/sessions.py`. Une session est un dossier, écrit au fil du direct ou d'une vidéo traitée par `/api/translate` :

```text
<PORTEPAROLE_SESSIONS>/<sid>/
    meta.json        sid, source, moteur, cree_le, duree_s, titre, bilan_direct
    audio.wav        tout l'audio reçu, 16 kHz mono int16
    segments.jsonl   un segment clos par ligne : id, debut_s, fin_s, ar, fr, locuteur
    rapport/         transcript.md, resume.md, sous_titres.srt, rapport.json, transcript.docx
```

L'identifiant `sid` a la forme `AAAAMMJJ-HHMMSS-xxxxxx`.

### Rapport

`speech_app/session_report.py`, lancé en tâche de fond par `POST /api/sessions/{sid}/rapport`. Quatre étapes :

1. **Diarisation** : `pyannote/speaker-diarization-community-1` si `HF_TOKEN` y donne accès. À défaut, des empreintes ECAPA par segment, regroupées par classification agglomérative (seuil de similarité 0,30). Si les deux échouent, le rapport reprend les numéros de voix posés pendant le direct.
2. **Relecture** : Qwen2.5-7B-Instruct quantifié en 4 bits nf4 (calcul fp16) relit les segments qui contiennent un nom, un chiffre, un mot répété ou une majuscule suspecte, par lots de huit, avec deux segments avant et un après comme contexte et le lexique de noms propres `donnees/noms/noms_v1.csv`. Le texte d'origine est gardé à côté. Le modèle se charge pendant la diarisation.
3. **Résumé** : sujet, points clés, personnes et lieux, chiffres, chaque point suivi d'un horodatage `[mm:ss]`. Avec `SCALEDOWN_API_KEY`, l'API ScaleDown résume la transcription **arabe** de S3, ce qui évite de propager les erreurs de traduction. Sans clé, Qwen2.5-7B résume le français relu, et sans GPU le rapport se contente d'un résumé extractif, signalé comme tel. Les horodatages cités sont vérifiés contre les segments et rattachés au plus proche si besoin.
4. **Écriture** des fichiers de `rapport/`.

Sans GPU ou en cas d'échec du LLM, le rapport est quand même produit et `rapport.json` liste les avertissements.

## 5. Lecture vocale

`speech_app/tts.py`. Piper (`piper-tts`, voix ONNX `fr_FR-siwis-medium` téléchargée depuis `rhasspy/piper-voices`) tourne sur CPU. edge-tts sert de repli (réseau requis). Les audios sont mis en cache selon le texte, la voix et le moteur ; le Markdown et les horodatages sont retirés avant la synthèse.

## 6. Routes de l'API

Relevées dans `api.py`, `pages.py`, `live.py` et `session_api.py`. La documentation interactive est servie sur `/docs` (OpenAPI).

### Pages

| Méthode | Chemin | Rôle |
|---|---|---|
| GET | `/` | page du direct (même page que `/live`) |
| GET | `/live` | direct ; `/live?sid=…` suit une session en cours ou rejoue une session passée |
| GET | `/video` | dépôt d'un fichier, sous-titres synchronisés, exports |
| GET | `/sessions` | liste des sessions |
| GET | `/rapport` | rapport d'une session (`?sid=…`) |
| GET | `/atelier` | ancienne interface Gradio (comparaison des moteurs) |
| GET | `/static/{nom}`, `/live/assets/{nom}`, `/manifest.webmanifest` | feuilles de style, scripts, icônes (liste blanche) |

### Traduction de fichiers

| Méthode | Chemin | Rôle |
|---|---|---|
| GET | `/api/health` | état du service, mode, nombre de moteurs |
| GET | `/api/engines` | moteurs disponibles et groupe chargé |
| POST | `/api/translate` | formulaire `file`, `engine` (`s0`, `seamless`, `s3`), `task` (`st` ou `asr`), `format` (`json`, `srt`, `vtt`), `wait` (`true` ou `false`) |
| GET | `/api/jobs/{job_id}` | état et résultat d'un travail lancé avec `wait=false` |
| GET | `/api/jobs/{job_id}/subtitles.{ext}` | sous-titres d'un travail terminé, `srt` ou `vtt` |

Exemple :

```bash
curl -F file=@journal.mp4 -F engine=seamless -F format=srt http://localhost:7860/api/translate
```

### Direct

| Méthode | Chemin | Rôle |
|---|---|---|
| WS | `/ws/live` | envoi de l'audio (`?source=micro` ou `onglet`, `&titre=…`), réception des messages de la session |
| WS | `/ws/regarder/{sid}` | spectateur : historique puis direct |
| GET | `/api/live/statut` | moteur disponible, groupe chargé, mode factice, présence de `ffmpeg` et `yt-dlp`, sessions en cours |
| POST | `/api/live/precharger` | charge S3, Opus-MT et NLLB en arrière-plan |
| POST | `/api/live/url` | corps JSON `{"url": "…", "titre": "…"}` : le serveur tire l'audio du lien |
| POST | `/api/live/fichier` | dépôt d'un fichier (formulaire `file`, `titre`) |
| POST | `/api/live/{sid}/demarrer` | rejoue le fichier déposé en temps réel |
| GET | `/api/live/{sid}/media` | fichier déposé, pour le lecteur vidéo |
| GET | `/api/live/{sid}/etat` | état, bilan, nombre de segments |
| POST | `/api/live/{sid}/arreter` | termine la session et renvoie le bilan (retard moyen, p95, max, taux de réécriture) |

### Sessions, rapport, lecture vocale

| Méthode | Chemin | Rôle |
|---|---|---|
| GET | `/api/sessions` | sessions existantes, les plus récentes d'abord |
| POST | `/api/sessions/{sid}/rapport` | lance le rapport ; options JSON facultatives : `diarisation` (`auto`, `pyannote`, `ecapa`, `off`), `nb_locuteurs`, `seuil_similarite`, `revision`, `resume`, `inclure_arabe`, `docx` |
| GET | `/api/sessions/{sid}/rapport` | état (`absent`, `en_cours`, `pret`, `erreur`) et rapport une fois prêt |
| GET | `/api/sessions/{sid}/fichiers/{nom}` | `transcript.md`, `resume.md`, `sous_titres.srt`, `rapport.json`, `transcript.docx`, `audio.wav`, `segments.jsonl`, `meta.json` |
| GET | `/api/tts?texte=…` | synthèse vocale du texte (wav ou mp3) ; 503 si aucun moteur de synthèse n'est disponible |

Si `PORTEPAROLE_API_TOKEN` est défini, les routes `/api/*` exigent l'en-tête `Authorization: Bearer <jeton>`. Les routes du direct et les WebSocket acceptent aussi le paramètre `?token=<jeton>`.

## 7. Messages WebSocket

Le client qui envoie l'audio sur `/ws/live` transmet des trames binaires (PCM 16 kHz mono int16, 8 000 octets par paquet de 250 ms) et termine par le texte `stop` ou `{"type": "stop"}`. Il reçoit les mêmes messages JSON que les spectateurs.

| `type` | Champs | Sens |
|---|---|---|
| `etat` | `etat` (`chargement`, `pret`, `en_cours`, `termine`, `erreur`), `message`, `sid`, `source`, `niveau_db`, `duree_s`, `nb_segments`, `retard_moyen_s`, `retard_dernier_s`, `factice` | état de la session, envoyé au moins toutes les 0,5 s pendant le traitement |
| `provisoire` | `id_segment`, `fr`, `fr_stable`, `ar`, `debut_s`, `fin_s`, `retard_s` (`null`) | texte gris du segment en cours |
| `definitif` | `id_segment`, `fr`, `ar`, `debut_s`, `fin_s`, `retard_s`, et si disponibles `locuteur`, `changement_locuteur` ; `vide: true` pour un bruit sans parole | texte blanc, remplace le provisoire du même `id_segment` |
| `fin` | `sid`, `bilan`, `erreur`, `duree_s` | fin de session |
| `ping` | | maintien de la connexion, toutes les 15 s sans message |

Exemple de message définitif :

```json
{"type": "definitif", "id_segment": "seg_0001", "debut_s": 0.0, "fin_s": 6.25,
 "ar": "أهلا بكم في نشرة الأخبار لهذا المساء", "fr": "Bonsoir et bienvenue dans ce journal.",
 "retard_s": 0.118, "locuteur": 1, "changement_locuteur": false}
```

Code de fermeture 4401 : jeton manquant ou invalide.

## 8. Variables d'environnement

Relevées par recherche de `PORTEPAROLE_` et `os.environ` dans `speech_app/` et `code/`.

### Modèles et dépôt

| Variable | Défaut | Effet |
|---|---|---|
| `HF_TOKEN` | aucun | jeton Hugging Face : encodeur Cohere de S3 (obligatoire pour S3), diarisation pyannote (facultatif) |
| `PORTEPAROLE_NLP_REPO` | aucun | racine du dépôt ; active les moteurs réels de `/video` et le direct S3 |
| `PORTEPAROLE_S3_REPO` | aucun | ancien nom de `PORTEPAROLE_NLP_REPO`, lu en second |
| `PORTEPAROLE_ENGINES` | `s0,seamless,s3` | moteurs proposés par `/video` et `/api/translate` ; `s3_direct` ajoute l'essai de traduction directe par S3 |
| `PORTEPAROLE_S3_CKPT` | `modeles/s3_temps1_50k/projecteur_meilleur.pt` | projecteur S3 à charger |
| `PORTEPAROLE_TRAD_BLANC` | `nllb` | traducteur du texte définitif du direct ; `qwen` : le Qwen de S3 |
| `PORTEPAROLE_LORA` | aucun | adaptateur LoRA posé sur le Qwen de S3 (utilisé seulement avec `PORTEPAROLE_TRAD_BLANC=qwen`) |
| `PORTEPAROLE_LLM_RAPPORT` | `Qwen/Qwen2.5-7B-Instruct` | LLM de relecture et de résumé du rapport ; `off` le désactive |
| `SCALEDOWN_API_KEY` | aucun | résumé du rapport par l'API ScaleDown, à partir de l'arabe |
| `SCALEDOWN_API_BASE` | `https://api.scaledown.xyz` | adresse de l'API ScaleDown |

### Serveur et stockage

| Variable | Défaut | Effet |
|---|---|---|
| `PORTEPAROLE_API_TOKEN` | aucun | protège l'API par un jeton |
| `PORTEPAROLE_MAX_UPLOAD_MB` | `500` | taille maximale d'un fichier déposé |
| `PORTEPAROLE_SESSIONS` | `speech_app/sessions` | dossier des sessions |
| `PORTEPAROLE_CACHE` | `speech_app/cache` | cache des empreintes ECAPA et des voix Piper |
| `PORTEPAROLE_LIVE` | aucun | `factice` : moteur de direct simulé, sans modèle |
| `PORTEPAROLE_LIVE_DELAI` | `0.05` | délai simulé de chaque appel du moteur factice, en secondes |
| `PORTEPAROLE_TTS` | `piper` | moteur de lecture vocale : `piper` (repli edge-tts), `edge` ou `off` |
| `PORTEPAROLE_TTS_VOIX` | `fr_FR-siwis-medium` | voix Piper |
| `PORTEPAROLE_LEXIQUE` | `donnees/noms/noms_v1.csv` | lexique de noms propres de la relecture |

### Ancienne interface (`/atelier` avec `PORTEPAROLE_FRONT=old`, `app.py`)

| Variable | Défaut | Effet |
|---|---|---|
| `PORTEPAROLE_FRONT` | aucun | `old` : ancienne interface complète (presse, vidéos, atelier) sur `/atelier` |
| `PORTEPAROLE_MT` | `nllb` | `off` : désactive la traduction des articles de presse |
| `PORTEPAROLE_NLLB` | `facebook/nllb-200-distilled-600M` | traducteur des articles de presse |
| `PORTEPAROLE_SNAPSHOT` | aucun | édition figée, sans réseau ni modèle |
| `PORTEPAROLE_SHARE` | aucun | `1` : lien public Gradio (`python app.py`) |
| `PORTEPAROLE_CODE`, `PORTEPAROLE_CKPT_PROJ`, `PORTEPAROLE_CKPT_XATTN` | aucun | moteurs expérimentaux à projecteur Whisper, hors du produit final |

## 9. Organisation des fichiers

```text
speech_app/
  api.py              serveur : API de traduction de fichiers, montage des routeurs et de Gradio
  pages.py            pages statiques
  live.py             direct : sessions, sources (WebSocket, yt-dlp, ffmpeg), diffusion
  live_moteurs.py     moteur du direct (S3, Opus-MT, NLLB, ECAPA) et moteur factice
  lazy_engines.py     moteurs s0, seamless, s3 chargés à la demande ; verrou GPU et registre
  s3_engine.py        moteur S3 de /video (transcription puis traduction par Qwen)
  sessions.py         format d'une session sur disque
  session_api.py      routes des sessions, du rapport et de la lecture vocale
  session_report.py   rapport de fin : diarisation, relecture, résumé, écriture
  tts.py              lecture vocale (Piper, edge-tts)
  verifier_reel.py    vérification de bout en bout d'un serveur avec de la parole FLEURS
  front.py, app.py    interfaces Gradio (/atelier)
  static/             pages HTML, CSS et JavaScript ; static/marque.js porte le nom du produit
  colab_live.ipynb    lancement sur Colab (référence)
  Dockerfile, docker-compose.yml
  tests/              tests sans GPU
code/
  direct/             flux_incremental.py (direct), flux_s3.py (rejeu d'un fichier), tests
  projecteur/         modèle S3, entraînement du temps 1, transcription, journal_s3.md
  ecoute/             encodage Cohere et comparaison des encodeurs
  demo/               baseline S0
  donnees/            normalisation de l'arabe, MGB-2, OPUS, Wikidata, validation
  mesures/            WER, CER, BLEU, chrF++, COMET, bootstrap apparié
  cascade/            préparation des exemples et LoRA de la cascade
  seamless/, traduction/   benchmarks sur un épisode de MGB-2
configs/              s0.yaml, s3.yaml, encodage_s3.yaml, direct.yaml, s2_cascade.yaml, choix_briques.yaml, benchmark_traduction.yaml
kaggle/               kernels d'évaluation (source.py → notebook), préparation des jeux Kaggle
resultats/            kaggle/<kernel>/ (FLEURS), direct/ (Al Jazeera), resultats.csv (tableau historique)
modeles/s3_temps1_50k/   projecteur entraîné, evaluation_test.json, metriques.jsonl
notebooks/            analyses et copies des kernels Kaggle exécutés
```
