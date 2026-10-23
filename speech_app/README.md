# Traduction de parole arabe : API, interface et image Docker

Un seul service (FastAPI + Gradio, port 7860) pour traduire une vidéo ou un audio arabe en sous-titres français avec le moteur de son choix :

| Moteur (`engine`) | Chaîne | Particularité |
|---|---|---|
| `s0` | VAD, Whisper small, Opus-MT | baseline, tourne sur CPU |
| `seamless` | SeamlessM4T v2 large, audio vers français | ~9 Go de poids, GPU conseillé |
| `s3` | encodeur Cohere gelé, projecteur entraîné, Qwen2.5-1.5B (transcription arabe puis traduction du texte) | GPU, `HF_TOKEN`, `transformers>=5.4` |

Un seul moteur reste en mémoire : demander un autre moteur décharge le précédent. Le temps 2 de S3 (français direct) n'est pas prévu, donc le mode direct n'est pas exposé.

## Construire et lancer
    cd speech_app
    docker build -t porteparole .
    docker run -d -p 7860:7860 -e PORTEPAROLE_ENGINES=s0       -v /chemin/vers/NLP:/nlp:ro -v ~/hfcache:/cache porteparole
    # GPU : ajouter --gpus all, -e HF_TOKEN=... et PORTEPAROLE_ENGINES=s0,seamless,s3

Ce que l'image ne contient pas : le dépôt NLP (monté sur `/nlp`, S0 et S3 y lisent leur code), les poids des modèles (téléchargés au premier appel dans `/cache`) et le projecteur S3 entraîné, qui n'est que sur la branche `benchmarking` (`modeles/s3_temps1_50k/projecteur_meilleur.pt`) : le monter ou donner son chemin avec `PORTEPAROLE_S3_CKPT`.
L'encodeur Cohere (`CohereLabs/cohere-transcribe-arabic-07-2026`) est à accès contrôlé : `HF_TOKEN` d'un compte qui y a accès.

## API
    GET  /api/engines                      moteurs disponibles
    POST /api/translate                    fichier + engine=s0|seamless|s3 [&task=st|asr] [&format=json|srt|vtt] [&wait=false]
    GET  /api/jobs/{id}                    résultat d'un travail lancé avec wait=false
    curl -F file=@video.mp4 -F engine=s0 -F format=srt http://localhost:7860/api/translate
`PORTEPAROLE_API_TOKEN` protège l'API (en-tête `Authorization: Bearer ...`). `PORTEPAROLE_FRONT=old` affiche l'ancienne interface (presse, vidéos, atelier).

## Colab
`colab_api.ipynb` lance le même service sur un GPU T4 avec un lien public (il attend `speech_app.zip` dans Drive).

## Limites
- Sur CPU, seul `s0` est utilisable ; Seamless charge ~9 Go en mémoire et S3 exige un GPU.
- L'onglet vidéos de l'ancienne interface télécharge via yt-dlp, que YouTube bloque depuis un serveur : déposer un fichier à la place.

---

# Le Porte-Parole

Presse arabe en direct (Maroc, Algérie, Liban, Arabie saoudite, Qatar) traduite en français, et vidéos des chaînes
officielles sous-titrées en français par le modèle de parole du projet.

| Onglet | Ce qu'il fait | Modèle |
|---|---|---|
| Presse | titres et résumés des flux RSS, arabe à droite, français à gauche | NLLB-200 (traduction de texte, pré-entraîné) |
| Vidéos | extrait d'une vidéo YouTube officielle, sous-titres français, export .srt/.vtt | modèle de parole du projet (checkpoint requis) |
| Atelier, Comparer | un enregistrement personnel dans un ou plusieurs moteurs | idem |
| Coulisses | courbes et exemples de validation lus dans `log.jsonl` / `val.jsonl` | aucun |

## Lancer en local (design, sans GPU)
Double-clic sur `lancer.bat` (Windows) ou les commandes ci-dessous. Au démarrage, un fil de fond relit les flux toutes les 10 min,
traduit titres et résumés par petits lots, et garde en mémoire le texte complet des 12 premiers articles : la page s'affiche
donc tout de suite avec ce qui est prêt, et le français arrive au fil de l'eau. Ouvrir le site 10 minutes avant une démonstration.
    pip install -r requirements.txt
    python app.py                       # http://127.0.0.1:7860
    PORTEPAROLE_MT=off python app.py    # sans traduction NLLB des articles
Sans checkpoint, les moteurs sont des démonstrations (texte fictif, signalé) et la traduction de vidéos est refusée.

## Lancer avec le système S3 de l'équipe
Cloner la branche `benchmarking` de SamiNakibETU/NLP, puis :

    set PORTEPAROLE_S3_REPO=chemin\vers\NLP
    set HF_TOKEN=...                  # encodeur Cohere Transcribe Arabic, à accès contrôlé
    python app.py

Deux moteurs apparaissent : **S3, via le texte arabe** (cascade : transcription arabe puis traduction du texte, c'est celle qui produit
du français aujourd'hui, moteur par défaut) et **S3, français direct (essai)** (audio projeté + « Traduis en français. », sans texte arabe).
Avec le projecteur du temps 1, le direct recopie l'arabe (leur notebook `s3_tester_audio.ipynb` enregistre « arabe dans la sortie : 100 % ») :
l'interface affiche alors un avertissement. Le vrai français direct demande le temps 2. Le projecteur est celui du dépôt
(`modeles/s3_temps1_50k/`), ou `PORTEPAROLE_S3_CKPT`. Il faut un GPU (CUDA ou MPS) : sur processeur seul, l'encodeur de 1,9 milliard de paramètres est très lent.

## Lancer avec mes projecteurs (Colab GPU)
Notebook `speech_app_colab.ipynb` : il décompresse `speech_xattn.zip` (code du modèle) et `speech_app.zip`, monte Drive,
lit les `last.pt` et lance l'application avec un lien public temporaire. Variables : `PORTEPAROLE_CODE`,
`PORTEPAROLE_CKPT_PROJ`, `PORTEPAROLE_CKPT_XATTN`, `PORTEPAROLE_SHARE=1`.

## Fichiers
`app.py` interface · `feeds.py` flux RSS/Atom et cache · `textmt.py` NLLB et cache de traductions · `clip.py` extrait de vidéo
(yt-dlp + ffmpeg) · `model_engine.py` moteur réel (fenêtres de 25 s, sous-titres) · `engines.py` moteurs de démonstration et
export .srt/.vtt · `render.py` HTML · `plots.py` journal d'entraînement · `style.css` charte.
Tests : `pytest` (le test du moteur réel exige torch et le dossier `speech_xattn`).

## Limites et droits
- On ne reprend que ce que les flux publient (titre, résumé, lien) : pas d'articles complets. Chaque entrée renvoie au journal.
- Récupérer l'audio d'une vidéo en ligne contrevient aux conditions de YouTube et aux droits des diffuseurs : essai local de
  recherche seulement, sans republication de la vidéo ni des sous-titres.
- Le moteur découpe l'audio en fenêtres fixes de 25 s, sans détection de silence : un mot peut être coupé à la frontière.
- Corpus MGB2 (QCRI) : recherche non commerciale, aucun extrait n'est inclus.

## Édition figée (site du prof, sans machine allumée)
Un site qui tourne sur un PC n'est joignable que PC allumé. Pour qu'il s'ouvre à tout moment, on fige une édition :
1. `python snapshot.py build --out snapshot` : relit les flux, traduit titres, résumés et le texte complet de quelques articles
   (idéalement sur Colab GPU), et copie les journaux d'entraînement (`--logs`).
2. Les sous-titres français de quelques vidéos, produits par le modèle, s'ajoutent avec `snapshot.save_video_result`
   (cellules du notebook Colab). On ne garde que les sous-titres ; la vidéo reste lue depuis YouTube.
3. `python make_space.py --snapshot snapshot --out space` : dossier prêt pour Hugging Face Spaces (CPU gratuit, sans modèle).
4. En local, pour essayer : `set PORTEPAROLE_SNAPSHOT=snapshot` puis `python app.py`.
Dans ce mode : aucun réseau, aucun modèle, pas d'Atelier ni de Comparer, la page indique la date de l'édition. Clic sur un
horodatage = la vidéo se place à cet instant. Garder l'espace **privé** : l'archive contient des textes de presse.
