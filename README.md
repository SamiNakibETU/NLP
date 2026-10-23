# Kalam (كلام)

Sous-titres français, en quasi temps réel, de journaux télévisés en arabe.

Kalam écoute un journal arabe (direct YouTube d'Al Jazeera, micro, fichier vidéo) et affiche des sous-titres français environ trois secondes après la parole. Le texte apparaît d'abord en gris (traduction rapide), puis il est remplacé par une traduction plus soignée. En fin de session, l'application produit un rapport : transcript par locuteur, résumé horodaté, exports SRT, Markdown et Word.

Projet de groupe du cours de NLP, CY Tech, octobre 2026.

Rapport : [rapport/rapport.pdf](rapport/rapport.pdf) (source LaTeX : [rapport/rapport.tex](rapport/rapport.tex)).

## Architecture

```text
 micro, lien YouTube/HLS,      ┌──────────────────────────── S3 : transcription de l'arabe ───────────────────────────┐
 fichier audio ou vidéo        │                                                                                      │
 ─────────────► FastAPI ─────► │ encodeur Cohere Transcribe Arabic ──► projecteur MLP ──► Qwen2.5-1.5B-Instruct       │
   (WebSocket, ffmpeg)         │ (gelé, 1,9 Md param.)                 (entraîné, 8,66 M)   (gelé)                    │
                               └──────────────────────────────────────────────┬───────────────────────────────────────┘
                                                                              │ texte arabe
                          ┌───────────────────────────────────────────────────┼───────────────────────────┐
                          ▼                                                   ▼                           ▼
                Opus-MT ar→fr : texte gris                 NLLB-200 1,3B : texte définitif      rapport de fin de session
                (pendant la phrase, 0,19 s)                (à la clôture, 0,82 s)               (locuteurs, relecture, résumé)
```

Chaque segment clos reçoit aussi une empreinte de voix ECAPA (speechbrain), si bien que l'affichage signale les changements de locuteur. Le détail des composants, des routes et des variables d'environnement est dans [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Résultats principaux

Mesures sur FLEURS `ar_eg` → `fr_fr` (267 phrases, GPU T4), sauf mention contraire. Sources et protocole : [docs/RESULTATS.md](docs/RESULTATS.md).

| Système | Entrée | WER (%) | BLEU | chrF++ | COMET |
|---|---|---:|---:|---:|---:|
| S0 : Whisper small + Opus-MT | parole | 46,41 | 11,81 | 35,04 | 0,548 |
| SeamlessM4T v2 large | parole | 8,58 | 17,38 | 46,22 | 0,690 |
| NLLB-200 distillé 1,3B | texte arabe de référence | n.a. | 31,26 | 55,19 | 0,826 |
| Opus-MT ar-fr | texte arabe de référence | n.a. | 24,31 | 49,58 | 0,761 |
| Qwen2.5-1.5B-Instruct + LoRA | texte arabe de référence | n.a. | 17,33 | 42,05 | 0,763 |
| S3 (transcription seule) | parole, test MGB-2 (1 000 segments) | 14,26 | n.a. | n.a. | n.a. |

S3 n'a pas de score sur FLEURS, car le kernel `kaggle/nlp-eval-s3` n'a pas été exécuté. En direct sur Al Jazeera (Colab T4, essai final avec NLLB), le texte définitif s'affiche avec un retard médian de 2,9 s (moyen 3,8 s).

## Tester en 10 minutes (Google Colab)

Il faut un compte Google et un compte Hugging Face qui a accepté les conditions du modèle [`CohereLabs/cohere-transcribe-arabic-07-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026).

1. Ouvrir [`speech_app/colab_live.ipynb` dans Colab](https://colab.research.google.com/github/SamiNakibETU/NLP/blob/integration/speech_app/colab_live.ipynb).
2. Exécution > Modifier le type d'exécution > GPU T4.
3. Ajouter le secret Colab `HF_TOKEN` (jeton Hugging Face en lecture) et activer son accès au notebook. `SCALEDOWN_API_KEY` est facultatif.
4. Exécution > Tout exécuter, puis ouvrir le lien `https://….trycloudflare.com/live` affiché (ou scanner le QR code).

La première exécution est la plus longue, le temps d'installer les paquets et de télécharger plusieurs Go de modèles. Le pas à pas, les démonstrations à essayer et le dépannage sont dans [docs/GUIDE.md](docs/GUIDE.md).

## Tester sans GPU

Le moteur factice rejoue quatre phrases arabes fixes. Il permet de parcourir l'interface et de tester l'API, mais ses sorties ne disent rien de la qualité des modèles.

```bash
pip install torch
pip install -r speech_app/requirements.txt fastapi "uvicorn[standard]" python-multipart pyyaml
cd speech_app
PORTEPAROLE_LIVE=factice python -m uvicorn api:app --port 8770
```

Sous Windows (PowerShell) : `$env:PORTEPAROLE_LIVE = "factice"; python -m uvicorn api:app --port 8770`. Ouvrir ensuite <http://127.0.0.1:8770/live>. Variante plus légère, sans l'interface Gradio : `python live.py --factice` (port 8765).

## Structure du dépôt

| Dossier | Contenu | Auteur principal | Branche d'origine |
|---|---|---|---|
| `speech_app/` | application Kalam : API FastAPI, pages web, direct, rapport, Docker, notebooks Colab | Paul Douat (socle), Sami Nakib (direct, rapport, interface) | `docker`, `integration` |
| `code/projecteur/`, `code/ecoute/` | S3 : encodage Cohere, projecteur, entraînement du temps 1, transcription | Nicolas Wujczak | `benchmarking` |
| `code/direct/` | segmentation du flux, double passe, mesure du retard | Sami Nakib, Nicolas Wujczak | `main`, `benchmarking` |
| `code/donnees/`, `code/mesures/`, `code/demo/`, `code/cascade/` | données MGB-2 et OPUS, métriques et bootstrap, baseline S0, cascade LoRA | Sami Nakib | `main` |
| `code/seamless/`, `code/traduction/` | benchmarks SeamlessM4T et traducteurs sur MGB-2 | Fatima Aguel | `benchmarking-projecteurv1` |
| `projecteur_v1/` | premier prototype de projecteur Whisper vers Qwen | équipe | `projecteurv1` |
| `notebooks/` | notebooks d'analyse ; `embeddinggemma_*` et `cometkiwi_comparaison` : Hanine Attia | équipe | toutes |
| `kaggle/` | kernels d'évaluation sur FLEURS (GPU T4) | Sami Nakib | `integration` |
| `resultats/` | mesures brutes : `kaggle/` (FLEURS), `direct/` (Al Jazeera) | Sami Nakib | `integration` |
| `modeles/s3_temps1_50k/` | projecteur S3 entraîné (35 Mo) et son évaluation | Nicolas Wujczak | `benchmarking` |
| `configs/` | réglages YAML (S0, S3, direct, traduction) | équipe | toutes |
| `rapport/` | rapport du projet (PDF, 5 pages, et source LaTeX) | Paul Douat | `integration` |
| `tp/` | travaux pratiques du cours (notebooks) | équipe | `integration` |

Sami Nakib a coordonné le projet et assuré l'intégration et l'évaluation finale. Paul Douat a rédigé le rapport.

## Documentation

- [rapport/rapport.pdf](rapport/rapport.pdf) : rapport du projet (méthode, résultats, analyse du LLM, limites).
- [docs/GUIDE.md](docs/GUIDE.md) : installation, démonstrations, dépannage, Docker, reproduction des mesures, tests.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) : composants, flux, API, messages WebSocket, variables d'environnement.
- [docs/RESULTATS.md](docs/RESULTATS.md) : tous les chiffres, leur source, le protocole et les limites.

## Licences et données

- MGB-2 (QCRI), SeamlessM4T v2 et NLLB-200 sont réservés à un usage de recherche non commercial. Les données MGB-2 (audio et transcriptions) ne sont pas versionnées.
- Le modèle Cohere Transcribe Arabic est à accès contrôlé : chacun doit accepter ses conditions sur Hugging Face.
- La police ABC Areal (Dinamo) n'est pas redistribuée. L'interface ne la charge que si elle est installée sur la machine (`local()` en CSS) ; sinon le navigateur utilise une police système.
- Les extraits d'Al Jazeera servent à un usage académique. Le résumé de fin peut passer par un service externe (ScaleDown) si une clé est fournie. Les sous-titres peuvent contenir des noms propres erronés : les relire avant toute diffusion.
- Le dépôt ne contient pas de fichier de licence pour le code.
