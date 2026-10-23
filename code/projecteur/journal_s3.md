# Journal S3 : préparation des données et encodage audio

Suivi du travail décrit dans `instructions-s3-encodage.md`. Chaque chiffre est marqué
« mesuré » (exécuté ici) ou « NON vérifié ».

## Environnement
- Aucun environnement Python du dépôt n'avait les dépendances (`python3` 3.14 système sans `yaml`
  ni `numpy`). Les étapes ci-dessous ont été exécutées avec un venv temporaire Python 3.13.
  Il contenait `pyyaml`, `numpy` 2.5.4 et `soundfile` 0.14.0.
- État des tests avant toute modification (mesuré) : `code/donnees/tests` donne 86 tests et 3 erreurs.
  Toutes viennent d'imports absents : `nbformat` et `py3langid`. `code/traduction/tests` donne
  17 tests OK. `code/projecteur/tests` et `code/ecoute/tests` ne se chargent pas (`torch` absent).
- Espace libre sur le disque (mesuré le 22/10/2026) : 522 Gio.

## Étape 1 : manifestes (22/10/2026)
Fichiers : `configs/encodage_s3.yaml`, `code/donnees/manifeste_s3.py`,
`code/donnees/tests/test_manifeste_s3.py` (12 tests, OK).

Commande : `python code/donnees/manifeste_s3.py --config configs/encodage_s3.yaml` (8 s).
Sortie : `donnees/externe/s3_encodage/{train,test}.jsonl` et `resume_manifeste.json`.
Ce dossier est ignoré par git (vérifié avec `git check-ignore`).

Choix de mise en œuvre :
- Filtres dans l'ordre durée, puis mots, puis WMER, puis plafond par épisode. Les bornes sont incluses
  (5 ≤ durée ≤ 25, mots ≥ 6, WMER ≤ 15). Un rejet compte pour son premier motif.
- `nb_mots` : décompte de `lire_xml_mgb2`, qui ne compte pas la ponctuation seule.
- Plafond de 200 segments par épisode : quand un épisode dépasse, on tire au hasard, avec une graine
  propre à l'épisode, au lieu de garder les 200 premiers.
- Découpage : `determiner_partie` (MD5, 5 %). « validation » y devient « test ».
- Tirage emboîté : la liste triée par id est mélangée avec la graine, puis on garde le début.
  L'ordre des lignes du manifeste est l'ordre du tirage. Les 5 000 lignes du pilote seront donc
  les 5 000 premières du manifeste à 50 000.

Résultats (mesurés) :

| | train | test |
|---|---|---|
| segments lus | 354 043 (1061,9 h, 2086 épisodes) | 21 968 (66,1 h, 128 épisodes) |
| rejets : durée | 103 593 | 6 265 |
| rejets : mots | 1 723 | 104 |
| rejets : WMER | 13 747 | 829 |
| rejets : plafond par épisode | 2 753 | 194 |
| sélection complète | **232 227 (675,4 h)** | **14 576 (42,3 h)** |
| tirage | 5 000 (14,46 h, 1835 épisodes) | 1 000 (2,85 h, 127 épisodes) |

- Total lu : 376 011 segments, ce qui correspond à la mesure annoncée.
- Sans le plafond par épisode : 234 980 segments en train (annoncé : 234 981) et 14 770 en test (identique).
  L'écart d'un segment en train n'est PAS expliqué. Ce n'est ni l'arrondi des durées
  (0 segment ne change de côté) ni le comptage des mots. Le plafond retire 1,2 % du train (2 753 segments, environ 7 h).
- Durée moyenne mesurée sur le tirage : 10,41 s en train et 10,26 s en test (repère des instructions : 10,45 s).

Vérifications indépendantes sur les fichiers écrits (mesurées) :
- 0 épisode commun entre train et test, et des ids uniques ;
- durées entre 5,0 et 24,98 s, mots ≥ 6, WMER ≤ 15 ;
- les 1835 + 127 wav existent et sont tous en 16 kHz, mono, PCM 16 bits ;
- sur les 300 premiers segments de chaque partie, aucun ne dépasse la fin de son wav.

## Environnement du dépôt (22/10/2026)
- `.venv/` (Python 3.13, uv), ajouté au `.gitignore`. Versions : torch 2.14.1, transformers 5.20.2, peft 0.21.2,
  numpy 2.5.4, soundfile 0.14.0, librosa 1.0.0. Les dépendances S3 sont dans `requirements.txt`.
  `requirements-fige.txt` (transformers 4.52.4) n'a PAS été modifié : il sert aux autres modules.
- Ajouté aussi pour les tests existants de `code/donnees` : `nbformat`, `nbclient`, `ipykernel`, `py3langid`.
- `HF_TOKEN` dans `.env` (compte nwujczak). Le modèle arabe est accessible.
  `CohereLabs/cohere-transcribe-03-2026` (le modèle de base) renvoie 403 : on n'en a pas besoin.

## Code repris du dépôt MLP (§ 1)
| Nouveau fichier | Source | Changements |
|---|---|---|
| `code/projecteur/projecteur_s3.py` | `speech_llm/projector.py` | `nb_reductions` paramétrable (0 pour S3) ; **correction** : le rembourrage est remis à zéro avant chaque convolution (voir plus bas) |
| `code/projecteur/modele_s3.py` | `speech_llm/model.py` | entrée = liste de vecteurs précalculés `(T_i, d_enc)` ; plus d'encodeur ni de Whisper ; message système et consignes lus dans `s3.yaml` |
| `code/projecteur/etapes_s3.py` | `speech_llm/training.py` | LoRA lu dans `s3.yaml` (`temps_2.lora`) |
| `code/projecteur/factices_s3.py` | `speech_llm/tiny.py` | petit `CohereAsrForConditionalGeneration` (encodeur Parakeet) au lieu du petit `WhisperEncoder` |
| `code/ecoute/audio_s3.py` | `speech_llm/audio.py` | `charger_extrait` : lecture du seul passage, sans limite de 30 s ni features Whisper |
| `code/projecteur/tests/test_modele_s3.py` | `tests/test_miniature.py` | converti en unittest ; les 7 contrôles sont gardés (correspondance en tête du fichier) |

- Le contrôle « encodeur gelé et en eval » est passé dans `code/ecoute/tests/test_encoder_s3.py`, parce que
  l'encodeur n'est plus dans `SpeechLLMS3`.
- Cibles LoRA : le dépôt MLP visait aussi `gate_proj`, `up_proj` et `down_proj`. Ici, on suit `s3.yaml`
  (`q_proj`, `k_proj`, `v_proj`, `o_proj`). À trancher avant le temps 2.
- **Défaut trouvé dans le projecteur de référence** (test `test_rembourrage_sans_effet_sur_un_segment`) :
  avec `nb_reductions > 0`, les positions de rembourrage (après LayerNorm, elles valent le biais et non 0)
  entraient dans la convolution. Le dernier vecteur d'un segment changeait donc entre seul et en lot.
  C'est corrigé par un masque avant chaque convolution. Sans effet pour S3 (`nb_reductions: 0`).
- `configs/s3.yaml` : `modele_llm_id` vaut Qwen/Qwen2.5-1.5B-Instruct et `modele_ecoute_id` le modèle Cohere arabe.
  Ajout de `nb_reductions: 0`, `max_tokens_cible: 256` et `consignes.systeme`. Retrait de `duree_fenetre_encodeur_s`,
  `debit_encodeur_hz` et `facteur_regroupement_k` (le regroupement par 4 de Whisper est remplacé par `nb_reductions`).
  Aucun code ne lisait ces clés.
- **Gabarit ChatML** (mesuré, et testé par `test_gabarit_identique_a_apply_chat_template`) :
  `prefixe + audio + suffixe` est identique, caractère pour caractère, à `apply_chat_template` de Qwen2.5,
  avec le message système « You are a helpful assistant. » comme avec le message par défaut de Qwen.
  Sans message système, Qwen2.5 ajoute « You are Qwen, created by Alibaba Cloud... ». EOS = `<|im_end|>` (151645).
  Le préfixe fait 14 tokens. Le suffixe fait 12 tokens au temps 1.
- Projecteur S3 : LayerNorm(1280), puis Linear 1280 → 3072, GELU, Linear 3072 → 1536. Cela fait 8 657 920 paramètres
  (mesuré avec `nb_parametres()` ; d_llm = 1536 lu dans la config de Qwen2.5-1.5B).

## Encodeur Cohere Transcribe Arabic (§ 3) : mesures avant encodage en masse
Lecture du code de transformers 5.20.2 :
- `CohereAsrForConditionalGeneration.get_encoder()` renvoie un `ParakeetEncoder`. La projection 1280 → 1024
  (`decoder.proj`) est dans le décodeur : on garde bien la sortie native de l'encodeur.
- La sortie de l'encodeur contient un `attention_mask` sous-échantillonné. Le nombre de vecteurs utiles est sa somme.
- Le préparateur découpe les sons au-delà de `max_audio_clip_s - overlap_chunk_second` = 35 - 5 = **30 s**
  (et non 35 s). C'est sans effet pour nos 25 s max, et c'est testé (`test_pas_de_decoupage_a_25_s`).
  Un découpage éventuel lève une erreur.
- Le préparateur n'a pas besoin de la langue (`language` ne sert qu'au prompt du décodeur).
- Le dither est déterministe : la graine est la longueur utile, donc il ne dépend pas de la composition du lot.
- `preprocessor_config.json` du modèle ne contient aucun réglage : on garde ceux par défaut de la classe.
  Ils correspondent à la section `preprocessor` de `config.json` (128 bandes, n_fft 512, fenêtre 400, pas 160, dither 1e-5).

Mesures (Apple M5 Pro, 24 Go, MPS, float16) :
- Chargement : 0 clé manquante, 0 clé inattendue. Encodeur : 1 895,3 M paramètres (modèle complet : 2 065,6 M).
- Transcription témoin avec le modèle complet, sur 2 segments : quasi identique à la référence. Les poids sont bons.
- **d_enc = 1280.**
- **Formule exacte** : trames = n_échantillons // 160 ; puis 3 fois L → (L - 1) // 2 + 1 ;
  soit ceil(ceil(ceil((n // 160) / 2) / 2) / 2). Elle est vérifiée sur les 10 segments.
- **Fréquence** sur 10 segments de 5,0 à 24,78 s : de 12,50 à 12,60 Hz, et 12,53 Hz au total
  (l'arrondi par excès rajoute moins d'un vecteur par segment). **12,5 Hz confirmé, `nb_reductions: 0`.**
- Segment seul contre segment dans un lot : mêmes formes, écart max de 0,0044 pour des valeurs jusqu'à 1,5,
  soit quelques arrondis fp16. Tolérance retenue dans `encodage_s3.yaml` : |a - b| ≤ 0,01 + 0,01·|b|.
- Vitesse : 194,5 s d'audio par seconde (200 segments, lots de 120 s). Mémoire MPS : 3,8 Go après chargement,
  5,6 Go au maximum.
- Taille : 0,1156 Go par heure d'audio (estimation du plan : 0,115).

## Étape 3 : encodage du pilote (22/10/2026)
Commandes : `python code/ecoute/encoder_s3.py --partie train`, puis `--partie test` (MPS, float16, lots de 120 s).
Résultats dans `donnees/externe/s3_encodage/{train,test}/` (ignoré par git).

| | train | test |
|---|---|---|
| segments | 5 000 (14,46 h) | 1 000 (2,85 h) |
| fichiers npy | 5 | 1 |
| vecteurs | 653 044 | 128 733 |
| fréquence mesurée | 12,542 Hz | 12,543 Hz |
| valeurs NaN / inf | 0 | 0 |
| somme nb_vecteurs = lignes des fichiers | oui | oui |
| segments hors formule | 0 | 0 |
| taille sur disque | 1,67 Go (0,1156 Go/h) | 0,33 Go (0,1156 Go/h) |
| vitesse | 247 s d'audio/s (3 min 31 s) | 251 s d'audio/s (41 s) |
| réencodage seul de 3 segments | 3/3 conformes | **2/3 conformes** |

Écart sur le test (`EF138139_..._utt_124_align`, 19,54 s) : 295 valeurs sur 313 600 dépassent la tolérance,
toutes sur **un seul vecteur** (n° 184). L'écart max est de 0,034 et le cosinus min de 0,9983. Analyse (mesurée) :
- comparé à une référence fp32, le vecteur **stocké** (fp16 en lot) a un écart max de 0,0056, sans aucune valeur hors tolérance ;
- c'est l'encodage fp16 du segment **seul** qui s'écarte de 0,035 au vecteur 184 ;
- sur 30 autres segments du test, réencodés seuls : 0 valeur hors tolérance sur 5,46 M (cosinus min 0,99974).
Conclusion : les vecteurs stockés sont bons. Le critère valeur par valeur réagit à une instabilité fp16 ponctuelle
du réencodage. Le rapport du test reste à `conforme: false` tant que le critère n'est pas revu.

Repère de durée (estimé à partir de 247 s d'audio/s) : 50 000 segments ≈ 145 h, soit environ 35 min d'encodage
et environ 17 Go sur le disque.

## Étape 4 : DatasetS3 (22/10/2026)
`code/projecteur/donnees_s3.py` : `DatasetS3` et `regrouper_lot`. Test factice dans `test_modele_s3.py`.
Essai réel (script hors dépôt) : index réel du train, puis `DatasetS3`, le vrai Qwen2.5-1.5B-Instruct en fp16 sur MPS, et le temps 1.
- 8 657 920 paramètres entraînables (projecteur seul) ; le LLM ne reçoit aucun gradient.
- Lot de 4 (195, 85, 86, 133 vecteurs audio). Pertes sur 3 pas d'AdamW (lr 1e-4), même lot : 3,59, puis 3,51, puis 3,39.
  Environ 0,7 s par pas, et 11,7 Go de mémoire MPS au maximum.
- Ce n'est PAS un entraînement : la baisse sur un seul lot montre seulement que les gradients circulent.

## Tests (22/10/2026, .venv)
- `code/donnees/tests` : 107 tests OK (1 ignoré).
- `code/ecoute/tests` : 19 tests OK.
- `code/projecteur/tests` : 20 tests OK.
- `code/traduction/tests` : 17 tests OK (1 ignoré).

## Temps 1, pilote : optimisation de l'entraînement (22/10/2026)
Machine : Apple M5 Pro, 24 Go de mémoire unifiée, MPS. Banc d'essai : 8 pas après un pas de chauffe,
lots tirés du train (scripts hors dépôt).

- **Perte calculée sur la cible seule** (`SpeechLLMS3.forward`) : la tête de sortie (vocabulaire de 151 936)
  n'est plus appliquée qu'aux positions qui prédisent un token de la cible, et non aux environ 300 positions
  de la séquence. La perte est identique à celle de `llm(labels=...)` (test `test_perte_identique_a_celle_du_llm`).
  On évite des logits de taille lot × 300 × 151 936.
- Mémoire : les paramètres du LLM sont gelés, mais le gradient doit traverser ses 28 couches pour atteindre
  le projecteur. Toutes les activations restent donc en mémoire. Un lot de 8 (bf16, sans checkpointing) occupe
  20,6 Go côté pilote MPS et fait swapper la machine (2,7 segments/s). Un lot de 16 dépasse la mémoire (29 Go).
- Mesures (segments/s, lots de longueurs voisines) :

| précision LLM | micro-lot | checkpointing | segments/s | s/pas | pilote MPS |
|---|---|---|---|---|---|
| fp16 | 4 | non | 9,2 | 0,43 | 12,3 Go |
| bf16 | 4 | non | 9,2 | 0,44 | 12,2 Go |
| fp16 | 8 | oui | 7,0 | 1,14 | 7,4 Go |
| bf16 | 8 | oui | 7,2 | 1,11 | 7,4 Go |
| fp16 | 16 | oui | 6,8 | 2,37 | 9,7 Go |
| bf16 | 16 | oui | 6,8 | 2,36 | 11,4 Go |
| bf16 | 8 | non | 2,7 (swap) | 3,01 | 20,6 Go |

- Choix : **bf16, micro-lot de 4 sans checkpointing, accumulation de 4** (lot effectif de 16 = `temps_1.lot_effectif`).
  bf16 va aussi vite que fp16, avec la plage d'exposants du fp32 : pas de risque de dépassement dans les gradients
  qui traversent le LLM. Le checkpointing économise de la mémoire mais ralentit de 25 %.
- **Lots de longueurs voisines** (`lots_par_longueur` : seaux de 50 lots triés par longueur, puis ordre des lots
  mélangé) : moins de rembourrage, donc moins de calcul perdu.
- AdamW, montée linéaire, puis cosinus jusqu'à 0, gradients écrêtés à 1.
- `repetition_penalty` à 1,0 en génération : Qwen2.5 met 1,1 par défaut, ce qui pénalise les mots répétés,
  fréquents dans une transcription.
- Validation : 200 segments du train tirés au hasard et jamais appris (perte), dont 100 transcrits (WER/CER).
  Limite : ils peuvent venir des mêmes émissions que l'entraînement, donc le WER de validation est optimiste.
  Le test (épisodes jamais vus) ne sert qu'à la fin.

## Balayage du taux d'apprentissage sur le pilote (22/10/2026)
120 pas de 16 segments par essai, même initialisation du projecteur, perte de validation sur 200 segments.
La perte vaut 4,246 avant entraînement.

| taux | 1e-4 | 3e-4 | 1e-3 | 3e-3 |
|---|---|---|---|---|
| perte de validation | 3,404 | 3,321 | 3,319 | 3,309 |

- Choix : **1e-3**. Il est à 0,01 du meilleur, et un taux plus grand risque davantage sur un entraînement long.
- Les trois taux les plus grands finissent sur le même palier (environ 3,3). En fin d'essai, la norme du gradient
  tombe à 0,15 (contre 2 à 8 au début).
- **WER après 120 pas (taux de 1e-3), sur 100 segments de validation : 247,9 % (CER 208,0 %).** Les sorties sont de
  l'arabe de journal correct mais sans rapport avec l'audio, ou des boucles (« الله يحفظكم » répété).
  Le projecteur ne transmet pas encore l'audio : c'est le palier de démarrage d'un projecteur seul.
- L'entraînement complet du pilote a été lancé puis arrêté à la demande de l'utilisateur, avant la première mesure.

## Passage aux 50 000 segments (23/10/2026)
- `nb_segments_train: 50000`. Le manifeste a été régénéré : 50 000 segments, 144,88 h, 2 086 émissions.
  **Ses 5 000 premières lignes sont identiques au pilote** (vérifié avec `cmp`) : les 5 fichiers de vecteurs
  du pilote sont repris (« 5/50 fichier(s) déjà faits »).
- **4 segments sur 50 000 finissent après la fin de leur wav**, de 0,06 s au plus (rangs 5157, 5161, 25954 et 49153 ;
  0 dans le test). Le premier lancement s'est arrêté sur l'erreur explicite de `charger_extrait`.
  Correction : `controles.tolerance_fin_wav_s: 0.1`. Dans cette tolérance, on coupe à la fin du fichier ; au-delà,
  l'erreur reste. L'index note désormais `nb_echantillons` (échantillons réellement lus), que la vérification
  de la formule utilise. Test `test_tolerance_de_fin`.
- Entraînement : section `temps_1_50k` de `s3.yaml`. Mêmes réglages que le pilote, avec 2 passages (environ 6 200 pas),
  une mesure tous les 300 pas, une montée sur 200 pas et une patience de 4.
- Lancement : `caffeinate -i -s sh -c 'encoder_s3.py --partie train && entrainer_temps1.py --section temps_1_50k'`.
  Journal dans `modeles/s3_temps1_50k/journal.txt`. Durée estimée : environ 35 min d'encodage, puis environ 3 h d'entraînement.

## Temps 1 sur 50 000 segments : premiers résultats (23/10/2026)
Encodage des 50 000 segments : 144,88 h, 16,7 Go, 0 NaN, 0 segment hors formule, 12,542 Hz,
réencodage seul conforme (3/3). Le rapport est `conforme: true`.

Validation (100 segments transcrits, 200 pour la perte) :

| pas | 0 | 300 | 600 | 900 | 1200 | 1500 |
|---|---|---|---|---|---|---|
| perte de validation | 4,109 | 0,854 | 0,580 | 0,532 | 0,504 | 0,482 |
| WER (%) | 159,0 | 30,8 | 16,2 | 15,6 | 17,2 | 13,4 |
| CER (%) | 149,3 | 14,2 | 7,4 | 8,0 | 10,5 | 6,8 |

- Le palier de démarrage est franchi avant le pas 300 (le balayage du pilote s'arrêtait au pas 120).
- Vitesse : environ 8,8 à 9,3 segments/s.
- Entraînement arrêté au pas 1 730, à la demande de l'utilisateur, pour évaluer sur le test.
  Point de reprise : pas 1 500 (`modeles/s3_temps1_50k/reprise.pt`).

**Test (1 000 segments, émissions jamais vues, projecteur du pas 1 500) : WER 17,28 %, CER 9,66 %**
(mesuré ; 2,85 h d'audio transcrites en 124,5 s, RTF de 0,012 sur MPS, vecteurs précalculés).
- WER médian par segment : 10,5 %. 203 segments sur 1 000 sont transcrits sans erreur.
- 9 segments (moins de 1 %) partent en boucle (par exemple « بسم الله » répété jusqu'à la limite de 160 tokens),
  avec un WER par segment au-delà de 100 %. Les 20 pires segments font 25 % des erreurs.
  Sans eux, le WER est de 13,25 %.
- Repère : le modèle Cohere complet, 15,5 % sur MGB-2 (classement publié, NON vérifié ici).
- Pistes contre les boucles, sans réentraîner : limiter la longueur générée en fonction de la durée
  de l'audio, `no_repeat_ngram_size`, ou une `repetition_penalty` légère. NON testé.

## Anti-boucles et test sur un fichier audio (23/10/2026)
Deux mécanismes, réglés sur les **références du train** (pas sur le test) :
- **Longueur maximale par segment** : `ceil(7 × durée) + 10` tokens (`LimiteParSegment`, qui impose l'EOS).
  Sur 50 000 références : moyenne de 3,82 tokens/s, p99,9 de 8,39. 49 cibles seulement dépasseraient cette limite.
- **Nettoyage des boucles** (`retirer_boucles`) : une suite de 1 à 8 mots répétée au moins 4 fois de suite est
  ramenée à une seule occurrence. Cette règle modifie 2 références sur 50 000 (54 avec un seuil de 3, à cause
  de vrais « جداً جداً جداً », d'où le seuil de 4).
- Écarté : `no_repeat_ngram_size`. Un 6-gramme de tokens revient déjà dans 2,7 % des références (5 % pour un 5-gramme).

Test (1 000 segments, projecteur du pas 1 500) :

| génération | WER | CER | segments au-delà de 100 % | calcul |
|---|---|---|---|---|
| sans anti-boucles | 17,28 % | 9,66 % | 9 | 124,5 s |
| nettoyage seul (anciennes transcriptions) | 14,26 % | 7,29 % | | |
| **limite de longueur + nettoyage** | **14,26 %** | **7,22 %** | 0 | 115,6 s |

- Les anciens fichiers sont gardés : `evaluation_test_sans_antiboucles.json` et `hypotheses_test_sans_antiboucles.jsonl`.
- **Correction** : boucle infinie dans la première version de `retirer_boucles` (le curseur n'avançait plus
  en fin de texte). Elle a été repérée par un test bloqué à 100 % de CPU, puis corrigée.

Test sur un fichier audio quelconque : `code/projecteur/transcrire_s3.py` (`--audio`, `--srt`, `--essai`)
et `notebooks/s3_tester_audio.ipynb`. Le fichier est découpé en morceaux de 20 s au plus, coupés au point le plus calme
des 5 dernières secondes (section `transcription` de `s3.yaml`), puis passe par l'encodeur Cohere, le projecteur et Qwen.
- Exécution réelle (MPS) sur un extrait de 33 s du test MGB-2 (émission 1E084D4F, calé sur 5 segments entiers) :
  transcrit en 3,1 s, encodage compris ; **WER 16,13 %, CER 6,44 %** sur l'extrait.
  Les modèles se chargent en 9 s (déjà en cache).
- **Défaut corrigé dans le notebook** : la première version coupait l'extrait à 60 s pile. Elle excluait alors de la référence
  un segment qui dépassait la fin de l'extrait, alors que le modèle l'avait transcrit (WER faussé à 80,65 %).
  L'extrait est maintenant calé sur des segments entiers.
- Tests : `test_transcrire_s3.py` (découpage, chaîne factice, SRT, notebook en mode essai).
  `code/projecteur/tests` : 39 tests OK.

## Traduction française en cascade dans le notebook (23/10/2026)
`TranscripteurS3.traduire` traduit chaque morceau transcrit avec le même Qwen2.5-1.5B-Instruct, en mode texte.
Ses poids n'ont pas changé, puisque seul le projecteur a appris. Réglages : section `traduction_texte` de `s3.yaml`.
- Consignes comparées sur 5 textes arabes du test (lecture humaine, pas de score) :
  - A (système neutre + « Traduis en français le texte arabe suivant… ») : traduit, mais commence par
    « Voici la traduction en français : » et ajoute parfois un commentaire.
  - B (système traducteur + « Traduis en français : ») : 1 texte sur 3 reste en arabe.
  - C (système traducteur + texte seul) : la sortie reste en arabe (reformulation).
  - D (système traducteur + consigne longue avant le texte) : 3 textes sur 5 restent en arabe.
  - **E (système traducteur + texte puis « Traduis ce texte en français. ») : 5 textes traduits sur 5, sans introduction ;
    une phrase entre guillemets.** C'est la consigne retenue.
- Garde-fous : `nettoyer_traduction` retire « Voici… : » et les guillemets autour ; si de l'arabe reste dans la sortie,
  le texte est retraduit avec la consigne A (`gabarit_secours`).
- Exécution réelle sur l'extrait de test de 33 s (copie exécutée : `modeles/s3_temps1_50k/notebooks/`, ignorée par git) :
  2 morceaux traduits en 2,1 s, en français, sans méta-texte. L'erreur de transcription « الزيادة » (augmentation),
  au lieu de « الزينة » (décoration), devient « la quantité augmente » : la cascade propage les erreurs de l'ASR.
  Pas de score : MGB-2 n'a pas de traduction française de référence.
- Tests : 41 tests OK dans `code/projecteur/tests`.

## NON vérifié / en attente
- L'origine de l'écart d'un segment (234 980 contre 234 981).
- Le critère du réencodage seul (voir étape 3) : passer à un cosinus par vecteur, ou comparer à une référence fp32 ?
- Les cibles LoRA du temps 2 : attention seule (s3.yaml) ou attention + MLP (dépôt MLP) ?
- Déplacement proposé : `lire_xml_mgb2` de `code/traduction/benchmark_traduction.py` vers `code/donnees/`.
  Pour l'instant, `manifeste_s3.py` l'importe, ce qui charge aussi `code/mesures/scores.py`.
- Le prototype `architecture_s3.py` et `entrainement_s3.py` (faux décodeur) : on peut le retirer maintenant que le nouveau code est testé.
- Le plan d'entraînement réel du temps 1 (§ 9) : pas commencé.
