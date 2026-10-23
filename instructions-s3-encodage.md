# Instructions pour l'agent : système S3, préparation des données et encodage audio

Ce document décrit le travail à faire dans le dépôt NLP. Le lire en entier avant de commencer.
Le travail se fait par étapes : terminer et vérifier chaque étape, puis faire un compte rendu
à l'utilisateur avant de passer à la suivante.

## 0. Règles
- Écrire en français : code, commentaires, messages, journal.
- Ne jamais présenter un chiffre comme vérifié s'il ne l'est pas. Écrire « estimé » ou « NON vérifié ».
- Aucun chemin ni seuil codé en dur : tout passe par un YAML de `configs/`. Les seuils du plan de données
  sont dans `code/donnees/seuils.yaml`, qu'on lit sans le modifier.
- L'audio, les textes et les vecteurs MGB-2 ne vont jamais dans git (licence de recherche QCRI).
  Vérifier `.gitignore` avant chaque commit.
- Chaque script a un mode `--essai` (petites données factices, sans GPU ni réseau) et des tests
  `unittest` dans le dossier `tests/` de son module. En mode réel, un fichier ou un modèle manquant
  provoque une erreur explicite, sans repli silencieux vers des données factices.
- Ne rien supprimer sans demander. Demander avant tout changement d'architecture qui n'est pas décrit ici.
- Tenir un journal dans `code/projecteur/journal_s3.md` : ce qui a été exécuté, les chiffres mesurés,
  et ce qui reste NON vérifié.

## 1. Contexte : ce que l'on construit (S3)
Chaîne : parole arabe → encodeur audio gelé → projecteur entraîné → LLM `Qwen/Qwen2.5-1.5B-Instruct`
→ texte (transcription arabe au temps 1, traduction française au temps 2).

Deux changements par rapport à la configuration actuelle du dépôt :
1. **LLM : `Qwen/Qwen2.5-1.5B-Instruct`** à la place de `Qwen/Qwen3-4B` pour S3.
2. **Encodeur : on n'utilise plus Whisper.** Whisper impose des fenêtres de 30 s complétées par du silence.
   Pour un segment moyen de 10 s, environ 2/3 du calcul et de la sortie portent sur ce silence.
   On prend l'encodeur Conformer de **Cohere Transcribe Arabic** : il accepte des longueurs variables,
   sans complétion fixe (voir § 3).

### Code à reprendre
Le code de référence est dans un autre dépôt, déjà écrit et testé avec de petits modèles :
`/Users/nicolaswujczak/Desktop/Molecule/CY TECH/Projet NLP/MLP/nlp_project_mlp_cross_attention/projecteur_v1/`

| Fichier source | Rôle | Adaptation |
|---|---|---|
| `speech_llm/projector.py` | `AudioProjector` : LayerNorm, 2 × (Conv1d noyau 3, pas 2, GELU), MLP `d_enc → 2·d_llm → d_llm`, longueurs utiles | Le nombre de convolutions de pas 2 devient un paramètre du YAML (`nb_reductions`), calculé à partir de la fréquence réelle de l'encodeur (§ 3) |
| `speech_llm/model.py` | `SpeechLLM` : assemblage `[préfixe][audio][consigne][cible+EOS]`, labels à -100 hors cible, rembourrage à droite pour l'entraînement et à gauche pour la génération, `generate` à partir de `inputs_embeds` | Entrée = vecteurs d'encodeur **précalculés** (liste de `(T_i, d_enc)`), au lieu des features Whisper. Retirer `_encode` et la dépendance à Whisper |
| `speech_llm/training.py` | Temps 1 : projecteur seul. Temps 2 : projecteur + LoRA (r 16, alpha 32) | Paramètres LoRA lus dans `configs/s3.yaml` |
| `speech_llm/tiny.py` | Petits modèles aléatoires pour les tests | Remplacer le petit `WhisperEncoder` par une petite version du nouvel encodeur |
| `tests/test_miniature.py` | 7 tests : longueurs, rembourrage, labels, gradients aux temps 1 et 2, encodeur gelé, génération | Convertir en `unittest` (convention du dépôt), sans perdre un seul contrôle |
| `speech_llm/audio.py` | Lecture d'un passage `start`/`end` | Garder la lecture par passage. Retirer la limite de 30 s et les features Whisper |

Où mettre ce code : dans `code/projecteur/`, sous de nouveaux fichiers. Exemples de noms :
`projecteur_s3.py`, `modele_s3.py`, `etapes_s3.py`, `factices_s3.py`.
L'existant, `architecture_s3.py` et `entrainement_s3.py`, est un **prototype** : `DecodeurLLMPourInputsEmbeds`
est un faux décodeur, et les exemples sont des vecteurs aléatoires. Ne pas le supprimer : proposer à
l'utilisateur de le retirer une fois le nouveau code testé. Les fonctions utiles de `code/ecoute/encodeur.py`
(`lire_dimension_encodeur`, `lire_dimension_llm`) peuvent être réutilisées.

Mettre à jour `configs/s3.yaml` : `modele_llm_id`, `modele_ecoute_id`, `nb_reductions`, et retirer les
champs propres à Whisper (`duree_fenetre_encodeur_s`, `debit_encodeur_hz` fixe). Le gabarit de prompt
ChatML de `model.py` a été écrit à la main : le comparer à `tokenizer.apply_chat_template` de Qwen2.5,
et noter le résultat dans le journal.

## 2. Données sources : train MGB-2 (vérifié le 22/10/2026)
Dossier actuel : `/Users/nicolaswujczak/Downloads/train` (132 Go). L'utilisateur pourra le déplacer dans
`donnees/externe/mgb2_train/`, qui est déjà ignoré par git. **Le code ne doit dépendre que de la clé
`racine_mgb2` du YAML**, et les manifestes stockent des chemins **relatifs** à cette racine
(`wav/<episode>.wav`). Ainsi, déplacer le dossier ne casse rien.

Contenu :
- `wav/<episode>.wav` : 2214 émissions, PCM 16 bits, mono, 16 kHz. Aucun wav ne manque par rapport aux XML.
- `xml/utf8/<episode>.xml` : utiliser celui-ci, pas `xml/bw` qui est en Buckwalter. Chaque `<segment>` a les attributs
  `starttime` et `endtime` (en secondes), `WMER` (écart en % entre le texte et l'audio), `PMER`, `AWD`
  (débit de parole) et `who` (locuteur). Les mots sont dans `<element type="word">`. L'en-tête `<recording>` donne
  `filename`, `date` et `title`.
- `original_txt/` et `lm_text/` : pas utilisés ici.

Mesures déjà faites :
- 376 011 segments au total, soit 1128 h.
- Aucun chevauchement entre segments (sur 5296 paires dans 30 fichiers tirés au hasard).
- Certains segments durent jusqu'à 206 s.
- Après les filtres du plan (durée de 5 à 25 s, au moins 6 mots, WMER ≤ 15) et le découpage par épisode à 5 % :
  234 981 segments (682 h) pour le train et 14 770 segments (43 h) pour le test.
- La limite de 200 segments par épisode ne change presque rien.

Une fonction de lecture existe déjà : `lire_xml_mgb2` dans `code/traduction/benchmark_traduction.py`.
La réutiliser, et proposer de la déplacer dans `code/donnees/` plutôt que de la copier.

## 3. Encodeur : Cohere Transcribe Arabic
Modèle : **`CohereLabs/cohere-transcribe-arabic-07-2026`**, une version spécialisée arabe (dialectes compris) de
`CohereLabs/cohere-transcribe-03-2026`. Licence Apache 2.0. Le modèle est **à accès restreint** : il faut accepter
les conditions sur Hugging Face et mettre `HF_TOKEN` dans `.env`. Il est pris en charge nativement par
`transformers>=5.4.0` (classe `CohereAsrForConditionalGeneration`). Le modèle de base est déjà listé dans
`configs/choix_briques.yaml`.

Ce que disent les sources (recherche du 22/10/2026, NON vérifié en exécution) :
- Architecture : 2B paramètres en tout. Un grand encodeur Conformer (type Parakeet/FastConformer), qui représente presque
  tout le modèle (environ 7,6 Go en fp32, donc environ 3,8 Go en fp16), suivi d'un petit décodeur Transformer (8 couches) qu'on n'utilise pas.
- Entrée : log-Mel à 128 bandes, avec un pas de 160 échantillons, soit 100 trames par seconde à 16 kHz.
- **Réduction temporelle ×8 dans l'encodeur, donc 12,5 vecteurs par seconde** en sortie.
- Largeur : 1280 dans l'encodeur, puis une projection `1280 → 1024` vers le décodeur. Garder la sortie **native**
  de l'encodeur, avant cette projection, et lire `d_enc` dans la config au lieu de l'écrire en dur.
- **Longueur variable** : le lot est complété seulement jusqu'au plus long segment du lot (`padding="longest"`),
  avec un `attention_mask`. Il n'y a pas de fenêtre fixe de 30 s.
- Le préparateur de features découpe d'office les sons de plus de 35 s (`max_audio_clip_s`). Nos segments
  font 25 s au plus, donc ce découpage ne doit jamais se déclencher : le vérifier par un test.
- Sur MGB-2, le modèle complet obtient 15,5 % de WER (Open Universal Arabic ASR Leaderboard).

Utilisation :
- Charger le modèle, extraire son encodeur (`get_encoder()` ou l'attribut équivalent : vérifier dans le code
  de `transformers`), le geler et le mettre en `eval`. Libérer le décodeur.
- Appeler le préparateur avec `return_attention_mask=True`. Donner la langue (`language="ar"`) si le préparateur l'exige.

À **mesurer** avant tout encodage en masse, et à noter dans le journal :
- `d_enc` ;
- la fréquence réelle (nombre de vecteurs divisé par la durée, sur 10 segments de durées différentes ;
  12,5 est attendu) et la formule exacte du nombre de vecteurs utiles en fonction du nombre de trames ;
- la vitesse et la mémoire sur l'appareil réellement utilisé.

Conséquence pour le projecteur : à 12,5 Hz, le LLM reçoit déjà le bon débit, donc **`nb_reductions: 0`**.
Le projecteur se réduit à LayerNorm suivi du MLP `d_enc → 2·d_llm → d_llm` (les convolutions de pas 2 ne servent plus).
Si la fréquence mesurée n'est pas 12,5, s'arrêter et en parler à l'utilisateur.

Traitement par lots : regrouper des segments de durées voisines (moins de calcul perdu en complétion), passer
le masque, et ne garder que les vecteurs utiles de chaque segment. Test obligatoire : un segment encodé seul et
le même segment encodé dans un lot donnent les mêmes vecteurs utiles, aux arrondis fp16 près.

## 4. Configuration : `configs/encodage_s3.yaml`
```yaml
sources:
  racine_mgb2: "/Users/nicolaswujczak/Downloads/train"   # ou donnees/externe/mgb2_train après déplacement
  sous_dossier_xml: "xml/utf8"
  sous_dossier_wav: "wav"
sortie:
  dossier: "donnees/externe/s3_encodage"                 # ignoré par git
selection:
  graine: 42
  pct_episodes_test: 5            # même hachage MD5 que determiner_partie (code/donnees/pipeline_mgb2.py)
  nb_segments_train: 5000         # pilote (§ 4 bis) ; puis 50000
  nb_segments_test: 1000
  # durée, nombre de mots, WMER et segments max par épisode : lus dans code/donnees/seuils.yaml
encodage:
  modele_id: "CohereLabs/cohere-transcribe-arabic-07-2026"
  langue: "ar"
  appareil: "auto"                # cuda, puis mps, puis cpu
  precision: "float16"
  secondes_audio_par_lot: 120     # somme des durées d'un lot de segments de durées voisines
  segments_par_fichier_npy: 1000
```

## 4 bis. Taille du train (décision de l'utilisateur, en deux temps)
Repères : 10,45 s en moyenne par segment après filtres, et 12,5 vecteurs × 1280 × 2 octets = environ 0,115 Go par heure
(estimé, à remplacer par la mesure).

| Segments | Heures | Vecteurs sur disque | Rôle |
|---|---|---|---|
| 5 000 | ≈ 15 h | ≈ 1,7 Go | **Pilote** : valider toute la chaîne, mesurer la vitesse, vérifier que la perte baisse |
| 50 000 | ≈ 145 h | ≈ 17 Go | **Cible du temps 1**. C'est l'objectif du plan, et les mêmes segments recevront `texte_fr` pour le temps 2 |
| 234 981 (tout) | ≈ 682 h | ≈ 78 Go | Seulement si le WER progresse encore nettement entre 15 h et 145 h |
| test : 1 000 | ≈ 3 h | ≈ 0,3 Go | Évaluation (épisodes jamais vus) |

Le tirage est **emboîté** : avec la même graine, les 5 000 segments du pilote font partie des 50 000.
Monter en taille ne réencode donc que les nouveaux segments.

## 5. Étape 1 : les manifestes `train.jsonl` et `test.jsonl`
Script : `code/donnees/manifeste_s3.py` (`--config`, `--essai`).
1. Lire tous les XML de `racine_mgb2/xml/utf8`.
2. Filtrer avec `seuils.yaml` : `duree_segment_min_s`, `duree_segment_max_s`, `mots_min_segment`,
   `ecart_ecoute_max_pct` (appliqué au WMER) et `segments_max_par_episode`. Compter les rejets par motif.
3. Découper par **épisode**, jamais par segment : chaque émission va entièrement en train ou entièrement en test.
4. Tirer au hasard, avec une graine fixe, `nb_segments_train` et `nb_segments_test` segments.
5. Écrire une ligne JSON par segment, en UTF-8 avec `ensure_ascii=False` :
```json
{"id": "00152FDA_..._utt_1_align", "episode": "00152FDA-5AB7-4B47-993C-AC8AC4D8C3AB",
 "wav": "wav/00152FDA-5AB7-4B47-993C-AC8AC4D8C3AB.wav", "debut": 5.84, "fin": 11.98, "duree_s": 6.14,
 "texte_ar": "يلجأ النظام السوري إلى استخدام ذخائر عنقودية ...", "nb_mots": 12,
 "wmer": 20.0, "pmer": 18.0, "awd": 1.63, "locuteur": "speaker1_align",
 "date": "2013/01/17", "titre": "mA wrA' Alxbr", "partie": "train"}
```
   Le texte arabe reste brut : `normalisation.py` ne sert qu'aux mesures. Le champ `texte_fr` sera ajouté plus tard,
   pour le temps 2.
6. Écrire `resume_manifeste.json` : segments, heures, épisodes et rejets par motif, pour chaque partie.

Critères de réussite :
- aucun épisode n'apparaît dans les deux parties ;
- toutes les durées sont dans les bornes ;
- le résumé de la sélection complète (avant le tirage) est proche de 234 981 segments et 682 h pour le train.

## 6. Étape 2 : lire la partie utile de l'audio
Fonction : `charger_extrait(entree, racine)`, qui appelle
`soundfile.read(racine/entree["wav"], start=round(debut*sr), stop=round(fin*sr), dtype="float32")`.
On ne lit que le passage du segment, jamais l'émission entière, qui fait souvent 25 min ou plus.
Erreur explicite si le fichier n'est pas en 16 kHz ou s'il n'est pas mono.

Tests :
- la longueur lue vaut `(fin - debut) × 16000`, à un échantillon près ;
- sur un wav factice de quelques secondes, l'extrait est bien le bon morceau.

## 7. Étape 3 : encodage et stockage
Script : `code/ecoute/encoder_s3.py` (`--config`, `--partie train|test`, `--essai`).
- Lire le manifeste, puis encoder par lots (§ 3). Chaque segment donne une matrice `(T_i, d_enc)`, sans aucune complétion.
- **Stockage**, dans `sortie/<partie>/` :
  - `vecteurs_<numéro>.npy` : matrice `(somme des T_i, d_enc)` en float16, qui contient les vecteurs de `segments_par_fichier_npy` segments mis bout à bout ;
  - `index.jsonl` : la ligne du manifeste, plus `fichier_npy`, `ligne_debut`, `nb_vecteurs`, `encodeur` (id du modèle)
    et `d_enc`. C'est le lien entre les vecteurs et le passage audio analysé.
  - Lecture : `np.load(fichier, mmap_mode="r")[ligne_debut : ligne_debut + nb_vecteurs]`.
  - Écrire chaque fichier dans un fichier temporaire, puis le renommer quand il est complet.
- **Reprise après coupure** : un `vecteurs_<numéro>.npy` présent, avec ses lignes d'index, est considéré comme fait.
  Au redémarrage, on reprend au premier fichier manquant.
- **Vérifications à la fin**, écrites dans `rapport_encodage.json` :
  - aucun NaN ni inf ;
  - la somme des `nb_vecteurs` est égale à la somme des tailles des fichiers ;
  - 3 segments tirés au hasard et réencodés seuls donnent les mêmes vecteurs, aux arrondis fp16 près ;
  - on note la fréquence mesurée (vecteurs par seconde), la vitesse (secondes d'audio encodées par seconde)
    et la taille réelle sur disque.
- Taille attendue : `fréquence × d_enc × 2 octets × durée`, soit environ 0,115 Go par heure à 12,5 Hz et 1280 valeurs.
  NON vérifié : calculer la valeur réelle sur les 100 premiers segments,
  la comparer à l'espace libre, et **s'arrêter avant de saturer le disque**.

## 8. Étape 4 : jeu de données pour le projecteur
`DatasetS3` lit `index.jsonl` et renvoie `(vecteurs (T, d_enc) en float32, texte_ar)`. La fonction de
regroupement en lot (`collate`) renvoie la liste des vecteurs et la liste des textes : `SpeechLLM` gère
lui-même le rembourrage et les longueurs.

Test de bout en bout avec les modèles factices : index factice, puis `DatasetS3`, puis `SpeechLLM`, puis la perte,
qui doit être finie, avec des gradients sur le projecteur seulement.

## 9. Après ces étapes (ne pas commencer sans accord)
- Entraînement réel du temps 1 (transcription arabe) avec Qwen2.5-1.5B-Instruct : écrire d'abord un plan
  (lieu d'exécution, fp16 ou bf16, taille de lot, durée estimée), puis le présenter à l'utilisateur.
- Traductions françaises (`texte_fr`) pour le temps 2.

## Tests
`python -m unittest discover -s code/<module>/tests`, pour chaque module touché. Tous les tests existants
doivent continuer à passer.
