# Résultats

Tous les chiffres du projet, avec le fichier qui les contient. L'interprétation complète est dans le rapport : [rapport/rapport.pdf](../rapport/rapport.pdf).

## Sommaire

1. [Protocole](#1-protocole)
2. [Systèmes de la parole au français](#2-systèmes-de-la-parole-au-français)
3. [Traducteurs de texte](#3-traducteurs-de-texte)
4. [Adaptation LoRA de Qwen2.5-1.5B](#4-adaptation-lora-de-qwen25-15b)
5. [S3 : entraînement du projecteur](#5-s3--entraînement-du-projecteur)
6. [Mode direct sur Al Jazeera](#6-mode-direct-sur-al-jazeera)
7. [Rapport de fin de session](#7-rapport-de-fin-de-session)
8. [Le fichier resultats/resultats.csv](#8-le-fichier-resultatsresultatscsv)
9. [Limites](#9-limites)

## 1. Protocole

MGB-2, le corpus d'entraînement (journaux d'Al Jazeera), n'a pas de traduction française. C'est pourquoi la traduction est évaluée sur **FLEURS** : le test `ar_eg` (arabe lu) est apparié par identifiant de phrase au test `fr_fr`, en gardant la première lecture de chaque identifiant. On obtient 267 phrases et 49,4 minutes d'audio (2 961,1 s), chacune avec une traduction française humaine.

| Élément | Choix |
|---|---|
| Matériel | Kaggle, GPU T4, fp16 ; versions : torch 2.11.0+cu128, transformers 5.20.2 |
| BLEU, chrF++ | sacreBLEU 2.6.0, signatures `nrefs:1\|case:mixed\|eff:no\|tok:13a\|smooth:exp` et `nrefs:1\|case:mixed\|eff:yes\|nc:6\|nw:2\|space:no` |
| COMET | `Unbabel/wmt22-comet-da`, calculé à part (`kaggle/nlp-comet`) |
| WER, CER | transcription arabe, après normalisation de l'arabe (`code/donnees/normalisation.py`) |
| Écarts | bootstrap apparié, 300 tirages, graine 42, intervalle à 95 % ; un écart est jugé significatif si l'intervalle exclut zéro |
| Latence | par phrase, lot de 1 (le cas du direct) |
| RTF | temps de calcul divisé par la durée de l'audio, hors chargement |

Code des mesures : `code/mesures/` et `kaggle/commun/kaggle_commun.py`. Procédure de reproduction : [GUIDE.md, section 7](GUIDE.md#7-reproduire-les-mesures-sur-kaggle).

## 2. Systèmes de la parole au français

Source : `resultats/kaggle/nlp-eval-systemes/resultats_systemes.json` ; COMET : `resultats/kaggle/nlp-comet/resultats_comet.json`. Traductions phrase par phrase : `traductions_par_phrase.csv`.

| Système | WER (%) | CER (%) | BLEU | chrF++ | COMET | RTF | VRAM (Go) |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 : Whisper small + Opus-MT | 46,41 | 24,04 | 11,81 | 35,04 | 0,5477 | 0,088 | 0,73 |
| SeamlessM4T v2 large | 8,58 ¹ | 4,79 ¹ | 17,38 | 46,22 | 0,6901 | 0,146 | 3,48 |
| Whisper-large-v3 (transcription seule) | 9,65 | 3,41 | n.a. | n.a. | n.a. | 0,157 | 3,37 |

¹ Transcription arabe de Seamless, obtenue avec la langue cible arabe (RTF 0,113).

Bootstrap apparié (même fichier) :

| Comparaison | Mesure | Écart | Intervalle à 95 % |
|---|---|---:|---|
| Seamless contre S0 | chrF++ | +11,18 | [+9,19 ; +13,57] |
| Seamless contre S0 | BLEU | +5,57 | [+3,19 ; +7,71] |
| Seamless contre S0 | WER | −37,83 | [−45,67 ; −30,82] |
| Whisper-large-v3 contre S0 | WER | −36,76 | [−44,45 ; −29,55] |
| Seamless contre Whisper-large-v3 | WER | −1,07 | [−2,04 ; −0,08] |

Seamless dépasse S0 sur toutes les mesures de qualité, avec des intervalles qui excluent zéro. L'écart vient surtout de la reconnaissance : Whisper small se trompe sur près d'un mot sur deux, et Opus-MT ne peut pas rattraper ces erreurs. Whisper-large-v3 transcrit presque aussi bien que Seamless (1,07 point de WER d'écart, significatif mais petit) et obtient le meilleur CER.

**S3 sur FLEURS** (`resultats/kaggle/nlp-eval-s3/resultats_s3.json`, kernel lancé depuis l'interface Kaggle, seul moyen de lui transmettre le secret `HF_TOKEN`) : WER 21,12 %, CER 10,91 %, RTF de bout en bout 0,365 (transcription 0,208, traduction par Qwen 0,157), pic de VRAM 7,06 Go. Le texte arabe produit par S3 a ensuite été traduit par quatre traducteurs :

| Traduction du texte arabe de S3 | BLEU | chrF++ | Sorties en arabe (%) |
|---|---:|---:|---:|
| NLLB-200 1,3B | **24,57** | **48,79** | 0 |
| Opus-MT ar-fr | 18,21 | 43,68 | 0 |
| Qwen2.5-1.5B + LoRA | 13,53 | 37,83 | 0 |
| Qwen2.5-1.5B | 12,16 | 37,23 | 5,6 |

La chaîne de Kalam (S3 puis NLLB-1.3B) dépasse donc SeamlessM4T v2 de bout en bout (24,57 BLEU contre 17,38 ; 48,79 chrF++ contre 46,22), alors que S3 transcrit FLEURS moins bien (21,12 % contre 8,58 % de WER) : cette parole lue en arabe égyptien sort de son domaine d'entraînement, MGB-2, où son WER est de 14,26 %. Appliqué au texte de référence, NLLB-1.3B atteint 31,26 BLEU : les erreurs de reconnaissance coûtent environ sept points.

## 3. Traducteurs de texte

Chaque traducteur reçoit la transcription arabe **de référence** de FLEURS : ces scores isolent la traduction, sans erreur de reconnaissance. Source : `resultats/kaggle/nlp-eval-traducteurs/resultats_traducteurs.json` (résumé : `resume_traducteurs.csv`) ; COMET : `resultats/kaggle/nlp-comet/resultats_comet.json`.

| Traducteur | Phrases | BLEU | chrF++ | COMET | Latence médiane (s) | Latence p95 (s) | Sorties en arabe (%) | VRAM (Go) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Opus-MT ar-fr | 267 | 24,31 | 49,58 | 0,7610 | 0,194 | 0,367 | 0 | 0,17 |
| NLLB-200 distillé 600M | 267 | 29,33 | 52,96 | 0,8065 | 0,461 | 0,788 | 0 | 1,26 |
| NLLB-200 distillé 1,3B | 267 | 31,26 | 55,19 | 0,8261 | 0,820 | 1,383 | 0 | 2,78 |
| Qwen2.5-1.5B-Instruct | 267 | 15,23 | 40,24 | 0,7418 | 1,604 | 2,810 | 8,2 | 3,13 |
| Qwen2.5-1.5B-Instruct + LoRA | 267 | 17,33 | 42,05 | 0,7629 | 2,144 | 3,637 | 0 | 3,13 |
| Qwen2.5-7B-Instruct, nf4 | 150 ² | 24,27 | 50,80 | 0,8013 | 2,270 | 4,083 | 0,7 | 6,03 |

² Qwen2.5-7B a été limité aux 150 premières phrases (`limite=150` dans `kaggle/nlp-eval-traducteurs/source.py`) : son score n'est pas directement comparable aux autres.

NLLB-1.3B l'emporte sur les trois métriques, et Opus-MT est le plus rapide. Ainsi, ce tableau fixe les choix du direct : Opus-MT pour le texte gris, NLLB-1.3B pour le texte définitif. Le meilleur système de bout en bout (Seamless, 17,38 BLEU) reste loin du meilleur traducteur de texte (31,26) : la reconnaissance et la segmentation coûtent plus de dix points de BLEU.

## 4. Adaptation LoRA de Qwen2.5-1.5B

Source : `resultats/kaggle/nlp-lora-traduction/resultats_lora.json`, courbe `courbe_perte.png`, journal `metriques.jsonl`, adaptateur `adaptateur_lora/`.

| Réglage | Valeur |
|---|---|
| Modèle | Qwen/Qwen2.5-1.5B-Instruct, fp16 (T4) |
| LoRA | rang 16, α = 32, dropout 0,05, modules `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Optimisation | taux 2 × 10⁻⁴, lot effectif 16, 1 281 pas (un passage), 54,2 min |
| Données | 20 500 paires d'entraînement, 500 de validation, tirées du corpus OPUS filtré (News-Commentary, GlobalVoices, TED2020) ; 29 % avec les phrases précédentes en contexte |
| Perte de validation | 1,7106 au départ, 1,3451 au dernier pas (meilleur pas) |

| Jeu | Mesure | Avant | Après |
|---|---|---:|---:|
| FLEURS (267) | BLEU | 15,18 | 17,20 |
| FLEURS (267) | chrF++ | 40,25 | 41,98 |
| FLEURS (267) | COMET | 0,7417 | 0,7637 |
| FLEURS (267) | sorties en arabe (%) | 8,2 | 0 |
| OPUS validation, sans contexte (200) | BLEU | 10,74 | 13,19 |
| OPUS validation, avec contexte (200) | BLEU | 9,66 | 13,21 |

Les chiffres FLEURS de ce tableau viennent du kernel d'entraînement ; `nlp-eval-traducteurs` les recalcule à 15,23 et 17,33 BLEU (section 3). Le pourcentage de sorties en arabe vient de `resultats_traducteurs.json`.

Mais en cascade, derrière une transcription automatique, le gain du LoRA n'est plus significatif. Source : `resultats/kaggle/nlp-eval-s3-lora/resultats_s3_lora.json` (COMET : `resultats_comet.json`).

| Transcription arabe | chrF++ base | chrF++ LoRA | Intervalle de l'écart | BLEU base | BLEU LoRA | COMET base | COMET LoRA |
|---|---:|---:|---|---:|---:|---:|---:|
| Whisper small (S0) | 30,96 | 30,49 | [−1,83 ; +1,16] | 8,32 | 9,01 | 0,5887 | 0,5964 |
| SeamlessM4T v2 (arabe) | 39,42 | 40,58 | [−0,19 ; +2,98] | 14,74 | 15,68 | 0,7235 | 0,7451 |
| Whisper-large-v3 | 39,02 | 39,95 | [−0,41 ; +2,57] | 14,18 | 15,36 | 0,7171 | 0,7246 |

Les six intervalles bootstrap (chrF++ et BLEU) contiennent zéro. À taille comparable, NLLB-1.3B reste loin devant Qwen affiné (31,26 contre 17,33 BLEU) : c'est pourquoi Kalam traduit avec NLLB.

Sur la grille qualitative de 24 requêtes (16 phrases FLEURS, 8 phrases de journal avec noms propres, sigles, dialecte libanais et expressions idiomatiques), le chrF++ moyen des phrases FLEURS passe de 37,89 à 41,54 avec le LoRA. Source : `resultats/kaggle/nlp-analyse-llm/grille_24_requetes.csv`.

## 5. S3 : entraînement du projecteur

Sources : `modeles/s3_temps1_50k/evaluation_test.json`, `modeles/s3_temps1_50k/metriques.jsonl`, `code/projecteur/journal_s3.md`.

Données : 50 000 segments MGB-2 (144,88 h, 2 086 émissions). Validation : 200 segments tirés du jeu d'entraînement et jamais appris (100 transcrits pour le WER). Test : 1 000 segments de 127 émissions absentes de l'entraînement (2,85 h).

| Pas | 0 | 300 | 600 | 900 | 1 200 | 1 500 |
|---|---:|---:|---:|---:|---:|---:|
| Perte de validation | 4,109 | 0,854 | 0,580 | 0,532 | 0,504 | 0,482 |
| WER de validation (%) | 159,0 | 30,8 | 16,2 | 15,6 | 17,2 | 13,4 |
| CER de validation (%) | 149,3 | 14,2 | 7,4 | 8,0 | 10,5 | 6,8 |

L'entraînement a été arrêté au pas 1 730, soit 0,56 passage sur les deux prévus. Le projecteur retenu est celui du pas 1 500.

| Test MGB-2 (1 000 segments) | WER (%) | CER (%) | Segments au-delà de 100 % de WER | Calcul |
|---|---:|---:|---:|---:|
| Génération brute | 17,28 | 9,66 | 9 | 124,5 s |
| Limite de longueur + suppression des boucles | 14,26 | 7,22 | 0 | 115,6 s (RTF 0,0113) |

Sans les anti-boucles, le WER médian par segment est de 10,5 % et 203 segments sur 1 000 sont transcrits sans erreur. Les deux règles anti-boucles ont été calibrées sur les références de l'entraînement, mais elles ont été conçues après avoir vu les boucles du test : 14,26 % n'est donc pas une mesure aveugle. Pour repère, le modèle Cohere complet est annoncé à 15,5 % sur MGB-2 (classement publié, non vérifié ici, et sur un autre sous-ensemble). Les vecteurs de l'encodeur étaient précalculés : le RTF n'inclut pas l'encodage.

## 6. Mode direct sur Al Jazeera

Essais sur le direct YouTube d'Al Jazeera (<https://www.youtube.com/live/bNyUyrR0PHo>), audio tiré par `yt-dlp` sur le serveur, Colab T4, tunnel Cloudflare. Le retard est mesuré de l'arrivée de la fin de la parole à l'affichage du texte définitif.

| Configuration | Segments | Retard moyen (s) | Retard médian (s) | p95 (s) | Source |
|---|---:|---:|---:|---:|---|
| S3, Opus-MT, Qwen2.5-1.5B avec 2 phrases de contexte | 36 | 3,97 | 3,6 ³ | 6,67 | `resultats/direct/2026-10-23_aljazeera-direct_s3-qwen-contexte2_bilan.json` |
| S3, Opus-MT, NLLB-1.3B (configuration finale) | 23 | 3,8 | 2,9 | 6,7 | `resultats/direct/2026-10-23_aljazeera-direct_s3-opus-nllb_essai-final_bilan.json` |
| Session où le GPU saturait | 60 | 5,0 | 4,0 | 10,6 | `resultats/direct/2026-10-23_aljazeera-direct_s3-opus-qwen_gpu-sature_bilan.json` (médiane sur les 55 phrases reçues) |

³ Médiane calculée sur les 28 segments enregistrés par le spectateur (`…_messages.jsonl`), branché après le chargement ; le bilan du serveur porte sur 36 segments. n.m. : non mesuré.

Observations :

- Premier essai (Qwen avec contexte) : aucune sortie en arabe sur 28 segments, taux de réécriture de 67,8 %, segments de 7,4 s en moyenne. Le petit modèle mêlait les phrases de contexte à la phrase traduite, d'où le passage à NLLB sans contexte.
- Configuration finale : 23 phrases sans arabe, 3 voix repérées.
- L'option `-re` de `ffmpeg` freinait les flux HLS déjà en direct : 18 phrases en 13 minutes avec elle, 56 en 8,5 minutes sans elle.

## 7. Rapport de fin de session

Mesures relevées pendant la mise au point (rapport de projet, section 5), non versionnées dans `resultats/` :

- Rapport complet : 773 s pour 60 segments au premier essai, 61 s pour 10 minutes d'audio après optimisation (T4).
- Résumé par l'API ScaleDown à partir de l'arabe : 2,9 s, contre environ 110 s pour Qwen2.5-7B en local. Les 17 horodatages cités renvoient tous à des segments existants.
- Seuil ECAPA : sur FLEURS, deux lectures d'une même voix ont une similarité cosinus moyenne de 0,557, contre 0,266 entre voix différentes. Seuils retenus : 0,45 en direct, 0,30 dans le rapport.
- Vidéo déposée : une minute de journal demande 26 s de calcul.

## 8. Le fichier `resultats/resultats.csv`

Ce tableau a été écrit le 19 octobre 2026, avant les calculs sur GPU, par `code/mesures/tableau_resultats.py`. Il est conservé tel quel pour l'historique et **ne contient aucun résultat du projet final** :

- les lignes marquées `non mesuré` (`en_attente_gpu_kaggle_et_jeu_R`) sont des emplacements prévus pour S0, les encodeurs candidats et les LLM candidats. Les mesures réelles de S0 sont en section 2 ;
- les six lignes `S2_DIRECT_*` (retard 2,3 s, réécriture 0 %) proviennent d'un flux **simulé** de 600 s (`code/direct/flux_simule.py`), construit à partir de quatre phrases françaises fixes, sans audio ni modèle. S2 (cascade avec LoRA de Qwen3-4B) n'a jamais été entraîné. Ces chiffres ne décrivent aucun modèle.

Les chiffres de référence sont ceux de `resultats/kaggle/`, `resultats/direct/` et `modeles/s3_temps1_50k/`.

## 9. Limites

- FLEURS est de la parole lue, en arabe standard, sur des thèmes encyclopédiques ; les empreintes ECAPA n'y trouvent que des voix féminines. Les scores ne valent qu'en partie pour un journal télévisé.
- Aucune référence française n'existe pour MGB-2 : la qualité de traduction de S3 sur des journaux réels n'est pas mesurée. Sur un extrait de test, une erreur de reconnaissance (« augmentation » au lieu de « décoration ») passe telle quelle dans la traduction.
- La validation de S3 est tirée du jeu d'entraînement (WER optimiste) et le test anti-boucles n'est pas aveugle (section 5).
- Les mesures du direct portent sur quelques essais de quelques minutes, sur une seule chaîne.
- Aucune évaluation humaine n'a été conduite, hors la grille qualitative de 24 requêtes.
- Les métriques automatiques mesurent mal la fidélité sur les noms propres et les chiffres, qui comptent le plus pour un journaliste.
