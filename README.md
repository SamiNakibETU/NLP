# Sous-titrage français de journaux télévisés en arabe

Système de traduction de la parole arabe vers des sous-titres français en mode différé (`.srt`) et en mode direct (flux temps réel avec 2 à 3 secondes de latence).

Le projet compare deux architectures partageant le même encodeur audio (`openai/whisper-large-v3`), le même modèle de langue (`Qwen/Qwen3-4B`) et les mêmes données d'entraînement :
- **Cascade (`S1` / `S2`)** : transcription automatique en arabe par Whisper, suivie d'une traduction contextuelle en français par Qwen3 entraîné par adaptation LoRA.
- **Pont audio-LLM direct (`S3`)** : extraction des représentations de l'encodeur audio, regroupement temporel ($k = 4$), projection MLP (`Linear -> GELU -> Linear`) et décodage direct en français par le LLM sans étape intermédiaire de texte arabe.

## Structure du dépôt

```text
code/
  donnees/      Normalisation arabe, validation des schémas, collecte OPUS/Wikidata et pipeline MGB-2
  mesures/      Calcul des scores (WER, CER, chrF++, BLEU, COMET, RTF), bootstrap apparié et tableau CSV
  ecoute/       Extraction des vecteurs de l'encodeur audio et comparaison des modèles candidats
  cascade/      Préparation des exemples (40/30/20/10), entraînement LoRA avec reprise et inférence S1/S2
  projecteur/   Regroupement temporel, projecteur MLP, assemblage inputs_embeds et entraînement S3 (temps 1 et 2)
  direct/       Streaming temps réel (règle des deux passes, seuil de probabilité, serveur FastAPI/WebSocket)
  demo/         Baseline S0 (Whisper small + Opus-MT), découpage VAD et interface web d'export SRT
configs/        Fichiers de configuration YAML par système (s0, choix_briques, s2_cascade, s3, direct)
notebooks/      Notebooks d'entraînement et de distillation pour exécution GPU (Kaggle T4)
resultats/      Tableau récapitulatif des mesures (resultats.csv)
```

## Installation

```bash
pip install -r requirements.txt
cp .env.exemple .env
```

## Utilisation rapide

### 1. Sous-titrage d'un fichier audio (mode différé `.srt`)

```bash
# Baseline S0 (Whisper small + Opus-MT)
python code/demo/systeme_s0.py --audio chemin/vers/extrait.wav --sortie-srt sous_titres.srt

# Cascade S2 avec contexte des 2 phrases précédentes et lexique de noms propres
python code/cascade/inference_s1_s2.py --systeme S2 --audio chemin/vers/extrait.wav --sortie-srt sous_titres_s2.srt

# Interface web locale de dépôt de fichier (port 7860)
python code/demo/app.py --port 7860
```

### 2. Sous-titrage en direct (streaming WebSocket)

```bash
# Lancer le serveur temps réel FastAPI / WebSocket (page mobile sur http://127.0.0.1:8000)
python code/direct/serveur_direct.py --hote 127.0.0.1 --port 8000

# Simuler 10 minutes de flux continu et mesurer le retard et le taux de réécriture
python code/direct/flux_simule.py --essai --duree-s 600
```

### 3. Vérification et tests unitaires

Chaque module dispose d'un mode `--essai` et d'une suite de tests unitaires exécutable hors ligne :

```bash
python -m unittest discover -s code/donnees/tests
python -m unittest discover -s code/mesures/tests
python -m unittest discover -s code/demo/tests
python -m unittest discover -s code/ecoute/tests
python -m unittest discover -s code/cascade/tests
python -m unittest discover -s code/projecteur/tests
python -m unittest discover -s code/direct/tests
```
