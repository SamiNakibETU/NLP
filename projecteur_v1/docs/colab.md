# Lancer sur Colab (GPU T4)

Menu : Exécution -> Modifier le type d'exécution -> GPU T4. Glisse `speech_proj.zip` dans l'espace de fichiers de Colab (/content).
Ajoute ton jeton Hugging Face dans les Secrets Colab (clé : HF_TOKEN), après avoir accepté la licence de QCRI/mgb2.

## Cellule 1 : GPU, code, dépendances, tests
```python
!nvidia-smi -L
!cd /content && unzip -q -o speech_proj.zip
!pip install -q peft librosa soundfile sacrebleu sentencepiece accelerate pytest
%cd /content/speech_proj
!python -m pytest -q
```

## Cellule 2 : données dev (1 Go, téléchargées depuis Hugging Face)
```python
from google.colab import userdata
from huggingface_hub import hf_hub_download
import tarfile, os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
p = hf_hub_download("QCRI/mgb2", "dev/dev.tar.gz", repo_type="dataset")
os.makedirs("/content/dev", exist_ok=True)
with tarfile.open(p) as t:
    t.extractall("/content/dev_raw")
!mkdir -p /content/dev && cp -r /content/dev_raw/dev/wav /content/dev/wav && ls /content/dev/wav | head -3
```
(fichier confirmé : dev/dev.tar.gz dans QCRI/mgb2)

## Les manifestes
`manifest_dev_train.jsonl` (5069 segments, 15 émissions) et `manifest_dev_val.jsonl` (773 segments, 2 émissions) : champs audio, start, end, text_ar.
Les chemins audio pointent vers /content/dev/wav/<programme>.wav.
