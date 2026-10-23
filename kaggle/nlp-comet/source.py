# %% [markdown]
# # COMET (Unbabel/wmt22-comet-da) de toutes les traductions produites par les autres kernels
# COMET plante avec transformers 5 (installé par défaut sur Kaggle) : ce kernel fixe transformers 4.46 avant tout import.
# Source COMET = arabe de référence FLEURS, référence = français FLEURS.

# %%
import subprocess, sys, os, json, glob
def pip(*a): subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a], check=True)
pip("transformers==4.46.3", "tokenizers<0.21", "huggingface_hub<1.0", "sentencepiece", "numpy<2.1")
pip("--no-deps", "unbabel-comet")
pip("pytorch-lightning", "torchmetrics", "jsonargparse", "entmax", "pandas")
import torch, transformers, pandas as pd
print(torch.__version__, transformers.__version__)
from pathlib import Path
from comet import download_model, load_from_checkpoint
modele = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

# %%
ENTREE = Path("/kaggle/input")
travaux = []   # (fichier, nom colonne, src, ref, hyp)
def ajouter(fichier, colonnes_exclues, src="ar", ref="fr_ref", prefixe=""):
    d = pd.read_csv(fichier).fillna("")
    for c in d.columns:
        if c in colonnes_exclues or c in (src, ref): continue
        travaux.append((fichier, c, prefixe, d[src].tolist(), d[ref].tolist(), d[c].tolist(), d))
for f in ENTREE.rglob("traductions_fleurs.csv"):
    d = pd.read_csv(f)
    if "fr_base" in d.columns:   # kernel LoRA
        ajouter(f, {"id", "chrf_base", "chrf_lora", "comet_base", "comet_lora"}, prefixe="lora/")
    else:                         # kernel traducteurs
        ajouter(f, {"id"}, prefixe="traducteurs/fleurs/")
for f in ENTREE.rglob("traductions_s3.csv"):
    ajouter(f, {"id"}, prefixe="traducteurs/s3/")   # source = transcription S3 (jeu s3 du kernel traducteurs)
for f in ENTREE.rglob("traductions_par_phrase.csv"):
    d = pd.read_csv(f).fillna("")
    for c in d.columns:
        if c.endswith("_fr_hyp"):
            travaux.append((f, c, "systemes/", d["ar_ref"].tolist(), d["fr_ref"].tolist(), d[c].tolist(), d))
for f in ENTREE.rglob("traductions_cascade_lora.csv"):
    d = pd.read_csv(f).fillna("")
    for c in d.columns:
        if c.endswith("_fr") and "+Qwen" in c:
            travaux.append((f, c, "cascade/", d["ar_ref"].tolist(), d["fr_ref"].tolist(), d[c].tolist(), d))
print(len(travaux), "colonnes à scorer")

# %%
res, par_phrase = {}, {}
for f, col, pre, src, ref, hyp, d in travaux:
    nom = pre + col
    idx = [i for i, h in enumerate(hyp) if str(h).strip()]
    donnees = [{"src": src[i], "mt": str(hyp[i]), "ref": ref[i]} for i in idx]
    sortie = modele.predict(donnees, batch_size=32, gpus=1, progress_bar=False)
    complet = [None] * len(hyp)
    for i, s in zip(idx, sortie.scores): complet[i] = float(s)
    res[nom] = {"comet": round(float(sortie.system_score), 4), "n": len(hyp), "n_non_vides": len(idx)}
    par_phrase[nom] = complet
    print(nom, res[nom])
Path("/kaggle/working").mkdir(exist_ok=True)
json.dump(res, open("/kaggle/working/resultats_comet.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
pd.DataFrame(par_phrase).to_csv("/kaggle/working/comet_par_phrase.csv", index=False)
