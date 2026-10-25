# %% [markdown]
# # Cascade « transcription arabe + Qwen2.5-1.5B (base ou LoRA) » sur FLEURS
# On retraduit les transcriptions arabes produites par le kernel `nlp-eval-systemes` (S3 ; Whisper-large-v3 ; Whisper-small ; Seamless-ASR)
# avec Qwen2.5-1.5B-Instruct seul, puis avec l'adaptateur LoRA du kernel `nlp-lora-traduction`.
# Le gabarit est celui de `configs/s3.yaml` (`traduction_texte`), donc l'adaptateur se branche tel quel dans la cascade S3.
# Référence française = FLEURS fr_fr ; source COMET = texte arabe de référence.

# %%
import subprocess, sys, os, json, time, gc
def pip(*a): subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
pip("sacrebleu", "jiwer")
pip("--no-deps", "unbabel-comet")
pip("pytorch-lightning", "torchmetrics", "jsonargparse", "entmax", "sentencepiece")
import torch, transformers, peft
print(torch.__version__, transformers.__version__, peft.__version__)

# %%
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(next(p for p in Path("/kaggle/input").rglob("kaggle_commun.py")).parent))
import kaggle_commun as kc
RACINE = Path(kc.__file__).resolve().parent; kc.preparer_chemins(RACINE)
SORTIE = Path("/kaggle/working")
prompts = kc.charger_prompts(RACINE)
csv_sys = next(Path("/kaggle/input").rglob("traductions_par_phrase.csv"))
adaptateur = next(Path("/kaggle/input").rglob("adapter_config.json")).parent
print(csv_sys, adaptateur)
d = pd.read_csv(csv_sys).fillna("")
colonnes_ar = {c[:-len("_ar_hyp")]: c for c in d.columns if c.endswith("_ar_hyp")}
colonnes_ar = {s: c for s, c in colonnes_ar.items() if d[c].str.strip().str.len().gt(0).any()}
print("transcriptions arabes disponibles :", list(colonnes_ar), "| S3 présent :", "S3" in colonnes_ar)
if "S3" not in colonnes_ar:
    print("ATTENTION : pas de transcription S3 (kernel nlp-eval-systemes lancé sans le secret HF_TOKEN) ; "
          "la cascade n'est mesurée que sur les autres transcriptions. Relancer nlp-eval-systemes puis ce kernel.")
refs = d["fr_ref"].tolist(); srcs = d["ar_ref"].tolist()

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
MODELE = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODELE)
base = AutoModelForCausalLM.from_pretrained(MODELE, dtype=torch.float16).cuda()
modele = PeftModel.from_pretrained(base, adaptateur).eval()
SORT = {}   # nom -> liste de traductions
for s, col in colonnes_ar.items():
    textes = d[col].tolist(); vides = [not t.strip() for t in textes]
    with modele.disable_adapter():
        b = kc.traduire(modele, tok, prompts, [t if t.strip() else "." for t in textes])
    l = kc.traduire(modele, tok, prompts, [t if t.strip() else "." for t in textes])
    SORT[f"{s}+Qwen"] = ["" if v else x for v, x in zip(vides, b)]
    SORT[f"{s}+Qwen+LoRA"] = ["" if v else x for v, x in zip(vides, l)]
    print(s, "traduit")
del modele, base; torch.cuda.empty_cache(); gc.collect()

# %%
res = {"n": len(d), "systemes": {}}
for nom, hyp in SORT.items():
    m = kc.metriques_traduction(refs, hyp); res["systemes"][nom] = m
    c = kc.comet(srcs, hyp, refs)
    m["comet"] = c["comet"]
    if c["comet"] is None: m["comet_motif"] = c.get("motif")
    else: d[f"{nom}_comet"] = c["scores_segments"]
    d[f"{nom}_fr"] = hyp; d[f"{nom}_chrf"] = [kc.chrf_phrase(r, h) for r, h in zip(refs, hyp)]
    print(nom, {k: m.get(k) for k in ("bleu", "chrf_pp", "comet")})
    gc.collect(); torch.cuda.empty_cache()
# WER arabe des transcriptions (pour mémoire)
from scores import calculer_wer_cer
res["wer_asr"] = {s: calculer_wer_cer(srcs, d[c].tolist()) for s, c in colonnes_ar.items()}
res["wer_asr"] = {s: {"wer_pct": v["wer_pct"], "cer_pct": v["cer_pct"]} for s, v in res["wer_asr"].items()}

# %%
from reechantillonnage import comparer_systemes_bootstrap
res["bootstrap"] = []
for s in colonnes_ar:
    for m in ("chrf++", "bleu"):
        r = comparer_systemes_bootstrap(refs, SORT[f"{s}+Qwen"], SORT[f"{s}+Qwen+LoRA"], m, n_tirages=300)
        res["bootstrap"].append({"a": f"{s}+Qwen", "b": f"{s}+Qwen+LoRA", "metrique": m, "n_tirages": 300, **{k: r[k] for k in r if k != "ecarts"}})
kc.ecrire_json(SORTIE / "resultats_s3_lora.json", res)
d.to_csv(SORTIE / "traductions_cascade_lora.csv", index=False, encoding="utf-8")
print(pd.DataFrame(res["systemes"]).T[["bleu", "chrf_pp", "comet"]].to_string())
print(json.dumps(res["wer_asr"], indent=1))
