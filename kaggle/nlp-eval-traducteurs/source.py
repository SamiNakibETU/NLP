# %% [markdown]
# # Traducteurs TEXTE arabe -> français pour le « texte blanc » du direct
# Candidats : Opus-MT ar-fr, NLLB-200 600M et 1,3B, Qwen2.5-1.5B-Instruct (prompt de `configs/s3.yaml`), le même avec l'adaptateur LoRA,
# Qwen2.5-7B-Instruct en 4 bits nf4. Jeu : FLEURS test (arabe de référence -> français) et, si disponible, les transcriptions arabes de S3.
# Latence par phrase en lot de 1 (cas du direct), VRAM, part de sorties contenant de l'écriture arabe, BLEU, chrF++, COMET.

# %%
import subprocess, sys, os, json, time, gc, re
def pip(*a): subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
pip("sacrebleu", "bitsandbytes>=0.46", "sentencepiece")
pip("--no-deps", "unbabel-comet")
pip("pytorch-lightning", "torchmetrics", "jsonargparse", "entmax")
import torch, transformers, peft
import numpy as np, pandas as pd
from pathlib import Path
print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.get_device_name(0))
sys.path.insert(0, str(next(p for p in Path("/kaggle/input").rglob("kaggle_commun.py")).parent))
import kaggle_commun as kc
RACINE = Path(kc.__file__).resolve().parent; kc.preparer_chemins(RACINE)
SORTIE = Path("/kaggle/working")
prompts = kc.charger_prompts(RACINE)
fleurs = kc.charger_fleurs(avec_audio=False)
JEUX = {"fleurs": {"src": [p["ar"] for p in fleurs], "ref": [p["fr"] for p in fleurs]}}
csv_sys = list(Path("/kaggle/input").rglob("traductions_par_phrase.csv"))
if csv_sys:
    d = pd.read_csv(csv_sys[0]).fillna("")
    if "S3_ar_hyp" in d.columns and d["S3_ar_hyp"].str.strip().str.len().gt(0).any():
        JEUX["s3"] = {"src": d["S3_ar_hyp"].tolist(), "ref": d["fr_ref"].tolist(), "src_ar_ref": d["ar_ref"].tolist()}
print({k: len(v["src"]) for k, v in JEUX.items()})
adaptateur = next(Path("/kaggle/input").rglob("adapter_config.json")).parent
MOTIF_AR = re.compile(r"[؀-ۿ]")
RES = {"jeux": {k: len(v["src"]) for k, v in JEUX.items()}, "traducteurs": {}}
SORT = {}   # (traducteur, jeu) -> traductions

def sync(): torch.cuda.synchronize()

def evaluer(nom, fn, jeux=("fleurs", "s3"), limite=None):
    """fn(texte) -> traduction ; lot de 1, latence mesurée par phrase."""
    r = {}
    for jeu in jeux:
        if jeu not in JEUX: continue
        src = JEUX[jeu]["src"][:limite]; lat, hyp = [], []
        fn(src[0]); sync()   # échauffement
        torch.cuda.reset_peak_memory_stats()
        for t in src:
            sync(); t0 = time.perf_counter(); hyp.append(fn(t) if t.strip() else ""); sync(); lat.append(time.perf_counter() - t0)
        refs = JEUX[jeu]["ref"][:len(src)]
        m = kc.metriques_traduction(refs, hyp)
        m.update({"n": len(src), "latence_mediane_s": round(float(np.median(lat)), 3), "latence_p95_s": round(float(np.percentile(lat, 95)), 3),
                  "part_sorties_arabes": round(sum(bool(MOTIF_AR.search(h)) for h in hyp) / len(hyp), 4),
                  "pic_vram_go": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
        SORT[(nom, jeu)] = hyp; r[jeu] = m
        print(nom, jeu, {k: m[k] for k in ("bleu", "chrf_pp", "latence_mediane_s", "latence_p95_s", "part_sorties_arabes", "pic_vram_go")})
    RES["traducteurs"][nom] = r; sauver()

def sauver():
    kc.ecrire_json(SORTIE / "resultats_traducteurs.json", RES)
    for jeu in JEUX:
        cols = {"ar": JEUX[jeu]["src"], "fr_ref": JEUX[jeu]["ref"]}
        for (nom, j), h in SORT.items():
            if j == jeu: cols[nom] = h + [""] * (len(cols["ar"]) - len(h))
        for k in list(cols): pass
        pd.DataFrame(cols).to_csv(SORTIE / f"traductions_{jeu}.csv", index=False, encoding="utf-8")

def liberer(*objs):
    for o in objs: del o
    gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ## Opus-MT et NLLB (traducteurs dédiés, fp16, glouton)

# %%
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
def seq2seq(nom, mid, **tokargs):
    try:
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        tk = AutoTokenizer.from_pretrained(mid, **tokargs)
        m = AutoModelForSeq2SeqLM.from_pretrained(mid, dtype=torch.float16).cuda().eval(); sync()
        RES.setdefault("chargement_s", {})[nom] = round(time.time() - t0, 1)
        extra = {"forced_bos_token_id": tk.convert_tokens_to_ids("fra_Latn")} if "nllb" in mid else {}
        def fn(t):
            b = tk([t], return_tensors="pt", truncation=True, max_length=256).to("cuda")
            with torch.no_grad(): g = m.generate(**b, num_beams=1, max_new_tokens=256, **extra)
            return tk.batch_decode(g, skip_special_tokens=True)[0].strip()
        evaluer(nom, fn)
        del m
    except Exception as e:
        import traceback; traceback.print_exc(); RES["traducteurs"][nom] = {"erreur": f"{type(e).__name__}: {e}"}
    gc.collect(); torch.cuda.empty_cache()
seq2seq("opus-mt-ar-fr", "Helsinki-NLP/opus-mt-ar-fr")
seq2seq("nllb-200-distilled-600M", "facebook/nllb-200-distilled-600M", src_lang="arb_Arab")
seq2seq("nllb-200-distilled-1.3B", "facebook/nllb-200-distilled-1.3B", src_lang="arb_Arab")

# %% [markdown]
# ## Qwen2.5-1.5B-Instruct (base, puis avec l'adaptateur LoRA)

# %%
from transformers import AutoModelForCausalLM
from peft import PeftModel
try:
    t0 = time.time(); tq = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    bq = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float16).cuda().eval()
    pm = PeftModel.from_pretrained(bq, adaptateur).eval(); sync(); RES.setdefault("chargement_s", {})["qwen-1.5b+lora"] = round(time.time() - t0, 1)
    def un(t, adapte):
        if adapte: return kc.traduire(pm, tq, prompts, [t], taille_lot=1)[0]
        with pm.disable_adapter(): return kc.traduire(pm, tq, prompts, [t], taille_lot=1)[0]
    evaluer("qwen2.5-1.5b-base", lambda t: un(t, False))
    evaluer("qwen2.5-1.5b-lora", lambda t: un(t, True))
    del pm, bq
except Exception as e:
    import traceback; traceback.print_exc(); RES["traducteurs"]["qwen2.5-1.5b"] = {"erreur": f"{type(e).__name__}: {e}"}
gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ## Qwen2.5-7B-Instruct en 4 bits nf4 (calcul fp16), mêmes gabarits

# %%
try:
    from transformers import BitsAndBytesConfig
    t0 = time.time(); t7 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    m7 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map={"": 0}, dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)).eval()
    sync(); RES.setdefault("chargement_s", {})["qwen2.5-7b-nf4"] = round(time.time() - t0, 1)
    evaluer("qwen2.5-7b-nf4", lambda t: kc.traduire(m7, t7, prompts, [t], taille_lot=1)[0], jeux=("fleurs", "s3"), limite=150)
    del m7
except Exception as e:
    import traceback; traceback.print_exc(); RES["traducteurs"]["qwen2.5-7b-nf4"] = {"erreur": f"{type(e).__name__}: {e}"}
gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ## COMET (source = texte arabe lu par le traducteur ; pour le jeu s3, l'arabe de référence)

# %%
for (nom, jeu), hyp in SORT.items():
    n = len(hyp); src = JEUX[jeu].get("src_ar_ref", JEUX[jeu]["src"])[:n]
    c = kc.comet(src, hyp, JEUX[jeu]["ref"][:n])
    RES["traducteurs"][nom][jeu]["comet"] = c["comet"]
    if c["comet"] is None: RES["traducteurs"][nom][jeu]["comet_motif"] = c.get("motif")
    print(nom, jeu, "COMET", c["comet"]); sauver(); gc.collect(); torch.cuda.empty_cache()
rows = [{"traducteur": n, "jeu": j, **{k: v.get(k) for k in ("n", "bleu", "chrf_pp", "comet", "latence_mediane_s", "latence_p95_s", "part_sorties_arabes", "pic_vram_go")}}
        for n, r in RES["traducteurs"].items() for j, v in r.items() if isinstance(v, dict) and "bleu" in v]
pd.DataFrame(rows).to_csv(SORTIE / "resume_traducteurs.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
