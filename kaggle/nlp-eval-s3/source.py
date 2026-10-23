# %% [markdown]
# # S3 seul sur FLEURS ar_eg test (267 phrases) : transcription arabe, puis traduction par 4 traducteurs texte
# Exige le secret Kaggle `HF_TOKEN` attaché à ce kernel (accès restreint à CohereLabs/cohere-transcribe-arabic-07-2026).
# Mesures : WER/CER arabe de S3 ; BLEU et chrF++ du français obtenu avec Opus-MT, NLLB-1.3B, Qwen2.5-1.5B (base) et Qwen2.5-1.5B + LoRA ;
# RTF de bout en bout (transcription S3 + traduction par le Qwen de S3), pic de VRAM. COMET : kernel `nlp-comet`.

# %%
import subprocess, sys, os, json, time, gc, re
def pip(*a): subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
pip("-U", "transformers>=5.4", "peft", "accelerate", "sacrebleu", "jiwer", "librosa", "sentencepiece", "python-dotenv")
import torch, transformers
print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, str(next(p for p in Path("/kaggle/input").rglob("kaggle_commun.py")).parent))
import kaggle_commun as kc
RACINE = Path(kc.__file__).resolve().parent; kc.preparer_chemins(RACINE)
SORTIE = Path("/kaggle/working")
jeton = kc.jeton_hf()
if not jeton:
    raise PermissionError("secret Kaggle HF_TOKEN absent : l'attacher à ce kernel (Add-ons > Secrets) puis faire Save & Run All")
os.environ["HF_TOKEN"] = jeton
from huggingface_hub import hf_hub_download
hf_hub_download("CohereLabs/cohere-transcribe-arabic-07-2026", "config.json", token=jeton)   # lève une erreur si 403
print("accès Cohere : ok")

# %%
fleurs = kc.charger_fleurs()
duree = sum(p["duree_s"] for p in fleurs)
refs_fr = [p["fr"] for p in fleurs]; refs_ar = [p["ar"] for p in fleurs]
print(len(fleurs), "phrases,", round(duree / 60, 1), "min")
def sync(): torch.cuda.synchronize()
RES = {"n_phrases": len(fleurs), "duree_audio_s": round(duree, 1), "versions": {"torch": torch.__version__, "transformers": transformers.__version__}}

cfg = yaml.safe_load((RACINE / "configs" / "s3.yaml").read_text(encoding="utf-8"))
for sec in ("temps_1_pilote", "temps_1_50k"): cfg[sec]["precision_llm"] = "float16"     # T4 : fp16 au lieu de bf16
chemin_cfg = SORTIE / "s3_kaggle_fp16.yaml"; chemin_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
import transcrire_s3
transcrire_s3.CHEMIN_CONFIG_S3 = chemin_cfg
torch.cuda.reset_peak_memory_stats(); t0 = time.time()
s3 = transcrire_s3.TranscripteurS3.charger("temps_1_50k", RACINE / "modeles" / "s3_temps1_50k" / "projecteur_meilleur.pt")
sync(); RES["temps_chargement_s"] = round(time.time() - t0, 1)
s3.transcrire_signal(fleurs[0]["audio"]); sync()

# %% [markdown]
# ## Transcription arabe S3, puis traduction par le Qwen de S3 (cascade d'origine)

# %%
ar_h, fr_s3, t_asr, t_mt = [], [], 0.0, 0.0
for p in fleurs:
    sync(); a = time.time(); segs = s3.transcrire_signal(p["audio"]); sync(); b = time.time()
    textes = [s["texte"] for s in segs]; fr = s3.traduire(textes); sync(); c = time.time()
    t_asr += b - a; t_mt += c - b
    ar_h.append(" ".join(textes).strip()); fr_s3.append(" ".join(f for f in fr if f).strip())
from scores import calculer_wer_cer
w = calculer_wer_cer(refs_ar, ar_h)
m = kc.metriques_traduction(refs_fr, fr_s3)
RES["s3"] = {"wer_pct": w["wer_pct"], "cer_pct": w["cer_pct"], "rtf_bout_en_bout": round((t_asr + t_mt) / duree, 4),
             "rtf_transcription": round(t_asr / duree, 4), "rtf_traduction": round(t_mt / duree, 4),
             "pic_vram_go": round(torch.cuda.max_memory_allocated() / 1e9, 2), **m}
print(RES["s3"])
pd.DataFrame({"id": [p["id"] for p in fleurs], "ar_ref": refs_ar, "fr_ref": refs_fr, "S3_ar_hyp": ar_h, "S3_fr_hyp": fr_s3}
             ).to_csv(SORTIE / "traductions_par_phrase.csv", index=False, encoding="utf-8")
kc.ecrire_json(SORTIE / "resultats_s3.json", RES)
del s3; gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ## Traduction du texte arabe de S3 par Opus-MT, NLLB-1.3B, Qwen2.5-1.5B base et + LoRA

# %%
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
MOTIF_AR = re.compile(r"[؀-ۿ]")
d = pd.read_csv(SORTIE / "traductions_par_phrase.csv").fillna("")
RES["traducteurs"] = {}
def noter(nom, hyp):
    m = kc.metriques_traduction(refs_fr, hyp)
    m["part_sorties_arabes"] = round(sum(bool(MOTIF_AR.search(h)) for h in hyp) / len(hyp), 4)
    RES["traducteurs"][nom] = m; d[f"S3+{nom}_fr"] = hyp
    print(nom, m["bleu"], m["chrf_pp"]); kc.ecrire_json(SORTIE / "resultats_s3.json", RES)
    d.to_csv(SORTIE / "traductions_par_phrase.csv", index=False, encoding="utf-8")
def seq2seq(nom, mid, **tokargs):
    tk = AutoTokenizer.from_pretrained(mid, **tokargs)
    mm = AutoModelForSeq2SeqLM.from_pretrained(mid, dtype=torch.float16).cuda().eval()
    extra = {"forced_bos_token_id": tk.convert_tokens_to_ids("fra_Latn")} if "nllb" in mid else {}
    out = []
    for t in ar_h:
        if not t.strip(): out.append(""); continue
        b = tk([t], return_tensors="pt", truncation=True, max_length=256).to("cuda")
        with torch.no_grad(): g = mm.generate(**b, num_beams=1, max_new_tokens=256, **extra)
        out.append(tk.batch_decode(g, skip_special_tokens=True)[0].strip())
    noter(nom, out); del mm; gc.collect(); torch.cuda.empty_cache()
seq2seq("opus-mt-ar-fr", "Helsinki-NLP/opus-mt-ar-fr")
seq2seq("nllb-1.3B", "facebook/nllb-200-distilled-1.3B", src_lang="arb_Arab")
prompts = kc.charger_prompts(RACINE)
tq = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
bq = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float16).cuda()
adaptateur = next(Path("/kaggle/input").rglob("adapter_config.json")).parent
pm = PeftModel.from_pretrained(bq, adaptateur).eval()
with pm.disable_adapter():
    noter("qwen1.5b-base", kc.traduire(pm, tq, prompts, [t if t.strip() else "." for t in ar_h]))
noter("qwen1.5b-lora", kc.traduire(pm, tq, prompts, [t if t.strip() else "." for t in ar_h]))
print(json.dumps(RES, ensure_ascii=False, indent=1)[:3000])
