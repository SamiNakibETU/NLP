# %% [markdown]
# # Évaluation des systèmes audio arabe -> français sur FLEURS (ar_eg test)
# Systèmes : S0 (Whisper-small + Opus-MT), SeamlessM4T v2 large (parole -> français), S3 (encodeur Cohere gelé + projecteur
# + Qwen2.5-1.5B, cascade : transcription arabe puis traduction texte), et Whisper-large-v3 en ASR seul (WER).
# Mesures : WER/CER (arabe), BLEU, chrF++, COMET (source = texte arabe de référence), RTF, temps de chargement, pic de VRAM.
# S3 exige l'accès au modèle Cohere (restreint) : le secret Kaggle HF_TOKEN doit être attaché au kernel, sinon S3 est sauté.

# %%
import subprocess, sys, os, json, time, gc
def pip(*a): subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
pip("-U", "transformers>=5.4", "peft", "accelerate", "sacrebleu", "jiwer", "librosa", "sentencepiece", "python-dotenv")
pip("--no-deps", "unbabel-comet")
pip("pytorch-lightning", "torchmetrics", "jsonargparse", "entmax")
import torch, transformers
print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))

# %%
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(next(p for p in Path("/kaggle/input").rglob("kaggle_commun.py")).parent))
import kaggle_commun as kc
RACINE = Path(kc.__file__).resolve().parent; kc.preparer_chemins(RACINE)
SORTIE = Path("/kaggle/working"); SORTIE.mkdir(exist_ok=True)
NB_MAX = None       # essai rapide : 20
fleurs = kc.charger_fleurs(limite=NB_MAX)
duree_totale = sum(p["duree_s"] for p in fleurs)
print(len(fleurs), "phrases FLEURS ar_eg test appariées au français ;", round(duree_totale / 60, 1), "min d'audio")
RES = {"jeu": "FLEURS ar_eg test (1 phrase par id, 1re lecture), référence FR = fr_fr test", "n_phrases": len(fleurs),
       "duree_audio_s": round(duree_totale, 1), "systemes": {}, "versions": {"torch": torch.__version__, "transformers": transformers.__version__}}
HYP = {}   # système -> {"ar": [...], "fr": [...]}
def sync(): torch.cuda.synchronize()
def liberer():
    gc.collect(); torch.cuda.empty_cache()
def sauver():
    kc.ecrire_json(SORTIE / "resultats_systemes.json", RES)
    cols = {"id": [p["id"] for p in fleurs], "ar_ref": [p["ar"] for p in fleurs], "fr_ref": [p["fr"] for p in fleurs]}
    for s, h in HYP.items():
        if h.get("ar") is not None: cols[f"{s}_ar_hyp"] = h["ar"]
        if h.get("fr") is not None: cols[f"{s}_fr_hyp"] = h["fr"]
    pd.DataFrame(cols).to_csv(SORTIE / "traductions_par_phrase.csv", index=False, encoding="utf-8")

def mesurer_ar(nom, hyps_ar):
    from scores import calculer_wer_cer
    r = calculer_wer_cer([p["ar"] for p in fleurs], hyps_ar)
    RES["systemes"][nom].update({"wer_pct": r["wer_pct"], "cer_pct": r["cer_pct"]})

def asr_whisper(modele, proc, audio, langue="ar", max_new=225):
    """Whisper sur un signal 16 kHz ; au-delà de 30 s, tranches de 30 s mises bout à bout."""
    morceaux = [audio[i:i + 480000] for i in range(0, len(audio), 480000)]
    textes = []
    for m in morceaux:
        f = proc(m, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
        with torch.no_grad():
            ids = modele.generate(f, language=langue, task="transcribe", max_new_tokens=max_new)
        textes.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return " ".join(textes).strip()

# %% [markdown]
# ## S0 : Whisper-small + Opus-MT (découpage VAD d'énergie de code/demo/systeme_s0.py)

# %%
try:
    from systeme_s0 import charger_config_s0, decouper_audio_vad
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoModelForSeq2SeqLM, AutoTokenizer
    cfg = charger_config_s0(RACINE / "configs" / "s0.yaml")
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    wp = WhisperProcessor.from_pretrained(cfg["ecoute"]["modele_id"])
    wm = WhisperForConditionalGeneration.from_pretrained(cfg["ecoute"]["modele_id"], dtype=torch.float16).cuda().eval()
    mt_tok = AutoTokenizer.from_pretrained(cfg["traduction"]["modele_id"])
    mt = AutoModelForSeq2SeqLM.from_pretrained(cfg["traduction"]["modele_id"], dtype=torch.float16).cuda().eval()
    sync(); charge = time.time() - t0
    vad = cfg["vad"]; ar_h, fr_h = [], []
    asr_whisper(wm, wp, fleurs[0]["audio"], max_new=cfg["traduction"]["max_new_tokens"])   # échauffement
    sync(); t0 = time.time()
    for p in fleurs:
        tr = decouper_audio_vad(p["audio"], 16000, vad["duree_max_segment_s"], vad["silence_min_coupure_s"], vad["seuil_energie_db"])
        a, f = [], []
        for _, _, seg in tr:
            txt = asr_whisper(wm, wp, seg, max_new=cfg["traduction"]["max_new_tokens"]); a.append(txt)
            if txt.strip():
                b = mt_tok([txt], return_tensors="pt", truncation=True).to("cuda")
                with torch.no_grad():
                    g = mt.generate(**b, num_beams=int(cfg["traduction"]["num_beams"]), max_new_tokens=int(cfg["traduction"]["max_new_tokens"]))
                f.append(mt_tok.batch_decode(g, skip_special_tokens=True)[0].strip())
        ar_h.append(" ".join(a).strip()); fr_h.append(" ".join(f).strip())
    sync(); calc = time.time() - t0
    HYP["S0"] = {"ar": ar_h, "fr": fr_h}
    RES["systemes"]["S0"] = {"description": "openai/whisper-small + Helsinki-NLP/opus-mt-ar-fr (fp16, T4)", "temps_chargement_s": round(charge, 1),
                             "temps_calcul_s": round(calc, 1), "rtf": round(calc / duree_totale, 4), "pic_vram_go": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    mesurer_ar("S0", ar_h); print(RES["systemes"]["S0"])
    del wm, mt; liberer(); sauver()
except Exception as e:
    import traceback; traceback.print_exc(); RES["systemes"]["S0"] = {"erreur": f"{type(e).__name__}: {e}"}

# %% [markdown]
# ## SeamlessM4T v2 large (parole arabe -> texte français), puis son ASR arabe pour le WER

# %%
try:
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    sp = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    sm = SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large", dtype=torch.float16).cuda().eval()
    sync(); charge = time.time() - t0
    def seamless(audio, langue):
        try: ent = sp(audio=audio, sampling_rate=16000, return_tensors="pt")
        except TypeError: ent = sp(audios=audio, sampling_rate=16000, return_tensors="pt")
        ent = {k: (v.to("cuda", torch.float16) if v.dtype.is_floating_point else v.to("cuda")) for k, v in ent.items()}
        with torch.no_grad():
            ids = sm.generate(**ent, tgt_lang=langue)
        ids = ids[0] if isinstance(ids, (tuple, list)) else ids
        return sp.decode(ids[0].tolist() if ids.dim() > 1 else ids.tolist(), skip_special_tokens=True).strip()
    seamless(fleurs[0]["audio"], "fra")
    sync(); t0 = time.time(); fr_h = [seamless(p["audio"], "fra") for p in fleurs]; sync(); calc = time.time() - t0
    vram = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    t0 = time.time(); ar_h = [seamless(p["audio"], "arb") for p in fleurs]; sync(); calc_ar = time.time() - t0
    HYP["SEAMLESS"] = {"ar": ar_h, "fr": fr_h}
    RES["systemes"]["SEAMLESS"] = {"description": "facebook/seamless-m4t-v2-large, SpeechToText, tgt_lang=fra (fp16, T4) ; WER via tgt_lang=arb",
                                   "temps_chargement_s": round(charge, 1), "temps_calcul_s": round(calc, 1), "rtf": round(calc / duree_totale, 4),
                                   "rtf_asr_arabe": round(calc_ar / duree_totale, 4), "pic_vram_go": vram}
    mesurer_ar("SEAMLESS", ar_h); print(RES["systemes"]["SEAMLESS"])
    del sm; liberer(); sauver()
except Exception as e:
    import traceback; traceback.print_exc(); RES["systemes"]["SEAMLESS"] = {"erreur": f"{type(e).__name__}: {e}"}

# %% [markdown]
# ## Whisper-large-v3 (ASR arabe seul, pour comparer à l'encodeur retenu)

# %%
try:
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    lp = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    lm = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", dtype=torch.float16).cuda().eval()
    sync(); charge = time.time() - t0
    asr_whisper(lm, lp, fleurs[0]["audio"]); sync(); t0 = time.time()
    ar_h = [asr_whisper(lm, lp, p["audio"]) for p in fleurs]; sync(); calc = time.time() - t0
    HYP["WHISPER_L3"] = {"ar": ar_h, "fr": None}
    RES["systemes"]["WHISPER_L3"] = {"description": "openai/whisper-large-v3, ASR arabe seul (fp16, T4)", "temps_chargement_s": round(charge, 1),
                                     "temps_calcul_s": round(calc, 1), "rtf": round(calc / duree_totale, 4), "pic_vram_go": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    mesurer_ar("WHISPER_L3", ar_h); print(RES["systemes"]["WHISPER_L3"])
    del lm; liberer(); sauver()
except Exception as e:
    import traceback; traceback.print_exc(); RES["systemes"]["WHISPER_L3"] = {"erreur": f"{type(e).__name__}: {e}"}

# %% [markdown]
# ## S3 : encodeur Cohere gelé + projecteur MLP + Qwen2.5-1.5B (cascade)
# Précisions surchargées pour la T4 (fp16 au lieu de bf16) dans une copie de la configuration, pas dans le dépôt.

# %%
try:
    import yaml
    jeton = kc.jeton_hf()
    if not jeton:
        raise PermissionError("secret Kaggle HF_TOKEN absent : attacher le secret HF_TOKEN à ce kernel (Add-ons > Secrets), puis relancer")
    os.environ["HF_TOKEN"] = jeton
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download("CohereLabs/cohere-transcribe-arabic-07-2026", "config.json", token=jeton)
    except Exception as e:
        raise PermissionError(f"accès refusé à CohereLabs/cohere-transcribe-arabic-07-2026 ({type(e).__name__}) : S3 sauté")
    cfg_s3 = yaml.safe_load((RACINE / "configs" / "s3.yaml").read_text(encoding="utf-8"))
    for sec in ("temps_1_pilote", "temps_1_50k"): cfg_s3[sec]["precision_llm"] = "float16"
    chemin_cfg = SORTIE / "s3_kaggle_fp16.yaml"; chemin_cfg.write_text(yaml.safe_dump(cfg_s3, allow_unicode=True), encoding="utf-8")
    import transcrire_s3
    transcrire_s3.CHEMIN_CONFIG_S3 = chemin_cfg
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    s3 = transcrire_s3.TranscripteurS3.charger("temps_1_50k", RACINE / "modeles" / "s3_temps1_50k" / "projecteur_meilleur.pt")
    sync(); charge = time.time() - t0
    s3.transcrire_signal(fleurs[0]["audio"]); sync()
    ar_h, fr_h, t_asr, t_mt = [], [], 0.0, 0.0
    for p in fleurs:
        sync(); t0 = time.time(); segs = s3.transcrire_signal(p["audio"]); sync(); t1 = time.time()
        textes = [s["texte"] for s in segs]; fr = s3.traduire(textes); sync(); t2 = time.time()
        t_asr += t1 - t0; t_mt += t2 - t1
        ar_h.append(" ".join(textes).strip()); fr_h.append(" ".join(f for f in fr if f).strip())
    HYP["S3"] = {"ar": ar_h, "fr": fr_h}
    RES["systemes"]["S3"] = {"description": "Cohere Transcribe Arabic (encodeur gelé, fp16) + projecteur temps 1 (pas 1500) + Qwen2.5-1.5B fp16, cascade texte",
                             "temps_chargement_s": round(charge, 1), "temps_calcul_s": round(t_asr + t_mt, 1), "rtf": round((t_asr + t_mt) / duree_totale, 4),
                             "rtf_transcription": round(t_asr / duree_totale, 4), "rtf_traduction": round(t_mt / duree_totale, 4),
                             "pic_vram_go": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    mesurer_ar("S3", ar_h); print(RES["systemes"]["S3"])
    del s3; liberer(); sauver()
except Exception as e:
    import traceback; traceback.print_exc(); RES["systemes"]["S3"] = {"erreur": f"{type(e).__name__}: {e}", "sauté": True}
    sauver()

# %% [markdown]
# ## BLEU, chrF++, COMET et bootstrap apparié

# %%
refs = [p["fr"] for p in fleurs]; srcs = [p["ar"] for p in fleurs]
for s, h in HYP.items():
    if h.get("fr") is None: continue
    RES["systemes"][s].update(kc.metriques_traduction(refs, h["fr"]))
    h["chrf"] = [kc.chrf_phrase(r, x) for r, x in zip(refs, h["fr"])]
sauver()
for s, h in HYP.items():
    if h.get("fr") is None: continue
    c = kc.comet(srcs, h["fr"], refs)
    RES["systemes"][s]["comet"] = c["comet"]
    if c["comet"] is None: RES["systemes"][s]["comet_motif"] = c.get("motif")
    else: h["comet"] = c["scores_segments"]
    sauver(); print(s, {k: RES["systemes"][s].get(k) for k in ("bleu", "chrf_pp", "comet")})
    liberer()
d = pd.read_csv(SORTIE / "traductions_par_phrase.csv")
for s, h in HYP.items():
    for k in ("chrf", "comet"):
        if h.get(k): d[f"{s}_{k}"] = h[k]
d.to_csv(SORTIE / "traductions_par_phrase.csv", index=False, encoding="utf-8")

# %%
from reechantillonnage import comparer_systemes_bootstrap
RES["bootstrap"] = []
NB_TIRAGES = 300
def comparer(a, b, cle, metrique, refs_):
    try:
        r = comparer_systemes_bootstrap(refs_, HYP[a][cle], HYP[b][cle], metrique, n_tirages=NB_TIRAGES)
        RES["bootstrap"].append({"a": a, "b": b, "metrique": metrique, "n_tirages": NB_TIRAGES, **{k: r[k] for k in r if k != "ecarts"}})
    except Exception as e:
        RES["bootstrap"].append({"a": a, "b": b, "metrique": metrique, "erreur": str(e)})
dispo_fr = [s for s in HYP if HYP[s].get("fr")]
for i, a in enumerate(dispo_fr):
    for b in dispo_fr[i + 1:]:
        for m in ("chrf++", "bleu"): comparer(a, b, "fr", m, refs)
dispo_ar = list(HYP)
for i, a in enumerate(dispo_ar):
    for b in dispo_ar[i + 1:]: comparer(a, b, "ar", "wer", srcs)
sauver()
print(pd.DataFrame(RES["systemes"]).T.drop(columns=["description"], errors="ignore").to_string())
