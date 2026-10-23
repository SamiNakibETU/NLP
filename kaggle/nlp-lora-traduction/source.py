# %% [markdown]
# # LoRA de traduction arabe -> français (Qwen2.5-1.5B-Instruct)
# Adaptateur entraîné sur ~20 000 paires OPUS (presse + un peu d'oral), prompt identique à celui de la cascade S3
# (`configs/s3.yaml` : `traduction_texte`), la moitié des exemples avec les 2 phrases arabes précédentes.
# Évaluation avant/après sur FLEURS test (ar_eg -> fr_fr) : BLEU, chrF++, COMET.

# %%
import subprocess, sys, os, json, time, random, math
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])   # torchao 0.10 incompatible avec peft récent
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sacrebleu", "jiwer"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "unbabel-comet"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytorch-lightning", "torchmetrics", "jsonargparse", "entmax", "sentencepiece"], check=True)
import torch, transformers, peft
print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.get_device_name(0))

# %%
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path("/kaggle/input").rglob("kaggle_commun.py")).parent))
import kaggle_commun as kc
RACINE = Path(kc.__file__).resolve().parent; kc.preparer_chemins(RACINE)
SORTIE = Path("/kaggle/working"); SORTIE.mkdir(exist_ok=True)
NB_TRAIN_MAX = 0   # essai rapide : mettre 400
GRAINE = 42
random.seed(GRAINE); torch.manual_seed(GRAINE)
prompts = kc.charger_prompts(RACINE)
print(prompts["systeme"]); print(repr(prompts["gabarit"])); print(repr(prompts["gabarit_contexte"]))
f_opus = next(Path("/kaggle/input").rglob("opus_ar_fr.jsonl"))
lignes = [json.loads(l) for l in open(f_opus, encoding="utf-8")]
train = [r for r in lignes if r["partie"] == "train"]; val = [r for r in lignes if r["partie"] == "val"]
if NB_TRAIN_MAX: train, val = train[:NB_TRAIN_MAX], val[:100]
print(len(train), len(val))

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELE = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODELE)
base = AutoModelForCausalLM.from_pretrained(MODELE, dtype=torch.float16).cuda()   # T4 : fp16 (pas de bf16 natif)
LONG_MAX = 384   # 256 rejetait 38 % des exemples (l'arabe coûte beaucoup de tokens, surtout avec contexte)

def exemple(r, avec_ctx):
    ctx = r["ctx"] if (avec_ctx and len(r["ctx"]) == 2) else None
    invite = tok.apply_chat_template(kc.messages_traduction(prompts, r["ar"], ctx), tokenize=False, add_generation_prompt=True)
    p = tok(invite, add_special_tokens=False).input_ids
    c = tok(r["fr"].strip() + "<|im_end|>", add_special_tokens=False).input_ids
    if len(p) + len(c) > LONG_MAX:
        return exemple(r, False) if ctx is not None else None   # trop long avec contexte : sans contexte
    return {"input_ids": p + c, "labels": [-100] * len(p) + c, "ctx": ctx is not None}

def preparer(rows, graine):
    h = random.Random(graine); sortie, rejetes = [], 0
    for r in rows:
        e = exemple(r, h.random() < 0.5)
        if e is None:
            rejetes += 1
        else:
            sortie.append(e)
    return sortie, rejetes
ex_train, rej_t = preparer(train, 1); ex_val, rej_v = preparer(val, 2)
print("exemples", len(ex_train), len(ex_val), "rejetés (> LONG_MAX tokens)", rej_t, rej_v,
      "| part avec contexte", sum(e["ctx"] for e in ex_train) / len(ex_train))
i0 = ex_train[0]; print(tok.decode(i0["input_ids"])); print("--- cible (non masquée) :", tok.decode([t for t in i0["labels"] if t != -100]))

# %%
def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), tok.pad_token_id); lab = torch.full((len(batch), n), -100); att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"]); ids[i, :L] = torch.tensor(b["input_ids"]); lab[i, :L] = torch.tensor(b["labels"]); att[i, :L] = 1
    return ids.cuda(), att.cuda(), lab.cuda()

# %% [markdown]
# ## Mesures AVANT (modèle de base) sur FLEURS test et sur la validation OPUS

# %%
fleurs = kc.charger_fleurs(avec_audio=False)
print(len(fleurs), "phrases FLEURS appariées ar/fr")
from peft import LoraConfig, get_peft_model
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
modele = get_peft_model(base, lora); modele.print_trainable_parameters()
for p in modele.parameters():
    if p.requires_grad:
        p.data = p.data.float()

def traduire_tout(rows, avec_ctx, adapte):
    textes = [r["ar"] for r in rows]; ctx = [r.get("ctx") if avec_ctx else None for r in rows]
    modele.eval()
    if adapte:
        return kc.traduire(modele, tok, prompts, textes, ctx)
    with modele.disable_adapter():
        return kc.traduire(modele, tok, prompts, textes, ctx)

t0 = time.time()
fl_avant = traduire_tout(fleurs, False, adapte=False)
val_ev = val[:200]
val_avant_sans = traduire_tout(val_ev, False, False); val_avant_avec = traduire_tout(val_ev, True, False)
print("évaluation avant :", round(time.time() - t0), "s")
print(kc.metriques_traduction([r["fr"] for r in fleurs], fl_avant))

# %% [markdown]
# ## Entraînement (1 époque, lot effectif 16, lr 2e-4, fp16)

# %%
MICRO, ACCUM, LR = 4, 4, 2e-4   # micro-lot 4 x accumulation 4 = lot effectif 16 (8 x 2 dépassait les 15 Go de la T4)
params = [p for p in modele.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
n_pas = len(ex_train) // (MICRO * ACCUM); warm = max(1, int(0.04 * n_pas))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, n_pas - warm))))
scaler = torch.amp.GradScaler("cuda")
ordre = list(range(len(ex_train))); random.Random(GRAINE).shuffle(ordre)

@torch.no_grad()
def perte_val():
    modele.eval(); tot, n = 0.0, 0
    for i in range(0, len(ex_val), 8):
        ids, att, lab = collate(ex_val[i:i + 8])
        with torch.autocast("cuda", dtype=torch.float16):
            lg = modele(input_ids=ids, attention_mask=att).logits
        l = torch.nn.functional.cross_entropy(lg[:, :-1].float().reshape(-1, lg.size(-1)), lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        tot += l.item(); n += (lab[:, 1:] != -100).sum().item()
    modele.train()
    return tot / n

journal = open(SORTIE / "metriques.jsonl", "w", encoding="utf-8")
def noter(**kw):
    journal.write(json.dumps(kw) + "\n"); journal.flush(); print(kw)
meilleure, pas_meilleur = float("inf"), 0
v0 = perte_val(); noter(pas=0, perte_val=round(v0, 4), type="val")
modele.save_pretrained(SORTIE / "adaptateur_lora")   # repli : l'adaptateur initial (B = 0) si la perte ne baisse jamais
modele.train(); t_train = time.time(); somme, cnt = 0.0, 0
for pas in range(1, n_pas + 1):
    for k in range(ACCUM):
        debut = ((pas - 1) * ACCUM + k) * MICRO
        ids, att, lab = collate([ex_train[j] for j in ordre[debut: debut + MICRO]])
        with torch.autocast("cuda", dtype=torch.float16):
            lg = modele(input_ids=ids, attention_mask=att).logits
        perte = torch.nn.functional.cross_entropy(lg[:, :-1].float().reshape(-1, lg.size(-1)), lab[:, 1:].reshape(-1), ignore_index=-100)
        scaler.scale(perte / ACCUM).backward(); somme += perte.item(); cnt += 1
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
    if pas % 20 == 0:
        noter(pas=pas, perte_train=round(somme / cnt, 4), lr=sched.get_last_lr()[0], temps_s=round(time.time() - t_train), type="train"); somme, cnt = 0.0, 0
    if pas % 100 == 0 or pas == n_pas:
        v = perte_val(); noter(pas=pas, perte_val=round(v, 4), type="val")
        if v < meilleure:
            meilleure, pas_meilleur = v, pas; modele.save_pretrained(SORTIE / "adaptateur_lora")
duree_train = time.time() - t_train
print("entraînement", round(duree_train / 60, 1), "min ; meilleur pas", pas_meilleur, "perte val", meilleure)
journal.close()

# %%
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
M = [json.loads(l) for l in open(SORTIE / "metriques.jsonl")]
tr = [m for m in M if m["type"] == "train"]; va = [m for m in M if m["type"] == "val"]
plt.figure(figsize=(7, 4)); plt.plot([m["pas"] for m in tr], [m["perte_train"] for m in tr], label="train (moy. 20 pas)")
plt.plot([m["pas"] for m in va], [m["perte_val"] for m in va], "o-", label="validation"); plt.axvline(pas_meilleur, ls=":", c="gray")
plt.xlabel("pas d'optimisation (lot 16)"); plt.ylabel("perte (tokens de la traduction)"); plt.legend(); plt.title("LoRA ar->fr, Qwen2.5-1.5B"); plt.tight_layout()
plt.savefig(SORTIE / "courbe_perte.png", dpi=150)

# %% [markdown]
# ## Mesures APRÈS (meilleur adaptateur)

# %%
from peft import PeftModel
del modele, base; torch.cuda.empty_cache()
base = AutoModelForCausalLM.from_pretrained(MODELE, dtype=torch.float16).cuda()
modele = PeftModel.from_pretrained(base, SORTIE / "adaptateur_lora").eval()
fl_apres = traduire_tout(fleurs, False, adapte=True)
val_apres_sans = traduire_tout(val_ev, False, True); val_apres_avec = traduire_tout(val_ev, True, True)
refs_fl = [r["fr"] for r in fleurs]; refs_v = [r["fr"] for r in val_ev]
res = {"modele": MODELE, "precision": "float16 (T4)", "nb_train": len(ex_train), "nb_val": len(ex_val), "pas_total": n_pas,
       "meilleur_pas": pas_meilleur, "perte_val_initiale": round(v0, 4), "perte_val_meilleure": round(meilleure, 4),
       "duree_entrainement_min": round(duree_train / 60, 1), "part_exemples_avec_contexte": round(sum(e["ctx"] for e in ex_train) / len(ex_train), 3),
       "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "lr": LR, "lot_effectif": MICRO * ACCUM},
       "fleurs": {"n": len(fleurs), "avant": kc.metriques_traduction(refs_fl, fl_avant), "apres": kc.metriques_traduction(refs_fl, fl_apres)},
       "opus_val_sans_contexte": {"n": len(val_ev), "avant": kc.metriques_traduction(refs_v, val_avant_sans), "apres": kc.metriques_traduction(refs_v, val_apres_sans)},
       "opus_val_avec_contexte": {"n": len(val_ev), "avant": kc.metriques_traduction(refs_v, val_avant_avec), "apres": kc.metriques_traduction(refs_v, val_apres_avec)}}
kc.ecrire_json(SORTIE / "resultats_lora.json", res)
import pandas as pd
pd.DataFrame({"id": [r["id"] for r in fleurs], "ar": [r["ar"] for r in fleurs], "fr_ref": refs_fl, "fr_base": fl_avant, "fr_lora": fl_apres,
              "chrf_base": [kc.chrf_phrase(a, b) for a, b in zip(refs_fl, fl_avant)], "chrf_lora": [kc.chrf_phrase(a, b) for a, b in zip(refs_fl, fl_apres)]}
             ).to_csv(SORTIE / "traductions_fleurs.csv", index=False, encoding="utf-8")
pd.DataFrame({"id": [r["id"] for r in val_ev], "source": [r["source"] for r in val_ev], "ar": [r["ar"] for r in val_ev], "fr_ref": refs_v,
              "base_sans_ctx": val_avant_sans, "lora_sans_ctx": val_apres_sans, "base_avec_ctx": val_avant_avec, "lora_avec_ctx": val_apres_avec}
             ).to_csv(SORTIE / "traductions_opus_val.csv", index=False, encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))

# %% [markdown]
# ## COMET (wmt22-comet-da) sur FLEURS, base puis LoRA

# %%
del modele, base; torch.cuda.empty_cache()
srcs = [r["ar"] for r in fleurs]
c_av = kc.comet(srcs, fl_avant, refs_fl); c_ap = kc.comet(srcs, fl_apres, refs_fl)
res["fleurs"]["avant"]["comet"] = c_av["comet"]; res["fleurs"]["apres"]["comet"] = c_ap["comet"]
if c_av["comet"] is None:
    res["fleurs"]["comet_motif"] = c_av.get("motif") or c_ap.get("motif")
kc.ecrire_json(SORTIE / "resultats_lora.json", res)
if c_av["comet"] is not None:
    d = pd.read_csv(SORTIE / "traductions_fleurs.csv"); d["comet_base"] = c_av["scores_segments"]; d["comet_lora"] = c_ap["scores_segments"]
    d.to_csv(SORTIE / "traductions_fleurs.csv", index=False, encoding="utf-8")
print(res["fleurs"])
