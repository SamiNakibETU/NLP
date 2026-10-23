# %% [markdown]
# # Anatomie d'un LLM de traduction : Qwen2.5-1.5B-Instruct, de la tokenisation au LoRA
# Notebook pédagogique pour le projet « sous-titres français de journaux arabes ». On suit le chemin d'une phrase arabe :
# tokenisation, gabarit de conversation, logits du prochain token, échantillonnage (température, top-k, top-p), consigne système,
# exemples (few-shot), historique de conversation, puis une grille de 24 requêtes comparant le modèle de base et l'adaptateur LoRA.

# %%
import subprocess, sys, os, json, random, gc
def pip(*a): subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
pip("sacrebleu")
import torch, transformers, peft
print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.get_device_name(0))
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(next(p for p in Path("/kaggle/input").rglob("kaggle_commun.py")).parent))
import kaggle_commun as kc
RACINE = Path(kc.__file__).resolve().parent; kc.preparer_chemins(RACINE)
SORTIE = Path("/kaggle/working")
prompts = kc.charger_prompts(RACINE)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
MODELE = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODELE)
base = AutoModelForCausalLM.from_pretrained(MODELE, dtype=torch.float16).cuda()
adaptateur = next(Path("/kaggle/input").rglob("adapter_config.json")).parent
modele = PeftModel.from_pretrained(base, adaptateur).eval()
def tr(textes, ctx=None, lora=False, **opt):
    if lora: return kc.traduire(modele, tok, prompts, textes, ctx, **opt)
    with modele.disable_adapter(): return kc.traduire(modele, tok, prompts, textes, ctx, **opt)

# %% [markdown]
# ## 1. Tokenisation d'une phrase de journal
# Le modèle ne lit pas des lettres ni des mots mais des **tokens** (morceaux de mots appris par BPE sur un gros corpus).
# Un texte arabe coûte plus de tokens qu'un texte anglais de même sens : le vocabulaire de Qwen est dominé par l'anglais et le chinois.

# %%
phrase = "أعلنت وزارة الصحة اللبنانية أمس ارتفاع عدد الإصابات بالكوليرا في الشمال إلى 120 حالة."
ids = tok(phrase, add_special_tokens=False).input_ids
print(len(phrase.split()), "mots ->", len(ids), "tokens")
print([tok.decode([i]) for i in ids])
print(ids)
fr = "Le ministère libanais de la Santé a annoncé hier que le nombre de cas de choléra dans le Nord est passé à 120."
print("français :", len(fr.split()), "mots ->", len(tok(fr, add_special_tokens=False).input_ids), "tokens")

# %% [markdown]
# ## 2. Le gabarit de conversation (chat template)
# Un modèle « Instruct » est entraîné sur un format précis : ChatML, avec des jetons spéciaux `<|im_start|>` et `<|im_end|>`.
# Voici **exactement** ce que reçoit le modèle avec la consigne de la cascade S3 (`configs/s3.yaml`).

# %%
messages = kc.messages_traduction(prompts, phrase)
texte_modele = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(texte_modele)
print("=> %d tokens ; la génération commence après « <|im_start|>assistant »" % len(tok(texte_modele, add_special_tokens=False).input_ids))

# %% [markdown]
# ## 3. Logits du prochain token
# Le modèle ne produit pas un mot : il produit, pour chaque position, un vecteur de **logits** (un score par token du vocabulaire, ~152 000).
# `softmax` les transforme en probabilités (somme = 1) ; le décodage glouton prend l'`argmax`.

# %%
entree = tok(texte_modele, return_tensors="pt", add_special_tokens=False).to("cuda")
with torch.no_grad(), modele.disable_adapter():
    logits = modele(**entree).logits[0, -1].float()
print("taille du vocabulaire (logits) :", logits.shape[0])
p = torch.softmax(logits, dim=-1)
print("somme des probabilités :", round(p.sum().item(), 6))
top = torch.topk(p, 10)
print(pd.DataFrame({"rang": range(1, 11), "token": [tok.decode([i]) for i in top.indices], "id": top.indices.tolist(),
                    "logit": [round(logits[i].item(), 2) for i in top.indices], "proba": [round(x, 4) for x in top.values.tolist()]}).to_string(index=False))
print("argmax :", repr(tok.decode([int(torch.argmax(logits))])))
print("top-10 couvre", round(top.values.sum().item(), 3), "de la masse de probabilité")
pre = tr([phrase])[0]
print("premier token de la traduction gloutonne :", repr(tok.tokenize(pre)[0]), "| traduction :", pre)

# %% [markdown]
# ## 4. Température, top-k, top-p sur une même traduction
# - **Température T** : on divise les logits par T avant le softmax. T < 1 durcit la distribution (plus déterministe), T > 1 l'aplatit.
# - **top-k** : on ne garde que les k tokens les plus probables. **top-p** : on garde le plus petit ensemble de masse cumulée ≥ p.

# %%
for T in (0.2, 0.7, 1.2):
    pT = torch.softmax(logits / T, dim=-1); t5 = torch.topk(pT, 5)
    ent = -(pT * torch.log(pT + 1e-12)).sum().item()
    print(f"T={T}: top-1 {t5.values[0].item():.3f}, top-5 {t5.values.sum().item():.3f}, entropie {ent:.2f} nats ->", [tok.decode([i]) for i in t5.indices])
configs = [("glouton", dict(do_sample=False)), ("T=0.2", dict(do_sample=True, temperature=0.2)), ("T=0.7", dict(do_sample=True, temperature=0.7)),
           ("T=1.2", dict(do_sample=True, temperature=1.2)), ("T=0.7, top-k=5", dict(do_sample=True, temperature=0.7, top_k=5)),
           ("T=1.2, top-p=0.9", dict(do_sample=True, temperature=1.2, top_p=0.9))]
rows = []
for nom, opt in configs:
    for graine in (1, 2, 3) if opt.get("do_sample") else (0,):
        torch.manual_seed(graine); rows.append({"réglage": nom, "graine": graine, "traduction": tr([phrase], **opt)[0]})
pd.set_option("display.max_colwidth", 200); print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# ## 5. La consigne système
# Même phrase, deux consignes : celle de la cascade (traducteur strict) et une consigne « sous-titreur » qui impose une longueur courte.

# %%
sys_a = prompts["systeme"]
sys_b = "Tu es sous-titreur pour un journal télévisé. Traduis l'arabe en français courant, en une seule phrase de moins de 20 mots, sans commentaire."
long = "قال رئيس الوزراء في مؤتمر صحافي إن الحكومة ستقدم مشروع الموازنة إلى مجلس النواب خلال الأسبوع المقبل بعد الاتفاق مع صندوق النقد الدولي."
for nom, s in (("A (cascade S3)", sys_a), ("B (sous-titreur)", sys_b)):
    print(nom, "->", tr([long], systeme=s)[0])

# %% [markdown]
# ## 6. Zéro-shot contre few-shot (2 exemples dans la conversation)

# %%
opus = [json.loads(l) for l in open(next(Path("/kaggle/input").rglob("opus_ar_fr.jsonl")), encoding="utf-8")]
ex = [r for r in opus if r["partie"] == "val" and r["source"] == "News-Commentary" and 8 <= len(r["ar"].split()) <= 18][:2]
exemples = [{"ar": r["ar"], "fr": r["fr"]} for r in ex]
test = "تعهدت الحكومة بخفض العجز في الموازنة إلى ثلاثة في المئة من الناتج المحلي الإجمالي بحلول عام 2027."
print("zéro-shot :", tr([test])[0]); print("few-shot  :", tr([test], exemples=exemples)[0])
print("exemples fournis :", [e["fr"][:80] for e in exemples])

# %% [markdown]
# ## 7. Contrôle de l'historique : 0 ou 2 phrases de contexte
# Le pronom arabe de la phrase cible est ambigu en genre/personne ; le contexte précédent le résout. Les deux gabarits
# (`gabarit` et `gabarit_contexte`) viennent de `configs/s3.yaml`.

# %%
cas = [
    {"ctx": ["استقبل الرئيس السفيرة الفرنسية في القصر الجمهوري.", "ثم تحدثت مع الصحافيين."], "ar": "وقالت إنها ستعود إلى بيروت الأسبوع المقبل."},
    {"ctx": ["يتولى جيرار منصب وزير الخارجية منذ عامين.", "وهو معروف بمواقفه المتشددة."], "ar": "وأعلن أنه سيزور واشنطن الشهر المقبل."},
]
for c in cas:
    print("source :", c["ar"])
    for nom, lora in (("base", False), ("LoRA", True)):
        print(f"  {nom} 0 phrase  :", tr([c["ar"]], lora=lora)[0]); print(f"  {nom} 2 phrases :", tr([c["ar"]], [c["ctx"]], lora=lora)[0])

# %% [markdown]
# ## 8. Grille de 24 requêtes : base contre LoRA
# 16 phrases FLEURS (avec référence française, chrF++ par requête) et 8 phrases de journal télévisé écrites à la main
# (noms propres, chiffres, sigles, dialecte libanais, expression idiomatique). Les colonnes `*_fidelite`, `*_fluidite`, `*_noms` (1 à 5)
# sont vides : elles servent à la notation humaine.

# %%
fleurs = kc.charger_fleurs(avec_audio=False)
random.Random(7).shuffle(fleurs); fl16 = fleurs[:16]
jt = [("noms propres", "اجتمع الأمين العام للأمم المتحدة أنطونيو غوتيريش مع الرئيس الفرنسي إيمانويل ماكرون في باريس."),
      ("noms propres", "أعلن وزير الطاقة وليد فياض أن لبنان سيستورد كميات إضافية من الكهرباء من الأردن."),
      ("chiffres", "ارتفع سعر برميل النفط 3,5 في المئة ليبلغ 87 دولارا في بورصة نيويورك."),
      ("sigles", "قالت منظمة الصحة العالمية إن 120 شخصا أصيبوا بالكوليرا في الشمال، بحسب تقرير الأونكتاد وصندوق النقد الدولي."),
      ("lieu / sigle", "حذرت قوات اليونيفيل بعد سقوط قذيفة قرب الناقورة جنوب لبنان."),
      ("dialecte libanais", "شو القصة؟ ما في كهربا من الصبح وكل الناس عم تشتكي من الفاتورة، وهلق الوضع صعب كتير."),
      ("expression idiomatique", "رغم الوعود، ما زالت المفاوضات على كف عفريت."),
      ("expression idiomatique", "تحاول الحكومة ضرب عصفورين بحجر واحد عبر إقرار الموازنة والإصلاحات معا.")]
grille = [{"id": f"FLEURS-{r['id']}", "type": "FLEURS", "ar": r["ar"], "fr_ref": r["fr"]} for r in fl16] + \
         [{"id": f"JT-{i+1}", "type": t, "ar": a, "fr_ref": ""} for i, (t, a) in enumerate(jt)]
textes = [g["ar"] for g in grille]
b, l = tr(textes), tr(textes, lora=True)
for g, x, y in zip(grille, b, l):
    g["fr_base"], g["fr_lora"] = x, y
    g["chrf_base"] = kc.chrf_phrase(g["fr_ref"], x) if g["fr_ref"] else ""; g["chrf_lora"] = kc.chrf_phrase(g["fr_ref"], y) if g["fr_ref"] else ""
    for pre in ("base", "lora"):
        for col in ("fidelite", "fluidite", "noms"): g[f"{pre}_{col}"] = ""
df = pd.DataFrame(grille)
df.to_csv(SORTIE / "grille_24_requetes.csv", index=False, encoding="utf-8-sig")
avec_ref = df[df.fr_ref != ""]
print("chrF++ moyen par requête (16 FLEURS) : base", round(avec_ref.chrf_base.astype(float).mean(), 2), "| LoRA", round(avec_ref.chrf_lora.astype(float).mean(), 2))
print(df[["id", "type", "fr_base", "fr_lora", "chrf_base", "chrf_lora"]].to_string(index=False))
