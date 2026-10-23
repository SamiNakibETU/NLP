"""Ajoute une référence française (text_fr) à un manifeste, par traduction automatique NLLB du texte arabe.

Les références sont approximatives (traduction automatique segment par segment, sans contexte).
Usage : python make_french.py manifest_dev_train.jsonl manifest_dev_train_fr.jsonl [--limit N]
"""
import argparse
import json

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src"); p.add_argument("dst")
    p.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--beams", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    rows = [json.loads(l) for l in open(a.src, encoding="utf-8") if l.strip()]
    rows = rows[: a.limit] if a.limit else rows
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, src_lang="arb_Arab")
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    fr_id = tok.convert_tokens_to_ids("fra_Latn")

    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["text_ar"]))  # lots de longueurs voisines
    for k in range(0, len(order), a.batch):
        idx = order[k:k + a.batch]
        enc = tok([rows[i]["text_ar"] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=200).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, forced_bos_token_id=fr_id, num_beams=a.beams, max_new_tokens=200)
        for i, t in zip(idx, tok.batch_decode(out, skip_special_tokens=True)):
            rows[i]["text_fr"] = t.strip()
        if (k // a.batch) % 20 == 0:
            print(f"{k + len(idx)}/{len(rows)}", flush=True)
    with open(a.dst, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("écrit", a.dst, len(rows), "segments")


if __name__ == "__main__":
    main()
