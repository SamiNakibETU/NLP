"""Affiche l'erreur (ou la fin) d'un journal de kernel Kaggle : python kaggle/lire_log.py <fichier.log>"""
import json, sys
t = open(sys.argv[1], encoding="utf-8", errors="replace").read()
try:
    out = "".join(x["data"] for x in json.loads(t))
except Exception:
    out = t
i = out.find("Exception encountered")
sys.stdout.reconfigure(encoding="utf-8")
print(out[-3000:] if i < 0 else out[max(0, i - 800):i + 3000])
