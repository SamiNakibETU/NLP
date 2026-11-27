# lancer avec : uv run python -m tp_transformers 1   (ou 2, 3, 4, all)
# --rapide pour un tp4 qui tourne en une minute juste pour tester

import argparse

from . import tp1_bigramme, tp2_attention, tp3_layernorm_mlp, tp4_modele

parser = argparse.ArgumentParser()
parser.add_argument("tp", choices=["1", "2", "3", "4", "all"], nargs="?", default="all")
parser.add_argument("--rapide", action="store_true")
args = parser.parse_args()

tps = {"1": tp1_bigramme.main, "2": tp2_attention.main, "3": tp3_layernorm_mlp.main,
       "4": lambda: tp4_modele.main(rapide=args.rapide)}

for num in (tps if args.tp == "all" else [args.tp]):
    print(f"\n{'#' * 20} tp{num} {'#' * 20}\n")
    tps[num]()
