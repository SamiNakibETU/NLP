# tp4 : assembler le mini-transformer, l'entraîner et générer
# les données viennent du tp1 et le mlp du tp3, on ne les réécrit pas

import os
import time
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
from torch.nn import functional as F

from .tp1_bigramme import vocab_size, decode, get_batch, choisir_device
from .tp3_layernorm_mlp import FeedForward

SORTIES = os.path.join(os.path.dirname(__file__), "..", "outputs")


# au lieu de constantes globales comme sur la slide je mets tout dans une config,
# sinon impossible de tester block_size=16 / 64 etc sans tout relancer à la main
@dataclass
class Config:
    batch_size: int = 16
    block_size: int = 32
    max_iters: int = 5000
    eval_interval: int = 100
    learning_rate: float = 1e-3
    eval_iters: int = 200
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.0
    device: str = choisir_device()

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0  # sinon les têtes concaténées ne retombent pas sur n_embd


class Head(nn.Module):
    def __init__(self, cfg, head_size):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        # buffer et pas paramètre : ça suit le .to(device) mais ce n'est pas entraîné
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape                      # (16,32,64)
        k, q = self.key(x), self.query(x)      # (16,32,16) chacun
        # k.size(-1) = head_size, c'est la dim du produit scalaire q.k, pas C
        wei = q @ k.transpose(-2, -1) * k.size(-1) ** -0.5   # (16,32,32)
        # [:T,:T] car en génération T peut être plus petit que block_size
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)                      # (16,32,16)
        return wei @ v                         # (16,32,16)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)  # mélange ce que les différentes têtes ont trouvé
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # 4 x (B,T,16) -> (B,T,64), on concatène sur les features (dim -1), pas sur le temps
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.sa = MultiHeadAttention(cfg, cfg.n_head, head_size)
        self.ffwd = FeedForward(cfg.n_embd, dropout=cfg.dropout)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        # pré-norm : layernorm -> sous-couche -> addition résiduelle
        x = x + self.sa(self.ln1(x))    # communication entre les tokens
        x = x + self.ffwd(self.ln2(x))  # calcul, chaque token de son côté
        return x


class BigramLanguageModel(nn.Module):
    # le nom vient du cours, mais ce n'est plus du tout un bigramme évidemment
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(vocab_size, cfg.n_embd)        # (65, 64)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)  # (32, 64)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)                         # (64, 65)
        # au tp1 la table faisait les deux d'un coup (65x65). maintenant entre les deux il y a les blocs,
        # le vecteur de sortie n'est plus "le caractère" mais le contexte, donc il faut une projection à part

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                                     # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))   # (T,C)
        x = tok_emb + pos_emb   # broadcasting : (T,C) est recopié pour chaque élément du batch
        x = self.blocks(x)      # (B,T,C)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None  # en génération on garde (B,T,V) pour pouvoir prendre le dernier pas de temps
        else:
            logits = logits.reshape(B * T, vocab_size)
            targets = targets.reshape(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()  # pas besoin de garder le graphe pour le backward, ça économise mémoire et temps
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]  # les embeddings de position s'arrêtent à block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # multinomial et pas argmax : avec argmax on tombe vite dans des boucles "the the the..."
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model, cfg):
    out = {}
    model.eval()  # coupe le dropout pendant l'éval
    for split in ["train", "val"]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            X, Y = get_batch(split, cfg.batch_size, cfg.block_size, cfg.device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()  # et on le remet pour l'entraînement
    return out


def generer(model, n):
    ctx = torch.zeros((1, 1), dtype=torch.long, device=model.cfg.device)
    return decode(model.generate(ctx, n)[0].tolist())


def entrainer(cfg, verbose=True, seed=1337):
    torch.manual_seed(seed)
    model = BigramLanguageModel(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    hist = []
    t0 = time.time()
    for iteration in range(cfg.max_iters):
        if iteration % cfg.eval_interval == 0 or iteration == cfg.max_iters - 1:
            l = estimate_loss(model, cfg)
            hist.append((iteration, l["train"], l["val"]))
            if verbose:
                print(f"iter {iteration:5d} | train {l['train']:.4f} | val {l['val']:.4f} | {time.time() - t0:.0f}s")
        xb, yb = get_batch("train", cfg.batch_size, cfg.block_size, cfg.device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model, hist


def tracer(hists, nom_fichier, titre):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    for label, h in hists.items():
        it = [a for a, _, _ in h]
        ligne, = plt.plot(it, [b for _, b, _ in h], label=f"{label} train")
        plt.plot(it, [c for _, _, c in h], "--", color=ligne.get_color(), label=f"{label} val")
    plt.xlabel("itération")
    plt.ylabel("loss")
    plt.title(titre)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(SORTIES, nom_fichier), dpi=120)
    plt.close()


def main(rapide=False):
    os.makedirs(SORTIES, exist_ok=True)
    cfg = Config()
    suffixe = "_rapide" if rapide else ""  # pour ne pas écraser les sorties du vrai run
    if rapide:
        cfg = replace(cfg, max_iters=500, eval_iters=20)
    print("device :", cfg.device)
    print("dh =", cfg.n_embd // cfg.n_head, "| tokens par batch =", cfg.batch_size * cfg.block_size)
    # sur cpu je réduirais d'abord max_iters (et eval_iters) pour tester que tout tourne

    print("\n== vérif des formes")
    model = BigramLanguageModel(cfg).to(cfg.device)
    xb, yb = get_batch("train", cfg.batch_size, cfg.block_size, cfg.device)
    print("x :", tuple(xb.shape))
    h = model.blocks[0].sa.heads[0]
    emb = model.token_embedding_table(xb)
    print("une tête :", tuple(h(emb).shape), "| multi-têtes :", tuple(model.blocks[0].sa(emb).shape))
    logits, loss = model(xb)
    print("logits :", tuple(logits.shape), "| loss initiale :", round(model(xb, yb)[1].item(), 3), "(~ ln 65 = 4.17)")
    print("paramètres :", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    print("\n== test très court (200 itérations) pour voir que la loss descend")
    entrainer(replace(cfg, max_iters=200, eval_interval=100, eval_iters=20))

    print("\n== génération avant entraînement")
    torch.manual_seed(0)
    avant = generer(model, 300)
    print(avant)

    print(f"\n== entraînement complet ({cfg.max_iters} itérations)")
    model, hist = entrainer(cfg)
    tracer({"base": hist}, f"tp4_losses{suffixe}.png", "mini-transformer, config de base")

    print("\n== génération après entraînement (2000 caractères)")
    torch.manual_seed(0)
    apres = generer(model, 2000)
    print(apres)
    with open(os.path.join(SORTIES, f"tp4_generation{suffixe}.txt"), "w", encoding="utf-8") as f:
        f.write("--- avant entraînement\n" + avant + "\n\n--- après entraînement\n" + apres + "\n")
    torch.save(model.state_dict(), os.path.join(SORTIES, f"tp4_modele{suffixe}.pt"))

    experiences(cfg, rapide)


def experiences(cfg, rapide=False):
    print("\n== expériences")
    # 2000 itérations chacune pour que ça ne dure pas 1h, on compare juste entre elles
    base_exp = replace(cfg, max_iters=500 if rapide else 2000, eval_interval=500, eval_iters=50)
    variantes = {
        "base (T=32, 4 têtes)": base_exp,
        "block_size 16": replace(base_exp, block_size=16),
        "block_size 64": replace(base_exp, block_size=64),
        "1 tête": replace(base_exp, n_head=1),
        "dropout 0.2": replace(base_exp, dropout=0.2),
    }
    resultats, hists = {}, {}
    for nom, c in variantes.items():
        t0 = time.time()
        _, h = entrainer(c, verbose=False)
        hists[nom] = h
        resultats[nom] = (h[-1][1], h[-1][2], time.time() - t0)
        print(f"{nom:22s} | train {h[-1][1]:.3f} | val {h[-1][2]:.3f} | écart {h[-1][2] - h[-1][1]:+.3f} | {time.time() - t0:.0f}s")
    tracer(hists, "tp4_experiences" + ("_rapide" if rapide else "") + ".png", "variantes (2000 itérations)")

    # ce que j'obtiens (mps, 2000 itérations) :
    #   base         train 1.85 val 1.94
    #   block 16     train 1.94 val 1.99  -> moins de contexte, clairement moins bien
    #   block 64     train 1.82 val 1.94  -> le train baisse mais pas la val, l'écart grandit (début de sur-apprentissage)
    #   1 tête       train 1.84 val 1.93  -> pareil que 4 têtes ici, et 2x plus rapide. à cette taille et avec si peu
    #                                        d'itérations ça ne change rien, c'est sur des modèles plus gros que ça compte
    #   dropout 0.2  train 1.96 val 2.01  -> l'écart train / val diminue mais la loss est plus haute : on ne sur-apprend
    #                                        pas encore vraiment à 2000 itérations, donc le dropout freine plus qu'il n'aide
    # sur l'entraînement complet (5000 it) on finit à train 1.66 / val 1.82. l'écart s'ouvre doucement, c'est là
    # qu'un peu de dropout deviendrait utile. si la val remontait pendant que le train baisse ce serait du
    # sur-apprentissage franc, il faudrait arrêter plus tôt ou régulariser.
    #
    # le texte généré : avant c'est de la soupe de caractères, après on a la structure d'une pièce (NOM: puis
    # des vers, retours à la ligne, ponctuation), des vrais mots courts et des noms presque justes
    # ("BUCKINGHARD II", "DUKEMb OF YORK"), mais pas de sens. normal pour 0.2m de paramètres et 32 caractères de contexte.
    #
    # bilan des 4 tp :
    #   - Head / MultiHeadAttention : crée le contexte, c'est le seul endroit où les tokens échangent de l'info
    #   - FeedForward (+ LayerNorm) : transforme chaque token individuellement
    #   - token / position embedding : donnent l'identité du caractère et sa place
    #   - lm_head : produit les logits sur les 65 caractères
    #   - la cross entropy est calculée à la toute fin, mais via le backward (et les résiduels) son gradient
    #     remonte dans tous les blocs jusqu'aux embeddings, c'est là qu'elle agit sur tout le modèle


if __name__ == "__main__":
    main()
