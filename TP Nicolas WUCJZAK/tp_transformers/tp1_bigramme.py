# tp1 : du texte brut au modèle bigramme
# les données (encode, decode, train_data, get_batch...) sont définies au niveau du module
# pour que le tp4 puisse juste les importer sans tout réécrire

import os
from urllib.request import urlretrieve

import torch
import torch.nn as nn
from torch.nn import functional as F

CHEMIN = os.path.join(os.path.dirname(__file__), "..", "data", "tinyshakespeare.txt")
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if not os.path.exists(CHEMIN):
    os.makedirs(os.path.dirname(CHEMIN), exist_ok=True)
    urlretrieve(URL, CHEMIN)

# utf-8 explicite sinon sur certaines machines python prend l'encodage du système et ça casse
with open(CHEMIN, "r", encoding="utf-8") as f:
    text = f.read()

# vocab = tous les caractères distincts, triés pour que les ids soient les mêmes à chaque lancement
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join([itos[i] for i in ids])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]  # pas mélangé : la val c'est la fin du texte, jamais vue à l'entraînement


def choisir_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"  # sur mac c'est environ 2x plus rapide que le cpu même pour un petit modèle
    return "cpu"


def get_batch(split, batch_size=4, block_size=8, device="cpu"):
    source = train_data if split == "train" else val_data
    # on retire block_size sinon y (décalé de 1) pourrait déborder à la fin du texte
    starts = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i: i + block_size] for i in starts])
    y = torch.stack([source[i + 1: i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # table c x c : la ligne i donne directement les scores du caractère qui suit le caractère i
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # idx (B,T) -> logits (B,T,C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # cross_entropy veut (N, C), donc on aplatit : chaque position est juste une prédiction de plus
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # on ne garde que le dernier pas de temps, c'est lui qui prédit la suite
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def entrainer(model, n_steps, batch_size=32, block_size=8, lr=1e-3, device="cpu"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(n_steps):
        xb, yb = get_batch("train", batch_size, block_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return loss.item()


@torch.no_grad()
def loss_moyenne(model, split, n=200, batch_size=32, block_size=8, device="cpu"):
    return sum(model(*get_batch(split, batch_size, block_size, device))[1].item() for _ in range(n)) / n


def generer(model, n=300, device="cpu"):
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)  # id 0 = "\n", bon point de départ
    return decode(model.generate(ctx, n)[0].tolist())


def main():
    torch.manual_seed(1337)

    print("== étape 1 : le corpus")
    print("longueur :", len(text), "caractères")
    print(text[:1000])
    # un token = un caractère ici. le fichier n'est pas découpé en exemples, c'est un seul long texte,
    # avec des \n, de la ponctuation, et majuscules / minuscules qui sont des tokens différents

    print("\n== étape 2 : encode / decode")
    print(repr("".join(chars)))
    print("vocab_size =", vocab_size)
    message = "hii there"
    print(encode(message))
    print(decode(encode(message)))
    assert decode(encode(message)) == message  # les deux dicos sont bien inverses l'un de l'autre

    print("\n== étape 3 : contextes et cibles")
    print("train :", len(train_data), "| val :", len(val_data))
    block_size = 8
    x = train_data[:block_size]
    y = train_data[1:block_size + 1]
    for t in range(block_size):
        context = x[:t + 1]
        target = y[t]
        print(f"{context.tolist()} -> {target.item()}   ({decode(context.tolist())!r} -> {decode([target.item()])!r})")
    # un bloc de 8 donne 8 prédictions (contexte de 1 à 8 caractères)
    # t=3 : contexte "Firs" -> cible "t"

    print("\n== étape 4 : un mini-batch")
    xb, yb = get_batch("train")
    print(xb.shape, yb.shape)
    print(xb)
    print(yb)
    # une ligne = une séquence indépendante, une colonne = une position
    # B=32 -> plus de séquences en parallèle, gradient moins bruité. T=64 -> plus de contexte par séquence

    print("\n== étape 5 : le modèle bigramme")
    device = choisir_device()
    model = BigramLanguageModel(vocab_size).to(device)
    logits, loss = model(xb.to(device), yb.to(device))
    print("logits :", logits.shape, "| loss initiale :", round(loss.item(), 3))
    print("loss attendue pour un modèle uniforme : -ln(1/65) =", round(torch.log(torch.tensor(vocab_size)).item(), 3))
    # c x c parce que l'entrée est un caractère et la sortie un score pour chacun des 65 caractères
    # au tp4 cette table est coupée en deux : un embedding (vocab -> n_embd) et lm_head (n_embd -> vocab)

    print("\n== étapes 6 et 7 : générer, entraîner, regénérer")
    print("--- avant entraînement")
    print(generer(model, device=device))

    for total, steps in [(100, 100), (1000, 900), (10000, 9000)]:
        entrainer(model, steps, device=device)
        tr = loss_moyenne(model, "train", device=device)
        va = loss_moyenne(model, "val", device=device)
        print(f"\n--- après {total} étapes : loss train {tr:.3f} | val {va:.3f}")
        print(generer(model, device=device))

    # bilan : à 100 étapes ça ne bouge presque pas (4.5, lr de 1e-3 c'est petit), à 1000 la loss descend
    # à 3.7 mais le texte reste du bruit, à 10000 on plafonne vers 2.5 : des "mots" à peu près prononçables,
    # des retours à la ligne, des noms en majuscules, mais rien de cohérent. normal, le modèle ne voit qu'un
    # seul caractère de contexte. l'entraînement ne fait que changer les valeurs de la table 65x65.
    # (la loss initiale est au dessus de ln 65 parce que la table est initialisée en n(0,1), pas uniforme)


if __name__ == "__main__":
    main()
