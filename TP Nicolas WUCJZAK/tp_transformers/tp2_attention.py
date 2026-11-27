# tp2 : construire l'auto-attention pas à pas
# boucles -> produit matriciel -> masque + softmax -> attention apprise

import torch
import torch.nn as nn
from torch.nn import functional as F


def etape1_produit_matriciel():
    torch.manual_seed(42)
    a = torch.tril(torch.ones(3, 3))
    a = a / torch.sum(a, 1, keepdim=True)
    b = torch.randint(0, 10, (3, 2)).float()
    c = a @ b
    print("a =\n", a)
    print("b =\n", b)
    print("c =\n", c)
    # c2 (3e ligne) = moyenne des 3 lignes de b. chaque ligne de a somme à 1 parce qu'on a divisé
    # par le nombre de 1 de la ligne, donc c'est une moyenne et pas une somme
    print("c2 à la main :", b.mean(0))


def etape2_boucles(x):
    B, T, C = x.shape
    xbow = torch.zeros((B, T, C))
    for b in range(B):
        for t in range(T):
            xprev = x[b, :t + 1]  # (t+1, C) : (1,2) pour t=0, (4,2) pour t=3, (8,2) pour t=7
            xbow[b, t] = torch.mean(xprev, dim=0)  # dim 0 = les positions, on moyenne dans le temps
    return xbow


def head_attention(x, head_size=16, seed=0):
    # une tête d'attention "à la main", sans classe, comme sur la slide
    torch.manual_seed(seed)
    B, T, C = x.shape
    key = nn.Linear(C, head_size, bias=False)
    query = nn.Linear(C, head_size, bias=False)
    value = nn.Linear(C, head_size, bias=False)

    k = key(x)    # (B,T,16)
    q = query(x)  # (B,T,16)
    wei = q @ k.transpose(-2, -1)  # (B,T,16) @ (B,16,T) -> (B,T,T)
    wei = wei * head_size ** -0.5  # mise à l'échelle, voir étape 6
    tril = torch.tril(torch.ones(T, T))
    wei = wei.masked_fill(tril == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)

    v = value(x)  # (B,T,16)
    out = wei @ v  # (B,T,T) @ (B,T,16) -> (B,T,16)
    return out, wei


def main():
    print("== étape 1 : agréger par produit matriciel")
    etape1_produit_matriciel()

    print("\n== étape 2 : moyenne du passé avec des boucles")
    torch.manual_seed(1337)
    B, T, C = 4, 8, 2
    x = torch.randn(B, T, C)
    xbow = etape2_boucles(x)
    print(x[0])
    print(xbow[0])  # la 1ere ligne est identique, ensuite c'est la moyenne cumulée

    print("\n== étapes 3 et 4 : vectoriser")
    wei = torch.tril(torch.ones(T, T))
    wei = wei / wei.sum(1, keepdim=True)
    xbow2 = wei @ x  # (T,T) @ (B,T,C) : le (T,T) est broadcasté sur la dimension batch -> (B,T,C)
    print("version matrice == boucles :", torch.allclose(xbow, xbow2, atol=1e-6))

    tril = torch.tril(torch.ones(T, T))
    wei = torch.zeros((T, T))
    wei = wei.masked_fill(tril == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)  # exp(-inf) = 0 donc le futur a un poids nul, le reste se partage 1
    xbow3 = wei @ x
    print("version softmax == boucles :", torch.allclose(xbow, xbow3, atol=1e-6))
    print(wei)
    # intérêt de la version softmax : wei part de zéros ici, mais si on met des scores à la place
    # on obtient des poids qui dépendent des données. c'est exactement l'étape suivante

    print("\n== étape 5 : poids d'attention appris")
    torch.manual_seed(1337)
    B, T, C = 4, 8, 32
    x = torch.randn(B, T, C)
    out, wei = head_attention(x, head_size=16)
    print("out :", out.shape, "| wei :", wei.shape)
    print("wei[0] =")
    print(wei[0].detach().numpy().round(2))
    # la dernière ligne de wei[0] c'est le dernier token, le seul qui peut regarder les 8 positions.
    # on mélange v et pas k : k sert juste à dire "je contiens ça" pour calculer les scores,
    # v c'est l'info qu'on transmet vraiment. les séparer laisse le modèle apprendre les deux indépendamment

    print("\n== étape 6 : pourquoi diviser par racine de dh")
    head_size = 16
    k = torch.randn(B, T, head_size)
    q = torch.randn(B, T, head_size)
    raw = q @ k.transpose(-2, -1)
    scaled = raw * head_size ** -0.5
    print("variance brute :", round(raw.var().item(), 2), "| après échelle :", round(scaled.var().item(), 2))
    z = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])
    print(torch.softmax(z, dim=-1))
    print(torch.softmax(z * 8, dim=-1))  # quasi one-hot : toute l'attention part sur un seul token

    print("\n== bilan, vérifications")
    w = wei[0].detach()
    print("somme des lignes :", w.sum(-1))
    print("poids vers le futur (triangle sup) :", torch.triu(w, diagonal=1).abs().max().item())
    print("forme finale :", out.shape)
    # q = ce que le token cherche, k = ce que chaque token propose, v = ce qu'il transmet si on le choisit.
    # q.k donne la compatibilité, softmax en fait des poids, et la sortie est la moyenne pondérée des v


if __name__ == "__main__":
    main()
