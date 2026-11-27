# tp3 : layernorm et mlp, les opérations qui agissent sur chaque token séparément

import torch
import torch.nn as nn


class LayerNorm1d:
    # version pédagogique, pas un nn.Module, gamma et beta ne sont pas entraînés ici
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        # dim=-1 : on normalise chaque token sur ses features (batchnorm ce serait dim=0)
        xmean = x.mean(dim=-1, keepdim=True)
        xvar = x.var(dim=-1, keepdim=True, unbiased=False)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # eps évite de diviser par 0 si le vecteur est constant
        # gamma et beta rendent au réseau la possibilité de changer l'échelle / le décalage s'il en a besoin
        return self.gamma * xhat + self.beta


class FeedForward(nn.Module):
    # réutilisé tel quel au tp4 (avec dropout là-bas)
    def __init__(self, n_embd, mult=4, activation=nn.ReLU, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, mult * n_embd),
            activation(),
            nn.Linear(mult * n_embd, n_embd),  # on revient à C pour pouvoir additionner au résiduel
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    print("== étapes 1 et 2 : layernorm, quel axe ?")
    torch.manual_seed(1337)
    module = LayerNorm1d(100)
    x_brut = torch.randn(32, 100)
    x = module(x_brut)
    # je parie sur la 2e ligne ~ (0, 1) puisqu'on normalise par token
    print("feature 0 sur les 32 tokens :", round(x[:, 0].mean().item(), 3), round(x[:, 0].std().item(), 3))
    print("token 0 sur ses 100 features :", round(x[0, :].mean().item(), 3), round(x[0, :].std().item(), 3))
    ref = nn.LayerNorm(100)(x_brut)
    print("pareil que nn.LayerNorm :", torch.allclose(module(x_brut), ref, atol=1e-6))

    print("\n== étape 3 : feed-forward")
    ffwd = FeedForward(100)
    entree = torch.randn(32, 100)
    delta = ffwd(entree)
    print(delta.shape)
    cachee = ffwd.net[1](ffwd.net[0](entree))  # sortie de la relu, (32, 400)
    print("activations positives après relu :", round((cachee > 0).float().mean().item(), 3))
    # environ la moitié : à l'init les pré-activations sont centrées en 0, relu coupe tout ce qui est négatif

    print("\n== étapes 4 et 5 : pré-norm + résiduel")
    B, T, n_embd = 4, 8, 32
    layer_norm = nn.LayerNorm(n_embd)
    ffwd = FeedForward(n_embd)
    x = torch.randn(B, T, n_embd)
    delta = ffwd(layer_norm(x))
    x_out = x + delta
    print(x.shape, delta.shape, x_out.shape)
    # le mlp ne mélange pas les tokens : il agit sur la dernière dim seulement, on peut le vérifier
    x2 = x.clone()
    x2[:, 0] += 10  # on modifie seulement le token 0
    diff = (ffwd(layer_norm(x2)) - delta).abs().sum(-1)
    print("tokens modifiés en sortie :", (diff > 1e-6).sum(1).tolist(), "(1 par séquence, donc pas de mélange)")

    print("\n== variantes")
    torch.manual_seed(0)
    x = torch.randn(B, T, n_embd)
    for nom, act in [("relu", nn.ReLU), ("gelu", nn.GELU)]:
        f = FeedForward(n_embd, activation=act)
        h = f.net[1](f.net[0](x))
        print(f"{nom} : activations > 0 = {(h > 0).float().mean():.2f}, sortie {tuple(f(x).shape)}")
    # gelu laisse passer un peu de négatif (min ~ -0.17), plus lisse que relu

    for mult in [2, 4, 8]:
        f = FeedForward(n_embd, mult=mult)
        print(f"c -> {mult}c -> c : {n_params(f)} paramètres")
    # ~ 2 * mult * c^2 : les paramètres grandissent linéairement avec le facteur d'expansion

    sans_res = ffwd(layer_norm(x))
    avec_res = x + sans_res
    print("sans résiduel :", tuple(sans_res.shape), "| avec :", tuple(avec_res.shape))
    print("corrélation entrée / sortie sans résiduel :", round(torch.corrcoef(torch.stack([x.flatten(), sans_res.flatten()]))[0, 1].item(), 3))
    print("corrélation entrée / sortie avec résiduel :", round(torch.corrcoef(torch.stack([x.flatten(), avec_res.flatten()]))[0, 1].item(), 3))
    # mêmes formes dans les deux cas, mais sans le résiduel l'info d'entrée est perdue (corrélation ~0).
    # avec le résiduel le bloc ne fait qu'ajouter une correction et le gradient a un chemin direct
    # vers les couches du dessous, c'est ce qui permet d'empiler plein de blocs

    # bilan : layernorm garde la dimension c (elle ne change que l'échelle), le mlp ne mélange pas les tokens
    # (c'est le rôle de l'attention), le résiduel garde l'entrée et facilite l'entraînement profond


if __name__ == "__main__":
    main()
