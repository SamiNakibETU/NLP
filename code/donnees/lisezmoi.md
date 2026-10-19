# Scripts de collecte des données

Code et configuration du plan de collecte (`plan-donnees.md`, version du
7 octobre 2026). Python 3 avec `pyyaml` : `pip install -r requirements.txt`
depuis la racine du projet.

## Ce que fait chaque fichier

| Fichier | Rôle |
|---|---|
| `seuils.yaml` | La seule source des seuils et constantes du plan. Les seuils marqués "(proposé)" dans le plan sont sous `proposes` ; ils attendent la validation du groupe. Ne pas modifier les valeurs sans modifier le plan. |
| `seuils.py` | Charge `seuils.yaml` et refuse un fichier incomplet ou mal typé. |
| `schemas.py` | Champs des fiches des jeux T/R et D, avec leurs valeurs admises. `fiche_vide("tr")` ou `fiche_vide("d")` donne un modèle de fiche vide. |
| `validation.py` | Valide une fiche ou un fichier JSONL entier ; renvoie erreurs (rejet) et avertissements. |
| `valider_fiches.py` | Outil en ligne de commande du validateur. |
| `normalisation.py` | Normalisation du texte arabe pour les mesures (WER, écart d'écoute, comptage de mots) : `normaliser_arabe`, `compter_mots_arabe`, `taux_ecart_ecoute`. Ne remplace jamais le texte stocké dans les fiches. |
| `tests/` | Tests automatiques (unittest) et fiches d'exemple **factice** (`tests/exemples/`). |

## Lancer le validateur

```
python valider_fiches.py --help
python valider_fiches.py --type tr chemin/vers/t.jsonl
python valider_fiches.py --type d  chemin/vers/d_v1.jsonl --rapport rejets-validation.csv
```

Code de retour 0 si toutes les fiches sont complètes, 1 sinon.

## Lancer les tests

Depuis ce dossier :

```
python -m unittest discover -s tests -v
```

55 tests : validateur (accepte une fiche correcte et refuse trois fiches
fautives, pour T/R comme pour D), normalisation arabe, garde-fou du fichier
de seuils.

## Conventions

- Fichiers et dossiers en minuscules, sans accents ni espaces ; les fichiers
  de données gardent les noms du plan (`p_v1.jsonl`...), les scripts Python
  utilisent des underscores (les tirets empêchent l'import).
- Aucun seuil codé en dur dans les scripts : tout passe par `seuils.yaml`.
- Aucun jeton dans le code : les secrets vivent dans un `.env` (voir
  `.env.exemple` à la racine), exclu du dépôt par `.gitignore`.
- L'audio et les textes de MGB-2 ne vont jamais dans ce dépôt.
