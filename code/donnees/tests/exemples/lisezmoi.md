# Exemples factices pour les tests

Toutes les fiches de ce dossier sont **factice** : chaines, emissions, adresses
(domaine reserve `exemple.test`), textes et identifiants inventes pour faire
tourner les tests automatiques. Aucune donnee reelle (ni MGB-2, ni extrait de
JT) ne vit ici.

- `tr-ok-factice.jsonl` : une fiche T/R correcte, que le validateur accepte.
- `tr-ko-factice.jsonl` : trois fiches T/R fautives, que le validateur refuse
  (champ requis absent, valeur hors liste, fin avant debut).
- `d-ok-factice.jsonl` : une fiche D correcte, acceptee.
- `d-ko-factice.jsonl` : trois fiches D fautives, refusees (champ requis absent,
  valeur hors liste, score en chaine au lieu d'un nombre).

Les fichiers `ok` portent aussi un champ `note` inconnu du schema : le
validateur en fait un avertissement, pas un rejet — c'est voulu et teste.
