"""Valide un ou plusieurs fichiers JSONL de fiches (jeux T/R ou D).

Refuse toute fiche incomplete et affiche les codes d'erreur de chaque ligne.
Code de retour : 0 si toutes les fiches sont acceptees, 1 sinon.

Exemples :
  python valider_fiches.py --type tr tests/exemples/tr-ok-factice.jsonl
  python valider_fiches.py --type d  donnees/distille/d_v1.jsonl --rapport rejets-validation.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from seuils import CHEMIN_DEFAUT, charger_seuils
from validation import valider_fichier_jsonl

CODE_REJET = 1


def analyser_arguments(argv=None):
    analyseur = argparse.ArgumentParser(
        prog="valider_fiches",
        description="Valide des fichiers JSONL de fiches T/R ou D.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.splitlines()[2],
    )
    analyseur.add_argument(
        "fichiers", nargs="+",
        help="fichiers JSONL de fiches a valider",
    )
    analyseur.add_argument(
        "--type", dest="type_jeu", choices=("tr", "d", "a", "p"), required=True,
        help="'tr' pour les fiches des jeux T et R, 'd' pour le jeu distille, 'a' pour l'alignement, 'p' pour la presse",
    )

    analyseur.add_argument(
        "--seuils", default=None,
        help="chemin du fichier seuils.yaml (par defaut : celui a cote du script)",
    )
    analyseur.add_argument(
        "--rapport", default=None, metavar="CSV",
        help="ecrit aussi un CSV des erreurs et avertissements (separateur point-virgule)",
    )
    return analyseur.parse_args(argv)


def ecrire_rapport(chemin, resultats_par_fichier):
    with open(chemin, "w", encoding="utf-8-sig", newline="") as sortie:
        tableur = csv.writer(sortie, delimiter=";")
        tableur.writerow(["fichier", "ligne", "id", "severite", "code", "message"])
        for fichier, resultats in resultats_par_fichier:
            for resultat in resultats:
                for severite, liste in (("erreur", resultat["erreurs"]),
                                        ("avertissement", resultat["avertissements"])):
                    for code, message in liste:
                        tableur.writerow([fichier, resultat["ligne"], resultat["id"] or "",
                                          severite, code, message])


def main(argv=None):
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    arguments = analyser_arguments(argv)

    try:
        seuils = charger_seuils(arguments.seuils or CHEMIN_DEFAUT)
    except Exception as exc:
        print(f"ERREUR : {exc}")
        return CODE_REJET

    resultats_par_fichier = []
    total_valide = total_rejete = 0
    for chemin in arguments.fichiers:
        if not Path(chemin).is_file():
            print(f"ERREUR : fichier introuvable : {chemin}")
            return CODE_REJET
        resultats = valider_fichier_jsonl(chemin, arguments.type_jeu, seuils)
        resultats_par_fichier.append((chemin, resultats))
        valide = sum(1 for r in resultats if r["valide"])
        rejete = len(resultats) - valide
        total_valide += valide
        total_rejete += rejete
        print(f"{chemin} : {len(resultats)} fiche(s), {valide} valide(s), {rejete} rejetee(s)")
        for resultat in resultats:
            for severite, liste in (("ERREUR", resultat["erreurs"]),
                                    ("avertissement", resultat["avertissements"])):
                for code, message in liste:
                    print(f"  ligne {resultat['ligne']} ({resultat['id'] or 'sans id'}) "
                          f"{severite} {code} : {message}")

    if arguments.rapport:
        ecrire_rapport(arguments.rapport, resultats_par_fichier)
        print(f"rapport ecrit : {arguments.rapport}")

    print(f"total : {total_valide} valide(s), {total_rejete} rejetee(s)")
    return 0 if total_rejete == 0 else CODE_REJET


if __name__ == "__main__":
    sys.exit(main())
