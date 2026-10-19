#!/usr/bin/env python3
"""Filtrage des textes de presse OPUS pour produire le Jeu P (p_v1.jsonl).

Applique les filtres de l'étape D4 du plan :
- Filtre 2 (langue) : la langue détectée de la traduction doit être le français (fr).
- Filtre 3 (rapport de longueur) : le ratio caractères fr / caractères ar doit
  être compris entre rapport_longueur_min et rapport_longueur_max de seuils.yaml.
- Filtre 6 (doublons exacts) : élimination des doublons exacts (on n'en garde qu'un).

Produit :
- donnees/presse/p_v1.jsonl : fiches filtrées avec id, source, id_origine, ar, fr.
- donnees/presse/rejets.csv : fiches rejetées avec identifiant, filtre, motif,
  valeur_mesuree, seuil_applique.

Garantit la stricte égalité : lignes_entree == gardees + rejetees.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import py3langid
from seuils import charger_seuils

# Motif de détection de caractères arabes dans la sortie
_MOTIF_ARABE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")


def verifier_filtre_langue(fr_texte):
    """Vérifie le filtre 2 : texte français non vide, sans arabe, classé 'fr'."""
    texte = fr_texte.strip()
    if not texte:
        return False, "texte_vide", "0", "> 0"

    # Vérification d'absence de caractères arabes dans la sortie française
    if _MOTIF_ARABE.search(texte):
        return False, "arabe_dans_sortie", "contient_arabe", "aucun_arabe"

    lang, _ = py3langid.classify(texte)
    if lang != "fr":
        return False, "langue_non_francaise", lang, "fr"

    return True, None, lang, "fr"


def verifier_filtre_ratio(ar_texte, fr_texte, ratio_min, ratio_max):
    """Vérifie le filtre 3 : rapport de longueur en caractères fr / ar."""
    len_ar = len(ar_texte.strip())
    len_fr = len(fr_texte.strip())

    if len_ar == 0:
        return False, "arabe_vide", "0", "> 0"

    ratio = len_fr / len_ar
    valeur_str = f"{ratio:.3f}"
    seuil_str = f"[{ratio_min}, {ratio_max}]"

    if ratio < ratio_min or ratio > ratio_max:
        return False, "rapport_longueur_hors_bornes", valeur_str, seuil_str

    return True, None, valeur_str, seuil_str


def filtrer_corpus(
    dossier_brut,
    fichier_sortie,
    fichier_rejets,
    ratio_min,
    ratio_max,
    essai=False,
    max_essai=100,
):
    """Traite les fichiers bruts de presse et applique les filtres D4."""
    dossier_brut = Path(dossier_brut)
    fichier_sortie = Path(fichier_sortie)
    fichier_rejets = Path(fichier_rejets)

    fichier_sortie.parent.mkdir(parents=True, exist_ok=True)
    fichier_rejets.parent.mkdir(parents=True, exist_ok=True)

    fichiers_bruts = sorted(dossier_brut.glob("*.jsonl"))
    if not fichiers_bruts:
        raise FileNotFoundError(f"Aucun fichier jsonl trouvé dans {dossier_brut}")

    statistiques = {}
    paires_vues = set()
    total_entree = 0
    total_gardees = 0
    total_rejetees = 0

    id_compteur = 1

    with open(fichier_sortie, "w", encoding="utf-8") as f_out, open(
        fichier_rejets, "w", encoding="utf-8", newline=""
    ) as f_rej:
        writer_rej = csv.writer(f_rej, lineterminator="\n")
        writer_rej.writerow(
            ["identifiant", "filtre", "motif", "valeur_mesuree", "seuil_applique"]
        )

        for chemin_brut in fichiers_bruts:
            nom_corpus = chemin_brut.stem
            stats_c = {
                "entree": 0,
                "rejet_langue": 0,
                "rejet_ratio": 0,
                "rejet_doublon": 0,
                "gardees": 0,
            }

            with open(chemin_brut, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    stats_c["entree"] += 1
                    total_entree += 1

                    data = json.loads(line)
                    source = data.get("source", nom_corpus)
                    id_orig = data.get("id_origine", stats_c["entree"])
                    ar = data.get("ar", "").strip()
                    fr = data.get("fr", "").strip()
                    identifiant_brut = f"{source}:{id_orig}"

                    # 1. Filtre 2 : Langue
                    ok_lang, motif, val, seuil = verifier_filtre_langue(fr)
                    if not ok_lang:
                        stats_c["rejet_langue"] += 1
                        total_rejetees += 1
                        writer_rej.writerow(
                            [identifiant_brut, "D4_2_langue", motif, val, seuil]
                        )
                        continue

                    # 2. Filtre 3 : Ratio de longueur
                    ok_ratio, motif, val, seuil = verifier_filtre_ratio(
                        ar, fr, ratio_min, ratio_max
                    )
                    if not ok_ratio:
                        stats_c["rejet_ratio"] += 1
                        total_rejetees += 1
                        writer_rej.writerow(
                            [identifiant_brut, "D4_3_rapport_longueur", motif, val, seuil]
                        )
                        continue

                    # 3. Filtre 6 : Doublons exacts
                    cle_doublon = (ar, fr)
                    if cle_doublon in paires_vues:
                        stats_c["rejet_doublon"] += 1
                        total_rejetees += 1
                        writer_rej.writerow(
                            [
                                identifiant_brut,
                                "D4_6_doublons",
                                "doublon_exact",
                                "deja_vu",
                                "unique",
                            ]
                        )
                        continue
                    paires_vues.add(cle_doublon)

                    # Paire retenue
                    stats_c["gardees"] += 1
                    total_gardees += 1

                    fiche_p = {
                        "id": f"p-{id_compteur:06d}",
                        "source": source,
                        "id_origine": id_orig,
                        "ar": ar,
                        "fr": fr,
                    }
                    f_out.write(json.dumps(fiche_p, ensure_ascii=False) + "\n")
                    id_compteur += 1

                    if total_entree % 50000 == 0:
                        print(
                            f"  ... {total_entree} paires traitées ({total_gardees} gardées, {total_rejetees} rejetées)",
                            flush=True,
                        )

                    if essai and total_gardees >= max_essai:
                        break

            statistiques[nom_corpus] = stats_c
            print(
                f"  [CORPUS {nom_corpus}] Entrée: {stats_c['entree']}, Gardées: {stats_c['gardees']}, Rejets: {stats_c['rejet_langue'] + stats_c['rejet_ratio'] + stats_c['rejet_doublon']}",
                flush=True,
            )
            if essai and total_gardees >= max_essai:
                break

    # Vérification stricte d'intégrité
    assert total_entree == (
        total_gardees + total_rejetees
    ), f"Incohérence : {total_entree} != {total_gardees} + {total_rejetees}"

    return {
        "total_entree": total_entree,
        "total_gardees": total_gardees,
        "total_rejetees": total_rejetees,
        "statistiques": statistiques,
        "fichier_sortie": fichier_sortie,
        "fichier_rejets": fichier_rejets,
    }


def calculer_sha256(chemin_fichier):
    """Calcule le hash SHA256 d'un fichier."""
    h = hashlib.sha256()
    with open(chemin_fichier, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def enregistrer_journal(res, seuils, chemin_journal="journal-donnees.md"):
    """Ajoute une entrée immuable dans journal-donnees.md."""
    journal_path = Path(chemin_journal)
    maintenant = time.strftime("%Y-%m-%d %H:%M")
    sha_p1 = calculer_sha256(res["fichier_sortie"])
    sha_rej = calculer_sha256(res["fichier_rejets"])

    entree = [f"\n## {maintenant}  Filtrage D4 Jeu P (p_v1.jsonl)"]
    entree.append(f"Commande : python code/donnees/filtrer_presse.py")
    entree.append(f"Entrée   : {res['fichier_sortie'].parent}/brut/ ({res['total_entree']} paires)")
    entree.append(f"Sorties  : {res['fichier_sortie']} (gardées) et {res['fichier_rejets']} (rejets)")
    entree.append(
        f"Résultat : {res['total_gardees']} gardées (sha256: {sha_p1}), "
        f"{res['total_rejetees']} rejetées (sha256: {sha_rej})"
    )
    entree.append(
        f"Seuils   : ratio [{seuils['proposes']['rapport_longueur_min']}, {seuils['proposes']['rapport_longueur_max']}], langue fr, dédoublonnage exact"
    )
    entree_texte = "\n".join(entree) + "\n"

    with open(journal_path, "a", encoding="utf-8") as f:
        f.write(entree_texte)
    print(f"[JOURNAL] Entrée ajoutée à {journal_path}")


def enregistrer_manifeste(res, chemin_manifeste="donnees/manifeste.csv"):
    """Ajoute p_v1.jsonl et rejets.csv dans donnees/manifeste.csv."""
    manifeste_path = Path(chemin_manifeste)
    date_jour = time.strftime("%Y-%m-%d")

    chemin_p1 = str(res["fichier_sortie"]).replace("\\", "/")
    taille_p1 = res["fichier_sortie"].stat().st_size
    sha_p1 = calculer_sha256(res["fichier_sortie"])

    ligne_p1 = (
        f"{chemin_p1},P,v1,OPUS (News-Commentary/GlobalVoices/TED2020/UNPC),"
        f"https://opus.nlpl.eu,{date_jour},Mixte recherche / CC,{res['total_gardees']},"
        f"{taille_p1},{sha_p1},filtrer_presse.py,master\n"
    )

    with open(manifeste_path, "a", encoding="utf-8") as f:
        f.write(ligne_p1)
    print(f"[MANIFESTE] Ligne ajoutée à {manifeste_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filtrage des corpus OPUS pour générer p_v1.jsonl et rejets.csv."
    )
    parser.add_argument(
        "--dossier-brut",
        default="donnees/presse/brut",
        help="Dossier contenant les fichiers bruts jsonl (défaut : donnees/presse/brut).",
    )
    parser.add_argument(
        "--sortie",
        default="donnees/presse/p_v1.jsonl",
        help="Fichier de sortie pour le jeu P (défaut : donnees/presse/p_v1.jsonl).",
    )
    parser.add_argument(
        "--rejets",
        default="donnees/presse/rejets.csv",
        help="Fichier de sortie des rejets (défaut : donnees/presse/rejets.csv).",
    )
    parser.add_argument(
        "--essai",
        action="store_true",
        help="Mode essai : s'arrête à 100 paires gardées.",
    )
    args = parser.parse_args()

    seuils = charger_seuils()
    ratio_min = seuils["proposes"]["rapport_longueur_min"]
    ratio_max = seuils["proposes"]["rapport_longueur_max"]

    print("=== Filtrage des corpus de presse (D4 : Filtres 2, 3, 6) ===")
    print(f"Seuils appliqués (seuils.yaml) : rapport longueur [{ratio_min}, {ratio_max}]")

    t0 = time.time()
    res = filtrer_corpus(
        args.dossier_brut,
        args.sortie,
        args.rejets,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
        essai=args.essai,
    )
    duree = time.time() - t0

    print(f"\nFiltrage terminé en {duree:.1f} s.")
    print(f"Total entrée   : {res['total_entree']}")
    print(f"Total gardées  : {res['total_gardees']}")
    print(f"Total rejetées : {res['total_rejetees']}")
    print("Contrôle intégrité : entrée == gardées + rejetées -> CONFORME.")

    print("\nDétail par corpus :")
    for nom, stats in res["statistiques"].items():
        print(f" - {nom} :")
        print(f"     Entrée         : {stats['entree']}")
        print(f"     Rejet langue   : {stats['rejet_langue']}")
        print(f"     Rejet ratio    : {stats['rejet_ratio']}")
        print(f"     Rejet doublon  : {stats['rejet_doublon']}")
        print(f"     Gardées        : {stats['gardees']}")

    if not args.essai:
        enregistrer_journal(res, seuils)
        enregistrer_manifeste(res)


if __name__ == "__main__":
    main()
