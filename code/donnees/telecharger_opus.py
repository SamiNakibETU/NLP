#!/usr/bin/env python3
"""Téléchargement et extraction des corpus OPUS pour le Jeu P (Presse).

Télécharge la paire arabe-français depuis OPUS pour :
- News-Commentary (presse internationale)
- GlobalVoices (actualités citoyennes)
- TED2020 (registre parlé / sous-titres)
- UNPC (documents de l'ONU, échantillonné avec graine fixée)

Fonctionnalités :
- Chaque script s'exécute en CLI avec --help et --essai
- Reprise automatique là où il s'est arrêté
- Seuils issus de code/donnees/seuils.yaml
- Trace écrite dans journal-donnees.md et donnees/manifeste.csv
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import time
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from seuils import charger_seuils

# Définition des corpus OPUS retenus pour le Jeu P
CORPUS_OPUS = {
    "News-Commentary": {
        "nom_fichier": "news_commentary",
        "version": "v16",
        "url": "https://object.pouta.csc.fi/OPUS-News-Commentary/v16/moses/ar-fr.txt.zip",
        "licence": "Libre recherche / CC",
        "echantillonner": False,
    },
    "GlobalVoices": {
        "nom_fichier": "global_voices",
        "version": "v2018q4",
        "url": "https://object.pouta.csc.fi/OPUS-GlobalVoices/v2018q4/moses/ar-fr.txt.zip",
        "licence": "CC-BY",
        "echantillonner": False,
    },
    "TED2020": {
        "nom_fichier": "ted2020",
        "version": "v1",
        "url": "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/ar-fr.txt.zip",
        "licence": "CC-BY-NC-ND",
        "echantillonner": False,
    },
    "UNPC": {
        "nom_fichier": "unpc",
        "version": "v20090831",
        "url": "https://object.pouta.csc.fi/OPUS-UN/v20090831/moses/ar-fr.txt.zip",
        "licence": "Domaine public / ONU",
        "echantillonner": True,
    },
}


def calculer_sha256(chemin_fichier):
    """Calcule le hash SHA256 d'un fichier."""
    h = hashlib.sha256()
    with open(chemin_fichier, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def telecharger_fichier(url, chemin_destination, timeout=60):
    """Télécharge un fichier avec reprise ou réutilisation s'il existe déjà."""
    chemin_destination = Path(chemin_destination)
    chemin_destination.parent.mkdir(parents=True, exist_ok=True)
    if chemin_destination.exists() and chemin_destination.stat().st_size > 0:
        print(f"  [CACHE] Archive déjà présente : {chemin_destination}")
        return chemin_destination

    print(f"  [TELECHARGEMENT] Depuis {url} vers {chemin_destination}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (NLP-Research-Project)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(chemin_destination, "wb") as out:
        taille_totale = int(resp.headers.get("Content-Length", 0))
        telecharge = 0
        t0 = time.time()
        while chunk := resp.read(65536):
            out.write(chunk)
            telecharge += len(chunk)
            if taille_totale > 0 and time.time() - t0 > 2.0:
                pct = (telecharge / taille_totale) * 100
                print(f"    ... {pct:.1f}% ({telecharge / (1024*1024):.1f} Mo)", end="\r")
                t0 = time.time()
    print(f"\n  [OK] Téléchargé avec succès ({chemin_destination.stat().st_size} octets)")
    return chemin_destination


def extraire_paires_zip(chemin_zip, essai=False, max_essai=50):
    """Extrait les paires de phrases arabes et françaises depuis une archive moses OPUS."""
    with zipfile.ZipFile(chemin_zip, "r") as z:
        noms = z.namelist()
        fichiers_ar = [n for n in noms if n.endswith(".ar")]
        fichiers_fr = [n for n in noms if n.endswith(".fr")]
        if not fichiers_ar or not fichiers_fr:
            raise ValueError(f"Fichiers .ar ou .fr introuvables dans {chemin_zip}")

        ar_name = fichiers_ar[0]
        fr_name = fichiers_fr[0]

        with z.open(ar_name) as ar_f, z.open(fr_name) as fr_f:
            lignes_ar = [line.decode("utf-8", errors="replace").strip() for line in ar_f]
            lignes_fr = [line.decode("utf-8", errors="replace").strip() for line in fr_f]

    if len(lignes_ar) != len(lignes_fr):
        raise ValueError(
            f"Désalignement dans l'archive {chemin_zip} : "
            f"{len(lignes_ar)} lignes ar vs {len(lignes_fr)} lignes fr"
        )

    return lignes_ar, lignes_fr


def traiter_corpus(
    nom_corpus,
    config,
    dossier_sortie,
    dossier_archives,
    max_unpc,
    graine,
    essai=False,
    forcer=False,
):
    """Télécharge et extrait un corpus donné en fichier JSONL brut."""
    fichier_sortie = dossier_sortie / f"{config['nom_fichier']}.jsonl"
    if fichier_sortie.exists() and fichier_sortie.stat().st_size > 0 and not forcer:
        print(f"  [REPRISE] {fichier_sortie} existe déjà. Ignoré (utiliser --forcer pour régénérer).")
        # Calcul des stats existantes
        nb = sum(1 for _ in open(fichier_sortie, "r", encoding="utf-8"))
        sha = calculer_sha256(fichier_sortie)
        return {
            "corpus": nom_corpus,
            "version": config["version"],
            "url": config["url"],
            "licence": config["licence"],
            "fichier": fichier_sortie,
            "nb_paires": nb,
            "taille": fichier_sortie.stat().st_size,
            "sha256": sha,
            "reutilise": True,
        }

    nom_archive = f"{config['nom_fichier']}_{config['version']}.zip"
    chemin_zip = dossier_archives / nom_archive
    telecharger_fichier(config["url"], chemin_zip)

    lignes_ar, lignes_fr = extraire_paires_zip(chemin_zip)
    total_disponible = len(lignes_ar)
    print(f"  [EXTRACTION] {total_disponible} paires lues dans l'archive.")

    indices = list(range(total_disponible))
    if config["echantillonner"]:
        plafond = 50 if essai else max_unpc
        if total_disponible > plafond:
            rng = random.Random(graine)
            indices = sorted(rng.sample(indices, plafond))
            print(f"  [ÉCHANTILLONNAGE] UNPC plafonné à {plafond} phrases (graine={graine}).")
    elif essai:
        indices = indices[:50]
        print(f"  [MODE ESSAI] Limité aux 50 premières paires.")

    # Écriture dans le fichier jsonl brut
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    with open(fichier_sortie, "w", encoding="utf-8") as out:
        for idx in indices:
            fiche = {
                "id_origine": idx + 1,  # 1-indexed
                "source": nom_corpus,
                "ar": lignes_ar[idx],
                "fr": lignes_fr[idx],
            }
            out.write(json.dumps(fiche, ensure_ascii=False) + "\n")

    nb_ecrit = len(indices)
    taille = fichier_sortie.stat().st_size
    sha = calculer_sha256(fichier_sortie)
    print(f"  [ECRITURE] {nb_ecrit} paires enregistrées dans {fichier_sortie} (sha256: {sha[:12]}...)")

    return {
        "corpus": nom_corpus,
        "version": config["version"],
        "url": config["url"],
        "licence": config["licence"],
        "fichier": fichier_sortie,
        "nb_paires": nb_ecrit,
        "taille": taille,
        "sha256": sha,
        "reutilise": False,
        "total_disponible": total_disponible,
    }


def enregistrer_journal(resultats, graine, chemin_journal="journal-donnees.md"):
    """Ajoute une entrée immuable dans journal-donnees.md."""
    journal_path = Path(chemin_journal)
    maintenant = time.strftime("%Y-%m-%d %H:%M")
    entree = [f"\n## {maintenant}  Téléchargement OPUS Jeu P"]
    entree.append(f"Commande : python code/donnees/telecharger_opus.py (graine: {graine})")
    for r in resultats:
        entree.append(f"--- {r['corpus']} ({r['version']}) ---")
        entree.append(f"Entrée   : {r['url']}")
        entree.append(f"Sortie   : {r['fichier']}")
        entree.append(f"Résultat : {r['nb_paires']} paires, taille {r['taille']} octets, sha256 {r['sha256']}")
        if "total_disponible" in r and r.get("total_disponible") != r["nb_paires"]:
            entree.append(f"Remarque : échantillonné depuis {r['total_disponible']} paires disponibles")
    entree_texte = "\n".join(entree) + "\n"

    with open(journal_path, "a", encoding="utf-8") as f:
        f.write(entree_texte)
    print(f"[JOURNAL] Entrée ajoutée à {journal_path}")


def enregistrer_manifeste(resultats, chemin_manifeste="donnees/manifeste.csv"):
    """Ajoute les fichiers produits dans donnees/manifeste.csv."""
    manifeste_path = Path(chemin_manifeste)
    date_jour = time.strftime("%Y-%m-%d")

    lignes = []
    for r in resultats:
        chemin_relatif = str(r["fichier"]).replace("\\", "/")
        ligne = (
            f"{chemin_relatif},P,{r['version']},{r['corpus']},{r['url']},"
            f"{date_jour},{r['licence']},{r['nb_paires']},{r['taille']},{r['sha256']},"
            f"telecharger_opus.py,master\n"
        )
        lignes.append(ligne)

    with open(manifeste_path, "a", encoding="utf-8") as f:
        f.writelines(lignes)
    print(f"[MANIFESTE] {len(lignes)} lignes ajoutées à {manifeste_path}")


def main():
    parser = argparse.ArgumentParser(description="Télécharger et extraire les corpus OPUS pour le Jeu P.")
    parser.add_argument(
        "--corpus",
        choices=["News-Commentary", "GlobalVoices", "TED2020", "UNPC", "tous"],
        default="tous",
        help="Corpus à télécharger (défaut : tous).",
    )
    parser.add_argument(
        "--dossier-sortie",
        default="donnees/presse/brut",
        help="Dossier où stocker les fichiers jsonl bruts (défaut : donnees/presse/brut).",
    )
    parser.add_argument(
        "--dossier-archives",
        default="donnees/presse/brut/archives",
        help="Dossier où stocker les archives zip (défaut : donnees/presse/brut/archives).",
    )
    parser.add_argument(
        "--graine",
        type=int,
        default=42,
        help="Graine aléatoire pour le tirage de UNPC (défaut : 42).",
    )
    parser.add_argument(
        "--essai",
        action="store_true",
        help="Mode essai : limite à 50 paires par corpus.",
    )
    parser.add_argument(
        "--forcer",
        action="store_true",
        help="Forcer le re-téléchargement et la ré-extraction même si les fichiers existent.",
    )

    args = parser.parse_args()

    # Lecture des seuils depuis seuils.yaml
    seuils = charger_seuils()
    max_unpc = seuils["constantes"]["unpc_max_phrases"]

    dossier_sortie = Path(args.dossier_sortie)
    dossier_archives = Path(args.dossier_archives)

    corpus_a_traiter = (
        list(CORPUS_OPUS.keys()) if args.corpus == "tous" else [args.corpus]
    )

    resultats = []
    print(f"=== Téléchargement OPUS Jeu P (Mode: {'ESSAI' if args.essai else 'COMPLET'}) ===")
    for nom in corpus_a_traiter:
        print(f"\nTraitement du corpus : {nom}")
        res = traiter_corpus(
            nom,
            CORPUS_OPUS[nom],
            dossier_sortie,
            dossier_archives,
            max_unpc=max_unpc,
            graine=args.graine,
            essai=args.essai,
            forcer=args.forcer,
        )
        resultats.append(res)

    if not args.essai:
        enregistrer_journal(resultats, graine=args.graine)
        enregistrer_manifeste(resultats)

    print("\n=== Terminé avec succès ===")
    for r in resultats:
        print(f" - {r['corpus']}: {r['nb_paires']} paires dans {r['fichier']}")


if __name__ == "__main__":
    main()
