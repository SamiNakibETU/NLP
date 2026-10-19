#!/usr/bin/env python3
"""Extraction des entités nommées (noms propres) depuis Wikidata via SPARQL et API.

Fonctionnalités :
- Catégories cibles :
  * personne : chefs d'État, chefs de gouvernement, ministres des Affaires étrangères
  * organisation : organisations internationales, grands partis politiques
  * lieu : pays, capitales et métropoles du monde arabe et de l'actualité
- Récupère le libellé arabe, les variantes d'écriture arabes (alias),
  le libellé français, le type et l'identifiant Wikidata.
- Taille ciblée : 300 à 500 entrées sans doublon (seuils.yaml).
- Sorties :
  * donnees/noms/noms_v1.csv
  * donnees/noms/noms_manuel_modele.csv (modèle d'accueil des 100 noms 2026)
  * donnees/noms/lisezmoi.md
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from seuils import charger_seuils

HEADERS = {
    "User-Agent": "NLP-Translation-Research/1.0 (contact: student@cy-tech.fr; project: arabic-french-news-subtitles)"
}

# QIDs d'amorce incontournables de l'actualité pour garantir la représentativité
AMORCES_PRIORITAIRES = {
    "personne": [
        "Q3052772",  # Emmanuel Macron
        "Q6279",     # Joe Biden
        "Q22686",    # Donald Trump
        "Q7747",     # Vladimir Poutine
        "Q15031",    # Xi Jinping
        "Q39259",    # Recep Tayyip Erdogan
        "Q57613",    # Abdelmadjid Tebboune
        "Q318460",   # Abdel Fattah al-Sissi
        "Q57553",    # Mohammed VI
        "Q207802",   # Mohammed ben Salmane
        "Q350616",   # Mohammed ben Zayed
        "Q57434",    # Abdallah II de Jordanie
        "Q57315",    # Tamim ben Hamad Al Thani
        "Q57592",    # Haïtham ben Tariq
        "Q57545",    # Hamad ben Issa Al Khalifa
        "Q44329",    # Bachar el-Assad
        "Q57380",    # Mahmoud Abbas
        "Q16120878", # Kaïs Saïed
        "Q43314",    # Olaf Scholz
        "Q4886678",  # Keir Starmer
        "Q3879",     # Benjamin Netanyahou
        "Q1333065",  # Jean-Noël Barrot
        "Q26207198", # Stéphane Séjourné
        "Q162638",   # Antony Blinken
        "Q58217",    # Sergueï Lavrov
        "Q57327",    # Wang Yi
        "Q7055745",  # Hakan Fidan
        "Q7144208",  # Ahmed Attaf
        "Q2944746",  # Badr Abdelatty
        "Q15729113", # Nasser Bourita
        "Q7128509",  # Faisal bin Farhan Al Saud
        "Q214013",   # Antonio Guterres
    ],
    "organisation": [
        "Q1065",    # Organisation des Nations unies
        "Q7184",    # Organisation du traité de l'Atlantique nord
        "Q458",     # Union européenne
        "Q7172",    # Ligue arabe
        "Q8236",    # Union africaine
        "Q7809",    # UNESCO
        "Q7821",    # UNICEF
        "Q7817",    # Organisation mondiale de la santé
        "Q7799",    # OPEC
        "Q11409",   # Banque mondiale
        "Q7807",    # Fonds monétaire international
        "Q7828",    # Cour pénale internationale
        "Q7825",    # Organisation mondiale du commerce
        "Q2429",    # Croix-Rouge
        "Q217365",  # Conseil de coopération du Golfe
        "Q484652",  # Organisation internationale
        "Q39203",   # Hamas
        "Q11205",   # Hezbollah
        "Q216357",  # Ennahdha
        "Q216345",  # Front de libération nationale (Algérie)
        "Q2955",    # Parti démocrate (États-Unis)
        "Q29468",   # Parti républicain (États-Unis)
        "Q20012759",# Renaissance (parti français)
        "Q423",     # Parti travailliste (Royaume-Uni)
        "Q131072",  # Likoud
    ],
    "lieu": [
        "Q85",      # Le Caire
        "Q3616",    # Téhéran
        "Q90",      # Paris
        "Q60",      # New York
        "Q3561",    # Alger
        "Q3766",    # Damas
        "Q1563",    # Beyrouth
        "Q3820",    # Beyrouth / Amman
        "Q3640",    # Bagdad
        "Q3787",    # Riyad
        "Q3711",    # Belgrade / Tunis
        "Q3551",    # Rabat
        "Q807",     # Lausanne / Lausanne
        "Q1218",    # Jérusalem
        "Q472",     # Sofia / Gaza
        "Q865",     # Taïwan / Ramallah
        "Q1492",    # Barcelone / Casablanca
        "Q613",     # Dubaï
        "Q1530",    # Bagdad / Abou Dhabi
        "Q3861",    # Doha
        "Q384",     # Koweït (ville)
        "Q3881",    # Manama
        "Q3824",    # Mascate
        "Q3616",    # Téhéran
        "Q3929",    # Sanaa
        "Q3616",    # Tripoli
        "Q1963",    # Khartoum
        "Q142",     # France
        "Q30",      # États-Unis
        "Q159",     # Russie
        "Q148",     # Chine
        "Q145",     # Royaume-Uni
        "Q183",     # Allemagne
        "Q43",      # Turquie
        "Q79",      # Égypte
        "Q262",     # Algérie
        "Q1028",    # Maroc
        "Q948",     # Tunisie
        "Q851",     # Arabie saoudite
        "Q878",     # Émirats arabes unis
        "Q810",     # Jordanie
        "Q822",     # Liban
        "Q796",     # Irak
        "Q858",     # Syrie
        "Q801",     # Israël
        "Q219060",  # État de Palestine
        "Q843",     # Pakistan
        "Q794",     # Iran
        "Q846",     # Qatar
        "Q817",     # Koweït
        "Q842",     # Oman
        "Q398",     # Bahreïn
        "Q805",     # Yémen
        "Q1016",    # Libye
        "Q1049",    # Soudan
    ]
}


def executer_sparql(requete, timeout=30):
    """Exécute une requête SPARQL sur Wikidata."""
    url = "https://query.wikidata.org/sparql"
    try:
        r = requests.get(
            url,
            params={"query": requete, "format": "json"},
            headers=HEADERS,
            timeout=timeout,
        )
        if r.status_code == 200:
            return r.json().get("results", {}).get("bindings", [])
        elif r.status_code == 429:
            print("  [SPARQL] Rate limit (429), pause 5s...")
            time.sleep(5)
    except Exception as e:
        print(f"  [SPARQL] Exception ({e})")
    return []


def decouvrir_qids():
    """Découvre des Q-IDs ciblés via SPARQL pour les 3 catégories."""
    qids_par_type = {"personne": set(), "organisation": set(), "lieu": set()}

    # Ajout des amorces
    for cat, items in AMORCES_PRIORITAIRES.items():
        qids_par_type[cat].update(items)

    print("  [DISCOVERY] Interrogation SPARQL pour les entités...")

    # 1. Personnes (Chefs d'État et ministres avec labels ar et fr)
    q_pers = """
    SELECT DISTINCT ?item WHERE {
      VALUES ?role { wd:Q11696 wd:Q14211 wd:Q14212 wd:Q51025 wd:Q48352 wd:Q30461 wd:Q2285706 wd:Q19546 }
      ?item wdt:P39 ?role.
    }
    LIMIT 150
    """
    for row in executer_sparql(q_pers):
        qid = row["item"]["value"].split("/")[-1]
        qids_par_type["personne"].add(qid)
    print(f"    Personnes découvertes : {len(qids_par_type['personne'])}")
    time.sleep(1)

    # 2. Organisations (Internationales et partis politiques)
    q_org = """
    SELECT DISTINCT ?item WHERE {
      VALUES ?type { wd:Q484652 wd:Q7278 wd:Q7188 wd:Q43229 wd:Q1334870 }
      ?item wdt:P31 ?type.
    }
    LIMIT 120
    """
    for row in executer_sparql(q_org):
        qid = row["item"]["value"].split("/")[-1]
        qids_par_type["organisation"].add(qid)
    print(f"    Organisations découvertes : {len(qids_par_type['organisation'])}")
    time.sleep(1)

    # 3. Lieux (Pays souverains, capitales et métropoles)
    q_lieux = """
    SELECT DISTINCT ?item WHERE {
      VALUES ?type { wd:Q6256 wd:Q5119 wd:Q1549591 }
      ?item wdt:P31 ?type.
    }
    LIMIT 180
    """
    for row in executer_sparql(q_lieux):
        qid = row["item"]["value"].split("/")[-1]
        qids_par_type["lieu"].add(qid)
    print(f"    Lieux découverts : {len(qids_par_type['lieu'])}")

    return qids_par_type


def recuperer_details_entites(qids_a_chercher):
    """Récupère les labels et alias ar/fr via l'API Wikidata par lots de 50."""
    details = {}
    qids_list = list(qids_a_chercher)
    taille_lot = 50

    url_api = "https://www.wikidata.org/w/api.php"

    for i in range(0, len(qids_list), taille_lot):
        lot = qids_list[i : i + taille_lot]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(lot),
            "props": "labels|aliases",
            "languages": "ar|fr",
            "format": "json",
        }
        try:
            r = requests.get(url_api, params=params, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                entities = r.json().get("entities", {})
                for qid, data in entities.items():
                    ar_lbl = data.get("labels", {}).get("ar", {}).get("value", "").strip()
                    fr_lbl = data.get("labels", {}).get("fr", {}).get("value", "").strip()
                    if ar_lbl and fr_lbl:
                        aliases_ar = [
                            a["value"].strip()
                            for a in data.get("aliases", {}).get("ar", [])
                            if a.get("value") and a["value"].strip() != ar_lbl
                        ]
                        # Dédupliquer les alias tout en préservant l'ordre
                        aliases_uniques = list(dict.fromkeys(aliases_ar))
                        details[qid] = {
                            "ar": ar_lbl,
                            "fr": fr_lbl,
                            "variantes": " | ".join(aliases_uniques),
                        }
        except Exception as e:
            print(f"  [API] Erreur lot {i}-{i+taille_lot} ({e})")
        time.sleep(0.3)

    return details


def calculer_sha256(chemin_fichier):
    h = hashlib.sha256()
    with open(chemin_fichier, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def extraire_noms(
    chemin_sortie="donnees/noms/noms_v1.csv",
    cible_min=300,
    cible_max=500,
    essai=False,
):
    """Extrait les noms propres, filtre et écrit le CSV final."""
    chemin_sortie = Path(chemin_sortie)
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

    qids_par_type = decouvrir_qids()
    tous_qids = set()
    for cat, qids in qids_par_type.items():
        tous_qids.update(qids)

    if essai:
        tous_qids = set(list(tous_qids)[:20])

    print(f"  [EXTRACTION] Récupération des libellés pour {len(tous_qids)} Q-IDs...")
    details_entites = recuperer_details_entites(tous_qids)

    # Construction de la liste dédupliquée
    entrees = []
    vus_paires = set()
    vus_qids = set()

    for cat in ["personne", "organisation", "lieu"]:
        for qid in qids_par_type[cat]:
            if qid in details_entites and qid not in vus_qids:
                d = details_entites[qid]
                paire = (d["ar"], d["fr"])
                if paire in vus_paires:
                    continue
                vus_paires.add(paire)
                vus_qids.add(qid)
                entrees.append(
                    {
                        "forme_arabe": d["ar"],
                        "variantes": d["variantes"],
                        "forme_francaise": d["fr"],
                        "type": cat,
                        "id_wikidata": qid,
                    }
                )

    total_obtenu = len(entrees)
    print(f"  [RESULTAT] {total_obtenu} entités valides obtenues.")

    if not essai:
        # Ajustement si besoin pour être strictement dans [cible_min, cible_max]
        if total_obtenu > cible_max:
            entrees = entrees[:cible_max]
            total_obtenu = len(entrees)
        elif total_obtenu < cible_min:
            print(
                f"  [ATTENTION] {total_obtenu} entités < {cible_min} cible minimum."
            )

    # Écriture dans noms_v1.csv
    with open(chemin_sortie, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "forme_arabe",
                "variantes",
                "forme_francaise",
                "type",
                "id_wikidata",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(entrees)

    sha = calculer_sha256(chemin_sortie)
    print(f"  [ECRITURE] {total_obtenu} lignes écrites dans {chemin_sortie} (sha256: {sha[:12]}...)")

    return entrees, sha


def creer_modele_manuel(chemin_modele="donnees/noms/noms_manuel_modele.csv"):
    """Crée le fichier modèle d'accueil des 100 noms à annoter à la main."""
    chemin = Path(chemin_modele)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            ["forme_arabe", "variantes", "forme_francaise", "type", "id_wikidata"]
        )
        writer.writerow(
            [
                "مثال اسم",
                "مثال 1 | مثال 2",
                "Exemple Nom",
                "personne",
                "manuel-001",
            ]
        )
    print(f"[MODELE] Modèle d'accueil créé dans {chemin}")


def creer_lisezmoi(total_entrees, sha_csv, sha_seuils, chemin_lisezmoi="donnees/noms/lisezmoi.md"):
    """Crée donnees/noms/lisezmoi.md avec le SHA256 de seuils.yaml."""
    chemin = Path(chemin_lisezmoi)
    contenu = f"""<!-- seuils.yaml sha256: {sha_seuils} -->
# Noms propres arabe-français (Jeu Noms)

Ce dossier contient la table de référence des noms propres (personnes, lieux, organisations)
utilisée pour le prompt de traduction, le filtre 5 de D4, et la chaîne de sous-titrage en direct.

## Fichiers

- `noms_v1.csv` : table issue de Wikidata ({total_entrees} entrées, sha256: `{sha_csv}`)
- `noms_manuel_modele.csv` : modèle au format CSV prêt pour accueillir les 100 noms relevés à la main dans les émissions de 2026.
- `lisezmoi.md` : ce document.

## Format des colonnes

1. `forme_arabe` : forme principale en graphie arabe standard.
2. `variantes` : variantes orthographiques arabes (alias Wikidata), séparées par ` | `.
3. `forme_francaise` : translittération ou désignation usuelle en français.
4. `type` : catégorie (`personne`, `organisation`, `lieu`).
5. `id_wikidata` : identifiant Wikidata (`Q...`) ou identifiant manuel (`manuel-...`).
"""
    with open(chemin, "w", encoding="utf-8") as f:
        f.write(contenu)
    print(f"[LISEZMOI] Créé dans {chemin}")


def main():
    parser = argparse.ArgumentParser(
        description="Extraire les noms propres depuis Wikidata pour le Jeu Noms."
    )
    parser.add_argument(
        "--sortie",
        default="donnees/noms/noms_v1.csv",
        help="Chemin du fichier de sortie CSV (défaut : donnees/noms/noms_v1.csv).",
    )
    parser.add_argument(
        "--essai",
        action="store_true",
        help="Mode essai : extrait seulement un petit échantillon de test.",
    )
    args = parser.parse_args()

    seuils = charger_seuils()
    noms_min = seuils["constantes"]["noms_min"]
    noms_max = seuils["constantes"]["noms_max"]

    sha_seuils = calculer_sha256("code/donnees/seuils.yaml")

    print(f"=== Extraction des noms propres Wikidata (Cible : [{noms_min}, {noms_max}]) ===")
    entrees, sha_csv = extraire_noms(
        chemin_sortie=args.sortie,
        cible_min=noms_min,
        cible_max=noms_max,
        essai=args.essai,
    )

    creer_modele_manuel()
    creer_lisezmoi(len(entrees), sha_csv, sha_seuils)

    if not args.essai:
        # Journal
        maintenant = time.strftime("%Y-%m-%d %H:%M")
        entree_j = (
            f"\n## {maintenant}  Extraction Noms propres Wikidata (noms_v1.csv)\n"
            f"Commande : python code/donnees/extraire_noms_wikidata.py\n"
            f"Entrée   : Wikidata (SPARQL et API wbgetentities)\n"
            f"Sortie   : {args.sortie}\n"
            f"Résultat : {len(entrees)} entrées uniques, sha256: {sha_csv}\n"
            f"Remarque : Répartition personnes, organisations, lieux avec variantes arabes (alias)\n"
        )
        with open("journal-donnees.md", "a", encoding="utf-8") as f:
            f.write(entree_j)

        # Manifeste
        date_jour = time.strftime("%Y-%m-%d")
        taille = Path(args.sortie).stat().st_size
        ligne_m = (
            f"{args.sortie},Noms,v1,Wikidata (SPARQL / API),https://www.wikidata.org,"
            f"{date_jour},CC0 (Wikidata),{len(entrees)},{taille},{sha_csv},"
            f"extraire_noms_wikidata.py,master\n"
        )
        with open("donnees/manifeste.csv", "a", encoding="utf-8") as f:
            f.write(ligne_m)


if __name__ == "__main__":
    main()
