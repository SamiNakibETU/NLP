"""Validation des fiches des jeux T/R et D.

Une fiche est refusee des qu'elle est incomplete : champ requis absent, vide,
de mauvais type, hors des valeurs admises, ou incoherent (debut apres fin,
date d'avant le 1er octobre 2026 pour T/R...). Les avertissements signalent
des faits qui ne rendent pas la fiche invalide (duree hors fenetre 20-30 s,
ecart d'ecoute au-dessus du seuil D1...) : les scripts de lot suivants en
decident.

Renvoie des listes de (code, message) avec des codes stables, pour pouvoir
les retrouver dans les rapports et les rejets.csv.
"""

import json
import math
import re
from datetime import date

from schemas import SCHEMAS

# Champs dont la presence suffit a accepter une fiche ; a None ils sont
# simplement ignores (ex. fr_2, present sur 15 extraits seulement).
_ABSENT = object()

_MOTIF_ID_TR = re.compile(SCHEMAS["tr"]["id"]["motif"])
_MOTIF_ID_D = re.compile(SCHEMAS["d"]["id"]["motif"])
_MOTIF_ID_A = re.compile(SCHEMAS["a"]["id"]["motif"])
_MOTIF_ID_P = re.compile(SCHEMAS["p"]["id"]["motif"])


def valider_fiche(fiche, type_jeu, seuils):

    """Valide une fiche (dict) contre son schema.

    type_jeu : "tr" ou "d". seuils : dict renvoye par seuils.charger_seuils().
    Renvoie (erreurs, avertissements), listes de (code, message).
    """
    if type_jeu not in SCHEMAS:
        raise ValueError(f"type de jeu inconnu : {type_jeu}")
    schema = SCHEMAS[type_jeu]
    erreurs = []
    avertissements = []

    if not isinstance(fiche, dict):
        return [("fiche_invalide", "la fiche n'est pas un objet JSON (dict attendu)")], []

    for cle in fiche:
        if cle not in schema:
            avertissements.append(("champ_inconnu", f"champ non prevu par le schema : {cle}"))

    for nom, spec in schema.items():
        valeur = fiche.get(nom, _ABSENT)
        if valeur is _ABSENT or valeur is None:
            if spec["requis"]:
                erreurs.append(("champ_manquant", f"champ requis absent ou vide : {nom}"))
            continue
        _verifier_champ(nom, spec, valeur, seuils, type_jeu, erreurs)

    _verifier_coherences(fiche, type_jeu, seuils, erreurs, avertissements)
    return erreurs, avertissements


def _verifier_champ(nom, spec, valeur, seuils, type_jeu, erreurs):
    genre = spec["type"]
    if genre in ("texte", "date"):
        if not isinstance(valeur, str):
            erreurs.append(("type_invalide", f"{nom} : texte attendu, recu {type(valeur).__name__}"))
            return
        if not valeur.strip():
            erreurs.append(("champ_vide", f"{nom} : texte vide"))
            return
        motif = spec.get("motif")
        if motif and not re.fullmatch(motif, valeur):
            erreurs.append(("format_invalide", f"{nom} : '{valeur}' ne suit pas le format attendu ({motif})"))
        if genre == "date":
            _verifier_date(nom, spec, valeur, seuils, type_jeu, erreurs)
    elif genre == "enum":
        if not isinstance(valeur, str) or valeur not in spec["enum"]:
            erreurs.append(
                ("enum_invalide", f"{nom} : '{valeur}' hors des valeurs admises ({', '.join(spec['enum'])})")
            )
    elif genre == "entier":
        if isinstance(valeur, bool) or not isinstance(valeur, int):
            erreurs.append(("type_invalide", f"{nom} : entier attendu, recu {valeur!r}"))
            return
        minimum = spec.get("min")
        if minimum is not None and valeur < minimum:
            erreurs.append(("valeur_invalide", f"{nom} : {valeur} est en dessous du minimum {minimum}"))
        maximum = spec.get("max")
        if maximum is not None and valeur > maximum:
            erreurs.append(("valeur_invalide", f"{nom} : {valeur} est au-dessus du maximum {maximum}"))
    elif genre == "nombre":
        if isinstance(valeur, bool) or not isinstance(valeur, (int, float)):
            erreurs.append(("type_invalide", f"{nom} : nombre attendu, recu {valeur!r}"))
            return
        if isinstance(valeur, float) and not math.isfinite(valeur):
            erreurs.append(("valeur_invalide", f"{nom} : nombre fini attendu, recu {valeur!r}"))
            return
        minimum = spec.get("min")
        if minimum is not None and valeur < minimum:
            erreurs.append(("valeur_invalide", f"{nom} : {valeur} est en dessous du minimum {minimum}"))
        maximum = spec.get("max")
        if maximum is not None and valeur > maximum:
            erreurs.append(("valeur_invalide", f"{nom} : {valeur} est au-dessus du maximum {maximum}"))
    elif genre == "url":
        if not isinstance(valeur, str) or not (valeur.startswith("http://") or valeur.startswith("https://")):
            erreurs.append(("url_invalide", f"{nom} : adresse http(s) attendue, recu {valeur!r}"))
    elif genre == "liste":
        if not isinstance(valeur, list) or not all(isinstance(x, str) for x in valeur):
            erreurs.append(("liste_invalide", f"{nom} : liste de chaines attendue, recu {valeur!r}"))
    elif genre == "dict":
        if not isinstance(valeur, dict):
            erreurs.append(("dict_invalide", f"{nom} : dictionnaire attendu, recu {type(valeur).__name__}"))
    else:  # pragma: no cover - le schema ne doit jamais en arriver la
        erreurs.append(("schema_invalide", f"{nom} : type de champ inconnu {genre!r}"))


def _verifier_date(nom, spec, valeur, seuils, type_jeu, erreurs):
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", valeur):
        erreurs.append(("format_date", f"{nom} : date attendue au format AAAA-MM-JJ, recu {valeur!r}"))
        return
    try:
        date_valeur = date.fromisoformat(valeur)
    except ValueError:
        erreurs.append(("date_invalide", f"{nom} : date impossible {valeur!r}"))
        return
    cle_min = spec.get("apres")
    if cle_min and type_jeu == "tr":
        date_min = date.fromisoformat(seuils["constantes"][cle_min])
        if date_valeur < date_min:
            erreurs.append(
                ("date_avant_minimum",
                 f"{nom} : {valeur} est d'avant la date minimum du plan ({date_min.isoformat()}, "
                 "emissions d'apres le 1er octobre 2026)")
            )


def _verifier_coherences(fiche, type_jeu, seuils, erreurs, avertissements):
    debut = fiche.get("debut")
    fin = fiche.get("fin")
    if (isinstance(debut, (int, float)) and not isinstance(debut, bool) and math.isfinite(debut)
            and isinstance(fin, (int, float)) and not isinstance(fin, bool) and math.isfinite(fin)):
        if fin <= debut:
            erreurs.append(("coherence_debut_fin", f"fin ({fin}) doit etre apres debut ({debut})"))
        elif type_jeu == "tr":
            duree = fin - debut
            cmin = seuils["constantes"]["duree_extrait_min_s"]
            cmax = seuils["constantes"]["duree_extrait_max_s"]
            if not (cmin <= duree <= cmax):
                avertissements.append(
                    ("duree_hors_fenetre",
                     f"duree {duree:.0f} s hors de la fenetre {cmin}-{cmax} s "
                     "(le verifier_jeux du lot 4 en fait une erreur)")
                )
        elif type_jeu in ("d", "a", "d_selection"):
            duree = fin - debut
            cmin = seuils["constantes"]["duree_segment_min_s"]
            cmax = seuils["constantes"]["duree_segment_max_s"]
            if not (cmin <= duree <= cmax):
                avertissements.append(
                    ("duree_hors_fenetre",
                     f"duree {duree:.1f} s hors de la fenetre {cmin}-{cmax} s")
                )

    if type_jeu == "tr":
        jeu = fiche.get("jeu")
        identifiant = fiche.get("id")
        if isinstance(jeu, str) and isinstance(identifiant, str) and identifiant:
            prefixe_attendu = jeu.lower()
            if not identifiant.startswith(prefixe_attendu + "-"):
                erreurs.append(
                    ("id_jeu_incoherent", f"id '{identifiant}' incoherent avec jeu '{jeu}'")
                )

    if type_jeu in ("d", "a", "d_selection"):
        ecart = fiche.get("ecart_ecoute")
        if isinstance(ecart, (int, float)) and not isinstance(ecart, bool) and math.isfinite(ecart):
            seuil_pct = seuils["proposes"]["ecart_ecoute_max_pct"]
            if ecart * 100 >= seuil_pct:
                avertissements.append(
                    ("ecart_ecoute_haut",
                     f"ecart_ecoute {ecart} >= seuil D1 de {seuil_pct} % : le filtre D1 aurait du "
                     "ecarter ce segment")
                )
        audio = fiche.get("audio")
        if isinstance(audio, str) and audio and not audio.lower().endswith(".flac"):
            avertissements.append(("audio_pas_flac", f"audio '{audio}' ne finit pas par .flac"))



def valider_fichier_jsonl(chemin, type_jeu, seuils):
    """Valide chaque ligne d'un fichier JSONL de fiches.

    Renvoie une liste de resultats, un par ligne :
    {"ligne": n, "id": str ou None, "valide": bool, "erreurs": [...], "avertissements": [...]}
    """
    resultats = []
    with open(chemin, "r", encoding="utf-8") as lecteur:
        for numero, ligne in enumerate(lecteur, start=1):
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                fiche = json.loads(ligne)
            except json.JSONDecodeError as exc:
                resultats.append({
                    "ligne": numero, "id": None, "valide": False,
                    "erreurs": [("json_invalide", f"ligne illisible en JSON ({exc})")],
                    "avertissements": [],
                })
                continue
            erreurs, avertissements = valider_fiche(fiche, type_jeu, seuils)
            identifiant = fiche.get("id") if isinstance(fiche, dict) else None
            resultats.append({
                "ligne": numero, "id": identifiant, "valide": not erreurs,
                "erreurs": erreurs, "avertissements": avertissements,
            })
    _signaler_ids_doublons(resultats)
    return resultats


def _signaler_ids_doublons(resultats):
    """Ajoute une erreur id_doublon a chaque fiche repetee dans le fichier."""
    vus = {}
    for resultat in resultats:
        identifiant = resultat["id"]
        if identifiant is None:
            continue
        if identifiant in vus:
            resultat["erreurs"].append(
                ("id_doublon", f"id '{identifiant}' deja vu ligne {vus[identifiant]}")
            )
            resultat["valide"] = False
        else:
            vus[identifiant] = resultat["ligne"]

