"""Chargement du fichier de configuration des seuils (seuils.yaml).

Le fichier seuils.yaml, pose a cote de ce module, est la seule source des
valeurs chiffrees du plan de collecte . Les scripts
suivants passent tous par charger_seuils() ; ils ne codent aucun seuil en dur.
"""

from pathlib import Path

import yaml

DOSSIER = Path(__file__).resolve().parent
CHEMIN_DEFAUT = DOSSIER / "seuils.yaml"

# Chaque cle attendue, avec le(s) type(s) accepte(s). Un fichier seuils.yaml
# incomplet ou mal type doit etre signale tout de suite, pas plus tard dans
# un filtre de donnees.
_CLES_PROPPOSES = {
    "ecart_ecoute_max_pct": (int, float),
    "rapport_longueur_min": (int, float),
    "rapport_longueur_max": (int, float),
    "qe_rejet_pire_pct": (int, float),
    "taux_erreur_sens_max_pct": (int, float),
}

_CLES_CONSTANTES = {
    "duree_segment_min_s": (int, float),
    "duree_segment_max_s": (int, float),
    "mots_min_segment": int,
    "segments_max_par_episode": int,
    "segments_objectif": int,
    "segments_comparaison_modeles": int,
    "segments_mesure_vitesse": int,
    "taille_paquet_traduction": int,
    "relances_paquet_max": int,
    "extraits_t": int,
    "extraits_r": int,
    "duree_extrait_min_s": (int, float),
    "duree_extrait_max_s": (int, float),
    "date_min_extrait": str,
    "extraits_avec_fr2": int,
    "unpc_max_phrases": int,
    "noms_min": int,
    "noms_max": int,
    "segments_controle_humain": int,
    "segments_debuts_phrase": int,
    "coupe_debuts_pct": list,
    "debuts_relecture": int,
    "episodes_validation_pct": (int, float),
    "segments_max_par_episode_a": int,
    "heures_alignement_min": (int, float),
    "heures_alignement_max": (int, float),
}



class ErreurSeuils(Exception):
    """Le fichier seuils.yaml est absent, illisible ou incomplet."""


def charger_seuils(chemin=CHEMIN_DEFAUT):
    """Lit le fichier YAML et verifie que toutes les cles attendues existent.

    Renvoie un dict {"proposes": {...}, "constantes": {...}}.
    Leve ErreurSeuils si une cle manque ou a le mauvais type.
    """
    chemin = Path(chemin)
    if not chemin.is_file():
        raise ErreurSeuils(f"fichier de seuils introuvable : {chemin}")
    try:
        contenu = yaml.safe_load(chemin.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ErreurSeuils(f"fichier de seuils illisible : {chemin} ({exc})") from exc

    if not isinstance(contenu, dict) or "proposes" not in contenu or "constantes" not in contenu:
        raise ErreurSeuils(
            f"fichier de seuils inattendu : {chemin} doit avoir les sections proposes et constantes"
        )

    for section, cles in (("proposes", _CLES_PROPPOSES), ("constantes", _CLES_CONSTANTES)):
        bloc = contenu[section]
        if not isinstance(bloc, dict):
            raise ErreurSeuils(f"section '{section}' : attendu un dictionnaire")
        for cle, types in cles.items():
            if cle not in bloc:
                raise ErreurSeuils(f"section '{section}' : cle manquante '{cle}'")
            if isinstance(bloc[cle], bool) or not isinstance(bloc[cle], types):
                raise ErreurSeuils(
                    f"section '{section}' : cle '{cle}' doit etre de type "
                    f"{', '.join(t.__name__ for t in (types if isinstance(types, tuple) else (types,)))}"
                )
    return {"proposes": contenu["proposes"], "constantes": contenu["constantes"]}
