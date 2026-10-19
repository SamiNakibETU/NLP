"""Schémas des fiches des jeux T/R (extraits de test) et D (segments distillés).

Les champs, leurs valeurs possibles et leurs contraintes viennent de
Schémas de données pour les extraits d'évaluation (T/R)
et "Ce qu'on enregistre pour chaque segment" (jeu D).

Chaque spec de champ porte :
  requis      le champ doit etre present et non vide/null
  type        texte | enum | entier | nombre | date | url | liste
  enum        (type enum uniquement) valeurs admises
  motif       (optionnel) expression reguliere que la valeur doit couvrir entierement
  description reprise du plan

Une fiche est un objet JSON ; les fichiers de fiches sont des JSONL
(une fiche par ligne) : t.jsonl, r.jsonl pour T/R, d_v1.jsonl pour D.
"""

import re

# Types de fiches acceptes par le validateur.
TYPES_JEUX = ("tr", "d", "a", "p", "d_selection")


SCHEMA_TR = {
    "id": {
        "requis": True,
        "type": "texte",
        "motif": r"[tr]-\d{3,}",
        "description": "identifiant, ex. t-001 (T) ou r-001 (R)",
    },
    "jeu": {
        "requis": True,
        "type": "enum",
        "enum": ["T", "R"],
        "description": "T = test maison, R = reglage",
    },
    "chaine": {
        "requis": True,
        "type": "texte",
        "description": "nom de la chaine (au moins quatre chaines differentes)",
    },
    "emission": {
        "requis": True,
        "type": "texte",
        "description": "titre de l'emission",
    },
    "date": {
        "requis": True,
        "type": "date",
        "apres": "date_min_extrait",
        "description": "date de diffusion ; emissions d'apres le 1er octobre 2026",
    },
    "url": {
        "requis": True,
        "type": "url",
        "description": "adresse de la video (l'audio reste en local)",
    },
    "debut": {
        "requis": True,
        "type": "nombre",
        "min": 0,
        "description": "debut de l'extrait, en secondes",
    },
    "fin": {
        "requis": True,
        "type": "nombre",
        "min": 0,
        "description": "fin de l'extrait, en secondes (doit etre apres debut)",
    },
    "type": {
        "requis": True,
        "type": "enum",
        "enum": ["plateau", "reportage", "debat"],
        "description": "trois types a parts egales",
    },
    "dialecte": {
        "requis": True,
        "type": "enum",
        "enum": ["aucun", "un peu", "dominant"],
        "description": "presence de dialecte dans l'extrait",
    },
    "locuteurs": {
        "requis": True,
        "type": "entier",
        "min": 1,
        "description": "nombre de voix",
    },
    "ar": {
        "requis": True,
        "type": "texte",
        "description": "transcription arabe corrigee (hesitations comprises)",
    },
    "fr": {
        "requis": True,
        "type": "texte",
        "description": "traduction de reference",
    },
    "fr_2": {
        "requis": False,
        "type": "texte",
        "description": "deuxieme traduction independante (15 extraits seulement)",
    },
    "annotateurs": {
        "requis": True,
        "type": "texte",
        "description": "initiales (transcription, traduction, relecture)",
    },
}

SCHEMA_D = {
    "id": {
        "requis": True,
        "type": "texte",
        "motif": r"d-\d{6,}",
        "description": "identifiant, ex. d-000001",
    },
    "episode": {
        "requis": True,
        "type": "texte",
        "description": "identifiant MGB-2 de l'episode",
    },
    "debut": {
        "requis": True,
        "type": "nombre",
        "min": 0,
        "description": "debut du segment, en secondes dans l'episode",
    },
    "fin": {
        "requis": True,
        "type": "nombre",
        "min": 0,
        "description": "fin du segment, en secondes dans l'episode (doit etre apres debut)",
    },
    "audio": {
        "requis": True,
        "type": "texte",
        "description": "chemin du fichier audio (16 kHz, mono, FLAC)",
    },
    "ar_manuel": {
        "requis": True,
        "type": "texte",
        "description": "transcription MGB-2 d'origine",
    },
    "ar_ecoute": {
        "requis": True,
        "type": "texte",
        "description": "sortie de notre modele d'ecoute (version avec erreurs)",
    },
    "ecart_ecoute": {
        "requis": True,
        "type": "nombre",
        "min": 0,
        "max": 5.0,
        "description": "fraction entre 0 et 1 (valeur > 5 refusee) : ecart en mots entre transcription manuelle et sortie d'ecoute",
    },
    "fr": {
        "requis": True,
        "type": "texte",
        "description": "traduction du gros modele",
    },
    "modele_fr": {
        "requis": True,
        "type": "texte",
        "description": "nom et version du modele qui a traduit",
    },
    "score_qe": {
        "requis": True,
        "type": "nombre",
        "description": "score COMETKiwi (peut etre negatif)",
    },
    "signalements": {
        "requis": True,
        "type": "liste",
        "description": "liste des signalements (chiffres, noms) ; liste vide possible",
    },
    "partie": {
        "requis": True,
        "type": "enum",
        "enum": ["entrainement", "validation"],
        "description": "decoupage par episode, jamais les deux pour un meme episode",
    },
}

SCHEMA_A = {
    "id": {
        "requis": True,
        "type": "texte",
        "motif": r"a-\d{6,}",
        "description": "identifiant, ex. a-000001",
    },
    "episode": {
        "requis": True,
        "type": "texte",
        "description": "identifiant MGB-2 de l'episode",
    },
    "debut": {
        "requis": True,
        "type": "nombre",
        "min": 0,
        "description": "debut du segment, en secondes dans l'episode",
    },
    "fin": {
        "requis": True,
        "type": "nombre",
        "min": 0,
        "description": "fin du segment, en secondes dans l'episode (doit etre apres debut)",
    },
    "audio": {
        "requis": True,
        "type": "texte",
        "description": "chemin du fichier audio (16 kHz, mono, FLAC)",
    },
    "ar_manuel": {
        "requis": True,
        "type": "texte",
        "description": "transcription MGB-2 d'origine",
    },
    "ar_ecoute": {
        "requis": True,
        "type": "texte",
        "description": "sortie de notre modele d'ecoute (version avec erreurs)",
    },
    "ecart_ecoute": {
        "requis": True,
        "type": "nombre",
        "min": 0,
        "max": 5.0,
        "description": "fraction entre 0 et 1 (valeur > 5 refusee) : ecart en mots entre transcription manuelle et sortie d'ecoute",
    },
    "partie": {
        "requis": True,
        "type": "enum",
        "enum": ["entrainement", "validation"],
        "description": "decoupage par episode, jamais les deux pour un meme episode",
    },
    "id_source_mgb2": {
        "requis": False,
        "type": "texte",
        "description": "identifiant source du segment dans le corpus MGB-2",
    },
    "modele_asr": {
        "requis": False,
        "type": "texte",
        "description": "nom du modele ASR Whisper utilise",
    },
    "revision_asr": {
        "requis": False,
        "type": "texte",
        "description": "revision ou commit du modele ASR",
    },
    "parametres": {
        "requis": False,
        "type": "dict",
        "description": "parametres essentiels d'execution et seuils D1",
    },
    "date_execution": {
        "requis": False,
        "type": "texte",
        "description": "date et heure ISO de l'execution",
    },
}

SCHEMA_P = {
    "id": {
        "requis": True,
        "type": "texte",
        "motif": r"p-\d{6,}",
        "description": "identifiant, ex. p-000001",
    },
    "source": {
        "requis": True,
        "type": "texte",
        "description": "corpus OPUS d'origine (News-Commentary, GlobalVoices, TED2020, UNPC)",
    },
    "id_origine": {
        "requis": True,
        "type": "entier",
        "min": 1,
        "description": "numero de ligne dans le fichier source brut",
    },
    "ar": {
        "requis": True,
        "type": "texte",
        "description": "texte arabe d'origine",
    },
    "fr": {
        "requis": True,
        "type": "texte",
        "description": "traduction francaise filtree",
    },
}

SCHEMA_D_SELECTION = {
    "id": {
        "requis": True,
        "type": "texte",
        "motif": r"d-\d{6,}",
        "description": "identifiant interne stable, ex. d-000001",
    },
    "id_source_mgb2": {
        "requis": True,
        "type": "texte",
        "description": "identifiant source complet MGB-2 (format Kaldi)",
    },
    "episode": {
        "requis": True,
        "type": "texte",
        "description": "identifiant MGB-2 de l'episode",
    },
    "debut": {
        "requis": True,
        "type": "nombre",
        "min": 0,
        "description": "debut du segment, en secondes dans l'episode",
    },
    "fin": {
        "requis": True,
        "type": "nombre",
        "min": 0,
        "description": "fin du segment, en secondes dans l'episode",
    },
    "audio": {
        "requis": True,
        "type": "texte",
        "description": "chemin du fichier audio FLAC (16 kHz, mono, nom nettoye)",
    },
    "ar_manuel": {
        "requis": True,
        "type": "texte",
        "description": "transcription MGB-2 d'origine",
    },
    "ar_ecoute": {
        "requis": True,
        "type": "texte",
        "description": "sortie du modele d'ecoute ASR",
    },
    "ecart_ecoute": {
        "requis": True,
        "type": "nombre",
        "min": 0,
        "max": 5.0,
        "description": "taux d'ecart mesuré entre transcription manuelle et ASR",
    },
    "partie": {
        "requis": True,
        "type": "enum",
        "enum": ["entrainement", "validation"],
        "description": "decoupage par episode (95% entrainement, 5% validation)",
    },
    "modele_asr": {
        "requis": True,
        "type": "texte",
        "description": "nom du modele ASR Whisper utilise",
    },
    "revision_asr": {
        "requis": True,
        "type": "texte",
        "description": "revision ou commit du modele ASR",
    },
    "parametres": {
        "requis": True,
        "type": "dict",
        "description": "parametres essentiels d'execution et seuils D1",
    },
    "date_execution": {
        "requis": True,
        "type": "texte",
        "description": "date et heure ISO de l'execution",
    },
}

SCHEMAS = {
    "tr": SCHEMA_TR,
    "d": SCHEMA_D,
    "a": SCHEMA_A,
    "p": SCHEMA_P,
    "d_selection": SCHEMA_D_SELECTION,
}


_MOTIF_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def fiche_vide(type_jeu):
    """Renvoie un modele de fiche vide (toutes les cles a None).

    Sert de base aux modeles d'annotation du lot 4 : on copie ce dict, on
    remplit les champs, puis le validateur refuse la fiche tant qu'elle est
    incomplete. Un champ optionnel y reste a None.
    """
    if type_jeu not in SCHEMAS:
        raise ValueError(f"type de jeu inconnu : {type_jeu} (attendu : {', '.join(TYPES_JEUX)})")
    return {nom: None for nom in SCHEMAS[type_jeu]}
