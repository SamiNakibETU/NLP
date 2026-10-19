"""Normalisation du texte arabe pour les mesures (WER, ecart d'ecoute, comptage).

Attention : ces fonctions ne servent qu'au moment de MESURER. Elles ne
remplacent jamais le texte stocke dans les fiches (transcriptions,
traductions), qui garde son orthographe d'origine.

Deux niveaux :
  "complet"  (defaut, pour les mesures) : apres decomposition NFKD, retire
             diacritiques et marques de hamza, puis unifie alef wasla -> alef,
             ta marbuta -> ha, alef maqsura -> ya. Les graphies equivalentes
             ("أحمد" / "احمد", "مدرسة" / "مدرسه") donnent le meme resultat.
  "leger"    pour comparer deux orthographes sans fusionner les hamzas :
             retire diacritiques, tatweel, marques bidirectionnelles, puis
             re-compose (NFC) pour que les lettres precomposees restent
             distinctes ("أ" != "ا").

Dans les deux niveaux : tatweel retire, chiffres arabes orientaux convertis
en chiffres latins, ponctuation arabe remplacee par son equivalent latin,
espaces reduits a un seul.
"""

import re
import unicodedata

# Marques invisibles de direction et de formatage (LRM, RLM, pop directionnel...)
_MARKS_BIDI = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\u00ad]")

# Diacritiques : signes coraniques (0610-061A), harakat et tanwin (064B-0652),
# superscript alef (0670), annotations coraniques (06D6-06ED), petites marques
# hautes et basses (06DF-06E8, 06EA-06ED). Au niveau complet on ajoute
# 0653-0655 (madda, marques de hamza) : apres NFKD leur retrait unifie
# أ إ آ ؤ ئ vers ا و ي.
_DIACRITIQUES_LEGER = re.compile(r"[\u0610-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")
_DIACRITIQUES_COMPLET = re.compile(r"[\u0610-\u061A\u064B-\u0655\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"

# Lettres qui ne se decomposent pas en NFKD et doivent etre unifiees a la main.
_UNIFICATIONS_COMPLET = str.maketrans({
    "\u0671": "\u0627",  # alef wasla ٱ -> alef
    "\u0629": "\u0647",  # ta marbuta ة -> ha
    "\u0649": "\u064A",  # alef maqsura ى -> ya
})

_CHIFFRES = str.maketrans(
    "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
    "\u06F0\u06F1\u06F2\u06F3\u06F4\u06F5\u06F6\u06F7\u06F8\u06F9",
    "01234567890123456789",
)

# Table de translittération officielle Buckwalter (Tim Buckwalter / QAMUS / LDC / Kaldi MGB-2)
TABLE_BUCKWALTER_VERS_ARABE = {
    "'": "\u0621", "|": "\u0622", ">": "\u0623", "&": "\u0624", "<": "\u0625", "}": "\u0626",
    "A": "\u0627", "b": "\u0628", "p": "\u0629", "t": "\u062A", "v": "\u062B", "j": "\u062C",
    "H": "\u062D", "x": "\u062E", "d": "\u062F", "*": "\u0630", "r": "\u0631", "z": "\u0632",
    "s": "\u0633", "$": "\u0634", "S": "\u0635", "D": "\u0636", "T": "\u0637", "Z": "\u0638",
    "E": "\u0639", "g": "\u063A", "_": "\u0640", "f": "\u0641", "q": "\u0642", "k": "\u0643",
    "l": "\u0644", "m": "\u0645", "n": "\u0646", "h": "\u0647", "w": "\u0648", "Y": "\u0649",
    "y": "\u064A", "F": "\u064B", "N": "\u064C", "K": "\u064D", "a": "\u064E", "u": "\u064F",
    "i": "\u0650", "~": "\u0651", "o": "\u0652", "`": "\u0670", "{": "\u0671",
}


def bw_vers_arabe(texte: str) -> str:
    """Convertit une chaîne en translittération Buckwalter vers l'arabe Unicode standard."""
    if not isinstance(texte, str):
        raise TypeError("texte attendu en chaîne de caractères")
    return "".join(TABLE_BUCKWALTER_VERS_ARABE.get(c, c) for c in texte)


_NIVEAUX = ("leger", "complet")


def _retirer_ponctuation(texte):
    """Remplace la ponctuation (arabe et latine, guillemets, tirets) par des espaces."""
    res = []
    for ch in texte:
        if unicodedata.category(ch).startswith("P") or ch in r"!#$%&'()*+,-./:;<=>?@[\]^_`{|}~":
            res.append(" ")
        else:
            res.append(ch)
    return "".join(res)


_MOTIF_ARABE = re.compile(r"[\u0600-\u06FF]")


def normaliser_arabe(texte, niveau="complet"):
    """Renvoie le texte arabe normalise pour les mesures. Voir l'en-tete du module."""
    if niveau not in _NIVEAUX:
        raise ValueError(f"niveau inconnu {niveau!r} (attendu : {', '.join(_NIVEAUX)})")
    if not isinstance(texte, str):
        raise TypeError("texte attendu en chaine de caracteres")
    texte = texte.replace("\u00a0", " ")
    texte = _MARKS_BIDI.sub("", texte)
    if niveau == "complet":
        texte = unicodedata.normalize("NFKD", texte)
        texte = _DIACRITIQUES_COMPLET.sub("", texte)
        texte = texte.replace(_TATWEEL, "")
        texte = texte.translate(_UNIFICATIONS_COMPLET)
    else:
        texte = unicodedata.normalize("NFC", texte)
        texte = _DIACRITIQUES_LEGER.sub("", texte)
        texte = texte.replace(_TATWEEL, "")
    texte = texte.translate(_CHIFFRES)
    texte = _retirer_ponctuation(texte)
    return re.sub(r"\s+", " ", texte).strip()



def mots_arabe(texte, niveau="complet"):
    """Decoupe le texte normalise en mots (list[str])."""
    return [mot for mot in normaliser_arabe(texte, niveau).split(" ") if mot]


def compter_mots_arabe(texte, niveau="complet"):
    """Nombre de mots normalises ; sert au filtre D1 'au moins 6 mots'."""
    return len(mots_arabe(texte, niveau))


def distance_mots(reference, candidat, niveau="complet"):
    """Distance de Levenshtein entre deux textes, compte en mots normalises."""
    a = mots_arabe(reference, niveau)
    b = mots_arabe(candidat, niveau)
    if not a:
        return len(b)
    if not b:
        return len(a)
    ligne_precedente = list(range(len(b) + 1))
    for i, mot_a in enumerate(a, start=1):
        ligne = [i]
        for j, mot_b in enumerate(b, start=1):
            cout = 0 if mot_a == mot_b else 1
            ligne.append(min(
                ligne_precedente[j] + 1,      # suppression
                ligne[j - 1] + 1,             # insertion
                ligne_precedente[j - 1] + cout,  # substitution
            ))
        ligne_precedente = ligne
    return ligne_precedente[-1]


def taux_ecart_ecoute(ar_manuel, ar_ecoute, niveau="complet"):
    """Ecart en mots entre la transcription manuelle et la sortie d'ecoute.

    Renvoie un taux entre 0 et 1 : distance de Levenshtein (mots) divisee par
    le nombre de mots de la reference. Le filtre D1 garde les segments dont
    ce taux est sous ecart_ecoute_max_pct / 100 (seuil propose : 15 %).
    Leve ValueError si la reference est vide : sans reference, pas de mesure.
    """
    mots_reference = mots_arabe(ar_manuel, niveau)
    if not mots_reference:
        raise ValueError("la transcription de reference est vide : pas de mesure possible")
    return distance_mots(ar_manuel, ar_ecoute, niveau) / len(mots_reference)


def normaliser_fr(texte):
    """Normalisation legere d'un texte francais pour les mesures.

    Apostrophe typographique unifiee, minuscules, espaces simples. Sert a
    comparer des textes francais entre eux, jamais a remplacer le texte stocke.
    """
    if not isinstance(texte, str):
        raise TypeError("texte attendu en chaine de caracteres")
    texte = texte.replace("\u2019", "'").replace("\u00a0", " ")
    texte = _MARKS_BIDI.sub("", texte)
    texte = unicodedata.normalize("NFC", texte)
    return re.sub(r"\s+", " ", texte).strip().lower()
