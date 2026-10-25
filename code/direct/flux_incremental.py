#!/usr/bin/env python3
"""Mode direct incrémental : l'audio arrive par paquets, la fin du flux n'est jamais connue.

``FluxS3`` (flux_s3.py) rejoue un fichier entier, pour l'évaluation hors ligne. Ici, ``FluxIncremental``
reprend ses règles (silence relatif au niveau de parole, clôture après ``silence_cloture_s``, coupure
d'office au point le plus calme, durée minimale d'un segment) mais avance paquet par paquet :

    flux = FluxIncremental(moteur, reglages)
    flux.ajouter(paquet_float32_16khz)      # autant de fois qu'on veut
    for msg in flux.traiter(): ...          # messages provisoire / definitif à diffuser
    for msg in flux.terminer(): ...         # vide ce qui reste, à la fin du flux

Double passe à deux traducteurs :
  * toutes les ``intervalle_passe_s`` secondes d'audio, le segment en cours est transcrit en arabe par S3, puis
    traduit par un traducteur RAPIDE (Opus-MT) : message ``provisoire`` (texte gris) ;
  * à la clôture du segment, S3 retranscrit le segment entier, puis son Qwen le traduit avec les phrases arabes
    précédentes en contexte : message ``definitif`` (texte blanc), qui REMPLACE le provisoire du même ``id_segment``.

``moteur`` doit offrir trois méthodes (voir speech_app/live_moteurs.py : réel et factice) :
    transcrire(signal) -> str                       arabe
    traduire_provisoire(ar) -> str                  traduction rapide
    traduire_definitif(ar, contexte: list) -> str   traduction soignée avec les phrases arabes précédentes

Le retard est mesuré à l'horloge réelle : de l'arrivée du paquet qui contient la fin de la parole jusqu'à
l'émission du message définitif (``horloge`` est injectable pour les tests).
"""

import bisect
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from flux_s3 import FREQUENCE_ATTENDUE, FluxS3, charger_config_direct, energies_db, reglages_direct_s3  # noqa: E402
from scores import calculer_retard_et_reecriture  # noqa: E402

SR = FREQUENCE_ATTENDUE
HISTORIQUE_NIVEAU_S = 60.0   # le niveau de parole se mesure sur la dernière minute, pas sur toute l'écoute


class SuiviLocuteurs:
    """Qui parle, en ligne : chaque empreinte de voix rejoint la voix connue la plus proche (similarité cosinus
    au-dessus du seuil) ou ouvre une nouvelle voix. Le centre de chaque voix suit la moyenne de ses empreintes."""

    def __init__(self, seuil: float = 0.45, max_voix: int = 8):
        self.seuil, self.max_voix = seuil, max_voix
        self.centres: List[np.ndarray] = []
        self.effectifs: List[int] = []

    def attribuer(self, emb: np.ndarray) -> int:
        v = np.asarray(emb, dtype=np.float64)
        v = v / (np.linalg.norm(v) or 1.0)
        sims = [float(c @ v / (np.linalg.norm(c) or 1.0)) for c in self.centres]
        if sims and (max(sims) >= self.seuil or len(self.centres) >= self.max_voix):
            i = int(np.argmax(sims))
            self.centres[i] = self.centres[i] * self.effectifs[i] + v
            self.effectifs[i] += 1
            return i + 1
        self.centres.append(v)
        self.effectifs.append(1)
        return len(self.centres)


def reglages_direct_live() -> Dict[str, Any]:
    """Réglages de la page /live : ceux de direct_s3 (deux passes), complétés par la section direct_live."""
    reglages = reglages_direct_s3("deux_passes")
    reglages.update(charger_config_direct().get("direct_live") or {})
    return reglages


class FluxIncremental(FluxS3):
    """Sous-titrage en direct d'un flux reçu par paquets. Hérite de FluxS3 la règle de clôture (``_coupure``)."""

    def __init__(self, moteur, reglages: Optional[Dict[str, Any]] = None,
                 horloge: Callable[[], float] = time.perf_counter):
        super().__init__(None, reglages or reglages_direct_s3("deux_passes"))
        self.moteur = moteur
        self.horloge = horloge
        self._signal = np.zeros(SR * 60, dtype=np.float32)   # capacité doublée au besoin
        self.n = 0                                           # échantillons reçus
        self._arrivees_n: List[int] = []                     # échantillons cumulés à chaque paquet...
        self._arrivees_t: List[float] = []                   # ... et heure d'horloge correspondante
        self._t0: Optional[float] = None
        self._debut = 0
        self._numero = 1
        self._t_derniere_passe = 0.0                         # en secondes d'audio
        self._ar_provisoire: Optional[str] = None
        self._fr_provisoire = ""
        self._passes_provisoires: List[str] = []
        self.contexte_ar: List[str] = []
        self.niveau = 0.0                                    # dernier niveau (dBFS) du dernier paquet

    # ------------------------------------------------------------------ entrée
    def ajouter(self, paquet: np.ndarray) -> None:
        """Ajoute un paquet float32 (16 kHz, mono, [-1, 1])."""
        paquet = np.asarray(paquet, dtype=np.float32).ravel()
        if paquet.size == 0:
            return
        if self._t0 is None:
            self._t0 = self.horloge()
        fin = self.n + paquet.size
        while fin > len(self._signal):
            self._signal = np.concatenate([self._signal, np.zeros(len(self._signal), dtype=np.float32)])
        self._signal[self.n: fin] = paquet
        self.n = fin
        self._arrivees_n.append(fin)
        self._arrivees_t.append(self.horloge())
        e = energies_db(paquet, len(paquet))
        self.niveau = float(e[0]) if len(e) else -100.0

    def ajouter_pcm16(self, octets: bytes) -> None:
        """Ajoute des échantillons int16 little-endian (ce que envoie le navigateur)."""
        n = len(octets) // 2
        if n:
            self.ajouter(np.frombuffer(octets[: 2 * n], dtype="<i2").astype(np.float32) / 32768.0)

    def signal(self) -> np.ndarray:
        """Tout l'audio reçu (vue, pas copie)."""
        return self._signal[: self.n]

    @property
    def duree_s(self) -> float:
        return self.n / SR

    def _heure_arrivee(self, echantillon: int) -> float:
        """Heure d'horloge à laquelle l'échantillon est arrivé (celle du paquet qui le contenait)."""
        i = bisect.bisect_left(self._arrivees_n, max(1, echantillon))
        return self._arrivees_t[min(i, len(self._arrivees_t) - 1)] if self._arrivees_t else self.horloge()

    # ------------------------------------------------------------------ traitement
    def traiter(self, final: bool = False, provisoire: bool = True) -> List[Dict[str, Any]]:
        """Traite l'audio reçu : clôt tous les segments qui peuvent l'être, puis fait au plus une passe provisoire.

        ``provisoire=False`` saute la passe grise : l'appelant le demande quand l'audio s'accumule (calcul en retard),
        pour rattraper le direct avant de raffiner le gris."""
        r = self.reglages
        fenetre = int(r["fenetre_energie_s"] * SR)
        sortie: List[Dict[str, Any]] = []
        while self.n > self._debut:
            recu = self.n
            depart_niveau = max(0, recu - int(HISTORIQUE_NIVEAU_S * SR))
            niveaux = energies_db(self._signal[depart_niveau:recu], fenetre)
            relatif = (np.percentile(niveaux, r["centile_niveau_parole"]) if len(niveaux) else 0.0) - r["seuil_silence_db"]
            seuil_db = max(relatif, r["plancher_silence_db"])
            parole = energies_db(self._signal[self._debut:recu], fenetre) >= seuil_db
            if not parole.any():
                # Rien que du silence : on le jette, en gardant la trame incomplète du bout pour ne pas perdre de son.
                self._debut = recu - (recu - self._debut) % fenetre if not final else recu
                break
            self._debut += max(0, int(np.argmax(parole)) - int(round(r["marge_debut_s"] / r["fenetre_energie_s"]))) * fenetre
            coupure = self._coupure(self._signal, self._debut, recu, seuil_db)
            if coupure is None and final:
                coupure = {"fin_parole": recu, "coupure": recu, "reprise": recu}
            if coupure is None:
                break
            msg = self._clore(coupure)
            if msg is not None:
                sortie.append(msg)
        if not final and provisoire:
            msg = self._passe_provisoire()
            if msg is not None:
                sortie.append(msg)
        return sortie

    def terminer(self) -> List[Dict[str, Any]]:
        """Fin du flux : le segment en cours est clos tel quel."""
        return self.traiter(final=True)

    def _id(self) -> str:
        return f"seg_{self._numero:04d}"

    def _passe_provisoire(self) -> Optional[Dict[str, Any]]:
        r = self.reglages
        recu = self.n
        if (recu - self._debut) / SR < r["duree_min_passe_s"]:
            return None
        if recu / SR - self._t_derniere_passe < r["intervalle_passe_s"]:
            return None
        self._t_derniere_passe = recu / SR
        ar = self.moteur.transcrire(self._signal[self._debut:recu].copy()).strip()
        self._passe_couvre = (self._debut, recu, ar)            # pour la clôture : arabe déjà transcrit sur [debut, recu]
        if not ar or ar == self._ar_provisoire:
            return None                      # rien de neuf : on ne retraduit pas, on n'envoie rien
        self._ar_provisoire = ar
        precedent = self._fr_provisoire.split()
        self._fr_provisoire = self.moteur.traduire_provisoire(ar).strip()
        self._passes_provisoires.append(self._fr_provisoire)
        # Début commun aux deux dernières passes : la page peut le figer et ne réécrire que la fin.
        mots, n = self._fr_provisoire.split(), 0
        while n < min(len(mots), len(precedent)) and mots[n] == precedent[n]:
            n += 1
        return {"type": "provisoire", "id_segment": self._id(), "fr": self._fr_provisoire,
                "fr_stable": " ".join(mots[:n]), "ar": ar,
                "debut_s": round(self._debut / SR, 2), "fin_s": round(recu / SR, 2), "retard_s": None}

    def _clore(self, coupure: Dict[str, int]) -> Optional[Dict[str, Any]]:
        debut, fin = self._debut, coupure["coupure"]
        # La dernière passe grise couvre presque tout le segment : on reprend son arabe au lieu de tout retranscrire.
        couvre = getattr(self, "_passe_couvre", None)
        marge = int(self.reglages.get("reutiliser_passe_max_s", 0.0) * SR)
        if couvre and couvre[2] and couvre[0] == debut and 0 <= fin - couvre[1] <= marge:
            ar = couvre[2]
        else:
            ar = self.moteur.transcrire(self._signal[debut:fin].copy()).strip()
        identifiant = self._id()
        base = {"type": "definitif", "id_segment": identifiant, "debut_s": round(debut / SR, 2),
                "fin_s": round(fin / SR, 2)}
        if ar:
            fr = self.moteur.traduire_definitif(ar, list(self.contexte_ar)).strip()
            locuteur = self._locuteur(self._signal[debut:fin])
            maintenant = self.horloge()
            arrivee_fin_parole = self._heure_arrivee(coupure["fin_parole"])
            retard = maintenant - arrivee_fin_parole
            self.evenements.append({
                "id_segment": identifiant, "debut_s": base["debut_s"], "fin_s": base["fin_s"],
                "fin_parole_s": round(arrivee_fin_parole - self._t0, 3),
                "affichage_definitif_s": round(maintenant - self._t0, 3),
                "passes_provisoires": list(self._passes_provisoires), "texte_final": fr, "ar": ar})
            self.contexte_ar.append(ar)
            msg = dict(base, fr=fr, ar=ar, retard_s=round(retard, 3))
            if locuteur is not None:
                msg["locuteur"] = locuteur
                msg["changement_locuteur"] = self._dernier_locuteur is not None and locuteur != self._dernier_locuteur
                self._dernier_locuteur = locuteur
        else:
            msg = dict(base, fr="", ar="", retard_s=None, vide=True)     # bruit sans parole : efface le gris
        self._debut = coupure["reprise"]
        self._numero += 1
        self._t_derniere_passe = self._debut / SR
        self._ar_provisoire, self._fr_provisoire, self._passes_provisoires = None, "", []
        self._passe_couvre = None
        return msg

    def _locuteur(self, morceau: np.ndarray) -> Optional[int]:
        """Numéro de voix du segment, ou None (moteur sans empreinte, segment trop court) : on garde alors la voix d'avant."""
        if not self.reglages.get("detecter_locuteurs", False) or not hasattr(self.moteur, "empreinte"):
            return None
        if not hasattr(self, "_suivi"):
            self._suivi = SuiviLocuteurs(self.reglages.get("seuil_locuteur", 0.45))
            self._dernier_locuteur = None
        emb = self.moteur.empreinte(morceau.copy())
        return self._suivi.attribuer(emb) if emb is not None else self._dernier_locuteur

    # ------------------------------------------------------------------ bilan
    def bilan(self) -> Dict[str, Any]:
        """Retard (fin de parole -> affichage définitif, en s réelles) et taux de réécriture (gris -> blanc)."""
        if not self.evenements:
            return {"nb_segments": 0}
        return calculer_retard_et_reecriture(self.evenements)
