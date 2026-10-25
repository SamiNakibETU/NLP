"""Mécanismes d'affichage en direct : secours par morceaux, règle des deux passes et décision par probabilité."""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


class ModeSecoursParMorceaux:
    """Solution de secours : chaque segment clos est traduit et affiché d'un seul bloc."""

    def __init__(self):
        self.segments_figes: List[Dict[str, Any]] = []

    def traiter_passe(
        self,
        id_segment: str,
        traduction_courante: str,
        est_clos: bool,
        horodatage_s: float,
    ) -> Dict[str, Any]:
        """N'affiche rien tant que le segment n'est pas clos ; affiche tout en confirmé à la clôture."""
        if not est_clos:
            return {
                "id_segment": id_segment,
                "texte_confirme": "",
                "texte_provisoire": "",
                "horodatage": round(float(horodatage_s), 3),
                "est_clos": False,
            }
        texte_net = traduction_courante.strip()
        msg = {
            "id_segment": id_segment,
            "texte_confirme": texte_net,
            "texte_provisoire": "",
            "horodatage": round(float(horodatage_s), 3),
            "est_clos": True,
        }
        self.segments_figes.append(msg)
        return msg


class ConfirmateurDeuxPasses:
    """Règle des deux passes : un mot français est confirmé quand il occupe la même place deux passes de suite.

    Un mot confirmé ne change plus jamais. À la clôture du segment (silence >= 0,5 s ou durée >= 15 s),
    la suite du segment est figée et rattachée aux mots déjà confirmés.
    """

    def __init__(self):
        self.mots_confirmes: List[str] = []
        self.mots_passe_precedente: List[str] = []
        self.historique_affichages: List[str] = []

    def reinitialiser_segment(self) -> None:
        self.mots_confirmes = []
        self.mots_passe_precedente = []
        self.historique_affichages = []

    def mettre_a_jour(
        self,
        id_segment: str,
        hypothese_fr: str,
        est_clos: bool,
        horodatage_s: float,
    ) -> Dict[str, Any]:
        mots_courants = hypothese_fr.strip().split()
        n_deja = len(self.mots_confirmes)

        # Les mots déjà confirmés sont immuables ; on regarde les positions suivantes
        if not est_clos:
            idx = n_deja
            while (
                idx < len(mots_courants)
                and idx < len(self.mots_passe_precedente)
                and mots_courants[idx] == self.mots_passe_precedente[idx]
            ):
                self.mots_confirmes.append(mots_courants[idx])
                idx += 1

            n_conf = len(self.mots_confirmes)
            mots_provisoires = mots_courants[n_conf:]
            self.mots_passe_precedente = list(self.mots_confirmes) + mots_provisoires
        else:
            # À la clôture, la traduction finale REMPLACE tout le provisoire : la recoller derrière les mots
            # déjà confirmés produirait des doublons dès que le début de la phrase a changé. Une traduction
            # finale vide n'efface pas ce qui était confirmé.
            if mots_courants:
                self.mots_confirmes = list(mots_courants)
            mots_provisoires = []
            self.mots_passe_precedente = list(self.mots_confirmes)

        texte_conf = " ".join(self.mots_confirmes)
        texte_prov = " ".join(mots_provisoires)
        complet = " ".join(x for x in (texte_conf, texte_prov) if x).strip()
        if complet:
            self.historique_affichages.append(complet)

        return {
            "id_segment": id_segment,
            "texte_confirme": texte_conf,
            "texte_provisoire": texte_prov,
            "horodatage": round(float(horodatage_s), 3),
            "est_clos": bool(est_clos),
        }


class DecideurProbabiliteOui:
    """Variante de décision : lit la probabilité du token 'oui' avant de traduire un début de phrase."""

    def __init__(self, seuil_probabilite_oui: float = 0.60):
        if not (0.0 <= seuil_probabilite_oui <= 1.0):
            raise ValueError(f"Le seuil doit être dans [0, 1] (reçu : {seuil_probabilite_oui})")
        self.seuil = float(seuil_probabilite_oui)

    @staticmethod
    def calculer_probabilite_oui_depuis_logits(logit_oui: float, logit_non: float) -> float:
        """Calcule P('oui') par softmax entre les logits des tokens 'oui' et 'non'."""
        m = max(float(logit_oui), float(logit_non))
        exp_oui = math.exp(float(logit_oui) - m)
        exp_non = math.exp(float(logit_non) - m)
        return exp_oui / (exp_oui + exp_non)

    def doit_traduire(
        self,
        probabilite_oui: float,
        est_clos: bool = False,
    ) -> Dict[str, Any]:
        """Autorise la traduction si P('oui') >= seuil ou si le segment est clos."""
        p_val = float(probabilite_oui)
        declencher = bool(est_clos or (p_val >= self.seuil))
        return {
            "probabilite_oui": round(p_val, 4),
            "seuil": self.seuil,
            "traduire_maintenant": declencher,
        }
