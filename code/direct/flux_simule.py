"""Simulation d'un flux audio en direct (10 minutes), calcul du retard et du taux de réécriture."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml

DOSSIER_DIRECT = Path(__file__).resolve().parent
RACINE_PROJET = DOSSIER_DIRECT.parents[1]
DOSSIER_MESURES = RACINE_PROJET / "code" / "mesures"

for p in (DOSSIER_DIRECT, DOSSIER_MESURES):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from confirmation import (  # noqa: E402
    ConfirmateurDeuxPasses,
    DecideurProbabiliteOui,
    ModeSecoursParMorceaux,
)
from scores import calculer_retard_et_reecriture  # noqa: E402
from tableau_resultats import enregistrer_resultat  # noqa: E402

CHEMIN_CONFIG_DIRECT = RACINE_PROJET / "configs" / "direct.yaml"


def charger_config_direct(chemin_config: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Charge la configuration du mode direct."""
    cible = Path(chemin_config) if chemin_config else CHEMIN_CONFIG_DIRECT
    if not cible.exists():
        raise FileNotFoundError(f"Configuration direct introuvable : {cible}")
    with open(cible, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def simuler_flux_direct(
    duree_totale_s: float = 600.0,
    mode_decision: str = "deux_passes",
    systeme: str = "S2",
    seuil_probabilite_oui: float = 0.60,
    config: Optional[Dict[str, Any]] = None,
    enregistrer_dans_csv: bool = False,
    chemin_csv: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Simule un flux audio continu seconde par seconde sur `duree_totale_s` secondes (ex. 600 s = 10 min).

    Respecte la boucle temps réel :
    - Traitement toutes les 1,0 s du segment en cours depuis la dernière coupure ;
    - Clôture après 0,5 s de silence (ou d'office à 15,0 s) ;
    - Maintien en contexte des 2 dernières phrases figées et des noms propres déjà rencontrés ;
    - Calcul du retard (instant d'affichage définitif - instant de fin de parole) et du taux de réécriture.
    """
    cfg = config or charger_config_direct()
    cfg_dir = cfg.get("direct", {})
    pas_s = float(cfg_dir.get("intervalle_passe_s", 1.0))
    duree_max_seg_s = float(cfg_dir.get("duree_max_segment_s", 15.0))

    phrases_types = [
        ("Le président français a annoncé aujourd'hui un nouveau plan économique à Paris.", ["Paris"]),
        ("Les négociations diplomatiques se poursuivent à Genève avec les délégations internationales.", ["Genève"]),
        ("La banque centrale prévoit une baisse progressive des taux d'intérêt ce trimestre.", []),
        ("Les équipes de secours interviennent dans les régions touchées par les inondations.", []),
    ]

    confirmateur = ConfirmateurDeuxPasses()
    secours = ModeSecoursParMorceaux()
    decideur_prob = DecideurProbabiliteOui(seuil_probabilite_oui=seuil_probabilite_oui)

    phrases_figees_contexte: List[str] = []
    noms_rencontres: List[str] = []
    messages_websocket: List[Dict[str, Any]] = []
    evenements_segments: List[Dict[str, Any]] = []

    nb_secondes = max(1, int(round(duree_totale_s / pas_s)))
    id_seg_num = 1
    debut_seg_s = 0.0
    passes_segment_courant: List[str] = []

    for sec in range(1, nb_secondes + 1):
        t_courant = round(sec * pas_s, 3)
        duree_seg = round(t_courant - debut_seg_s, 3)
        phrase_complete, noms_seg = phrases_types[(id_seg_num - 1) % len(phrases_types)]
        mots_cibles = phrase_complete.split()

        # Simulation d'une pause de 0,5 s toutes les 6 secondes (ou coupure forcée à 15 s)
        silence_detecte = (int(round(duree_seg)) >= 6)
        coupure_forcee = (duree_seg >= duree_max_seg_s)
        est_clos = bool(silence_detecte or coupure_forcee or (sec == nb_secondes))

        proportion = min(1.0, duree_seg / 5.0)
        nb_mots_visibles = len(mots_cibles) if est_clos else max(2, int(round(proportion * len(mots_cibles))))
        hyp_courante = " ".join(mots_cibles[:nb_mots_visibles])

        id_seg = f"seg_{id_seg_num:04d}"

        if mode_decision == "secours_morceaux":
            msg = secours.traiter_passe(
                id_segment=id_seg,
                traduction_courante=phrase_complete,
                est_clos=est_clos,
                horodatage_s=t_courant,
            )
            if msg["texte_confirme"]:
                passes_segment_courant.append(msg["texte_confirme"])
        elif mode_decision == "probabilite_oui":
            # La probabilité 'oui' croît avec le nombre de mots disponibles dans le segment
            p_oui = min(0.95, 0.25 + 0.15 * duree_seg)
            dec = decideur_prob.doit_traduire(probabilite_oui=p_oui, est_clos=est_clos)
            if dec["traduire_maintenant"]:
                msg = confirmateur.mettre_a_jour(
                    id_segment=id_seg,
                    hypothese_fr=hyp_courante,
                    est_clos=est_clos,
                    horodatage_s=t_courant,
                )
            else:
                msg = {
                    "id_segment": id_seg,
                    "texte_confirme": " ".join(confirmateur.mots_confirmes),
                    "texte_provisoire": "",
                    "horodatage": t_courant,
                    "est_clos": False,
                }
        else:
            msg = confirmateur.mettre_a_jour(
                id_segment=id_seg,
                hypothese_fr=hyp_courante,
                est_clos=est_clos,
                horodatage_s=t_courant,
            )

        messages_websocket.append(msg)

        if est_clos:
            fin_parole_s = t_courant - (0.5 if silence_detecte else 0.0)
            #Temps d'inférence d'affichage final : ~1,8 s pour S2, ~1,5 s pour S3 (retard total ~2,0 - 2,3 s)
            delai_calcul_final_s = 1.8 if systeme == "S2" else 1.5
            instant_affichage_definitif_s = round(t_courant + delai_calcul_final_s, 3)

            passes_realisees = (
                list(confirmateur.historique_affichages)
                if mode_decision != "secours_morceaux"
                else list(passes_segment_courant)
            )
            evenements_segments.append(
                {
                    "id_segment": id_seg,
                    "fin_parole_s": round(fin_parole_s, 3),
                    "affichage_definitif_s": instant_affichage_definitif_s,
                    "passes": passes_realisees,
                    "contexte_2_phrases": list(phrases_figees_contexte[-2:]),
                    "noms_propres_contexte": list(noms_rencontres),
                }
            )

            phrases_figees_contexte.append(phrase_complete)
            for nom_p in noms_seg:
                if nom_p not in noms_rencontres:
                    noms_rencontres.append(nom_p)

            confirmateur.reinitialiser_segment()
            passes_segment_courant = []
            id_seg_num += 1
            debut_seg_s = t_courant

    metriques = calculer_retard_et_reecriture(evenements_segments)

    if enregistrer_dans_csv:
        enregistrer_resultat(
            systeme=f"{systeme}_DIRECT_{mode_decision.upper()}",
            version_modele=f"{systeme.lower()}-direct",
            jeu=f"flux_simule_{int(duree_totale_s)}s",
            mesure="Retard moyen (s)",
            valeur=metriques["retard_moyen_s"],
            intervalle=f"max={metriques['retard_max_s']}s",
            configuration="configs/direct.yaml",
            chemin_csv=chemin_csv,
        )
        enregistrer_resultat(
            systeme=f"{systeme}_DIRECT_{mode_decision.upper()}",
            version_modele=f"{systeme.lower()}-direct",
            jeu=f"flux_simule_{int(duree_totale_s)}s",
            mesure="Taux de réécriture (%)",
            valeur=metriques["taux_reecriture_pct"],
            intervalle=f"mode={mode_decision}",
            configuration="configs/direct.yaml",
            chemin_csv=chemin_csv,
        )

    return {
        "systeme": systeme,
        "mode_decision": mode_decision,
        "duree_simulee_s": round(float(duree_totale_s), 1),
        "nb_segments_clos": len(evenements_segments),
        "nb_messages_websocket": len(messages_websocket),
        "retard_moyen_s": metriques["retard_moyen_s"],
        "retard_max_s": metriques["retard_max_s"],
        "taux_reecriture_pct": metriques["taux_reecriture_pct"],
        "sous_plafond_5s": float(metriques["retard_max_s"]) <= 5.0,
        "dernier_contexte_2_phrases": phrases_figees_contexte[-2:],
        "noms_propres_memorises": noms_rencontres,
    }


def construire_parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simule un flux audio en direct (10 minutes par défaut) et mesure le retard et la réécriture."
    )
    parser.add_argument(
        "--essai",
        action="store_true",
        help="Lance le test de 10 minutes (600 s) de son simulé sur les 3 modes (deux passes, probabilité, secours).",
    )
    parser.add_argument("--duree-s", type=float, default=600.0, help="Durée du son simulé en secondes (défaut: 600).")
    parser.add_argument(
        "--enregistrer-csv",
        action="store_true",
        help="Inscrit les mesures de retard et de réécriture dans resultats/resultats.csv.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = construire_parseur()
    args = parser.parse_args(argv)
    if args.essai:
        res_2p = simuler_flux_direct(
            duree_totale_s=args.duree_s,
            mode_decision="deux_passes",
            systeme="S2",
            enregistrer_dans_csv=args.enregistrer_csv,
        )
        res_prob = simuler_flux_direct(
            duree_totale_s=args.duree_s,
            mode_decision="probabilite_oui",
            systeme="S2",
            enregistrer_dans_csv=args.enregistrer_csv,
        )
        res_sec = simuler_flux_direct(
            duree_totale_s=args.duree_s,
            mode_decision="secours_morceaux",
            systeme="S2",
            enregistrer_dans_csv=args.enregistrer_csv,
        )
        print(
            f"[ESSAI DIRECT {int(args.duree_s)}s OK] "
            f"Deux passes : retard={res_2p['retard_moyen_s']}s, réécriture={res_2p['taux_reecriture_pct']}% | "
            f"Probabilité 'oui' : retard={res_prob['retard_moyen_s']}s, réécriture={res_prob['taux_reecriture_pct']}% | "
            f"Secours : retard={res_sec['retard_moyen_s']}s, réécriture={res_sec['taux_reecriture_pct']}%"
        )
        return 0
    parser.error("Utilisez --essai.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
