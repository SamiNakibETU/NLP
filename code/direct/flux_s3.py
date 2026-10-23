#!/usr/bin/env python3
"""Mode direct avec le vrai système S3 : un fichier audio rejoué comme un flux et sous-titré au fil de l'eau.

Toutes les ``intervalle_passe_s`` secondes, le segment en cours (de la dernière coupure à
l'instant présent) est transcrit par S3 (encodeur Cohere -> projecteur -> Qwen2.5), puis
traduit en français par le même LLM. C'est une cascade, parce que la traduction directe
depuis l'audio ne marche pas avant le temps 2 (journal_s3.md). La règle des deux passes
(confirmation.py) décide quels mots français sont affichés comme confirmés.

Un segment est clos après ``silence_cloture_s`` de silence, ou d'office à
``duree_max_segment_s`` : la coupure tombe alors au point le plus calme des
``recherche_coupure_s`` dernières secondes, et la suite part dans le segment suivant.

Horloge : chaque passe démarre avec l'audio reçu jusqu'à ``t`` et finit à ``t + calcul``.
La passe suivante démarre à ``t + max(intervalle, calcul)`` : un calcul plus long que
l'intervalle retarde le sous-titrage, comme en vrai. ``temps_reel=True`` attend en plus
entre les passes pour suivre la lecture de l'audio (notebook) ; sinon la boucle va aussi
vite que possible, pour les mêmes mesures.

    python code/direct/flux_s3.py --audio extrait.wav [--mode secours_morceaux]
    python code/direct/flux_s3.py --essai
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

DOSSIER_DIRECT = Path(__file__).resolve().parent
RACINE_PROJET = DOSSIER_DIRECT.parents[1]
for _dossier in (DOSSIER_DIRECT, RACINE_PROJET / "code" / "mesures", RACINE_PROJET / "code" / "projecteur"):
    if str(_dossier) not in sys.path:
        sys.path.insert(0, str(_dossier))

from confirmation import ConfirmateurDeuxPasses  # noqa: E402
from flux_simule import charger_config_direct  # noqa: E402
from scores import calculer_retard_et_reecriture  # noqa: E402
from transcrire_s3 import FREQUENCE_ATTENDUE, point_le_plus_calme  # noqa: E402

MODES = ("deux_passes", "secours_morceaux")


def energies_db(extrait: np.ndarray, fenetre: int) -> np.ndarray:
    """Niveau (dB) de chaque trame de ``fenetre`` échantillons ; la trame incomplète de la fin est ignorée."""
    nb = len(extrait) // fenetre
    if nb == 0:
        return np.zeros(0)
    rms = np.sqrt((extrait[: nb * fenetre].reshape(nb, fenetre).astype(np.float64) ** 2).mean(axis=1))
    return 20 * np.log10(rms + 1e-8)


def debut_silence(silencieuses: np.ndarray, nb_min: int) -> Optional[int]:
    """Première trame d'une suite d'au moins ``nb_min`` trames silencieuses précédée de parole, sinon None."""
    parole_vue, debut, longueur = False, 0, 0
    for i, silence in enumerate(silencieuses):
        if not silence:
            parole_vue, longueur = True, 0
            continue
        if longueur == 0:
            debut = i
        longueur += 1
        if parole_vue and longueur >= nb_min:
            return debut
    return None


class FluxS3:
    """Sous-titrage en direct d'un signal 16 kHz avec un ``TranscripteurS3`` déjà chargé.

    ``reglages`` : section ``direct`` de configs/direct.yaml complétée par ``direct_s3``.
    Après ``diffuser``, ``evenements`` contient un bilan par segment clos (retard, passes).
    """

    def __init__(self, transcripteur, reglages: Dict[str, Any]):
        if reglages["mode"] not in MODES:
            raise ValueError(f"mode inconnu : {reglages['mode']} (connus : {MODES})")
        self.transcripteur = transcripteur
        self.reglages = reglages
        self.evenements: List[Dict[str, Any]] = []
        self._dernier_ar, self._dernier_fr = None, ""

    def _traduire(self, morceau: np.ndarray) -> Dict[str, str]:
        """Transcription arabe puis traduction ; la traduction précédente est reprise si l'arabe n'a pas changé."""
        ar = " ".join(s["texte"] for s in self.transcripteur.transcrire_signal(morceau)).strip()
        if ar != self._dernier_ar:
            self._dernier_ar, self._dernier_fr = ar, self.transcripteur.traduire([ar])[0]
        return {"ar": ar, "fr": self._dernier_fr}

    def _coupure(self, signal: np.ndarray, debut: int, recu: int, seuil_db: float) -> Optional[Dict[str, int]]:
        """Où clore le segment [debut, recu] : {"fin_parole", "coupure", "reprise"} en échantillons, ou None."""
        r, sr = self.reglages, FREQUENCE_ATTENDUE
        fenetre = int(r["fenetre_energie_s"] * sr)
        silencieuses = energies_db(signal[debut:recu], fenetre) < seuil_db
        silencieuses[: int(round(r["duree_min_segment_s"] / r["fenetre_energie_s"]))] = False  # pas de segment trop court
        i = debut_silence(silencieuses, int(round(r["silence_cloture_s"] / r["fenetre_energie_s"])))
        if i is not None:
            fin_parole = debut + i * fenetre
            moitie = int(r["silence_cloture_s"] * sr) // 2
            return {"fin_parole": fin_parole, "coupure": fin_parole + moitie, "reprise": fin_parole + 2 * moitie}
        longueur_max = int(r["duree_max_segment_s"] * sr)
        if recu - debut < longueur_max:
            return None
        recherche = int(r["recherche_coupure_s"] * sr)
        zone_debut = debut + longueur_max - recherche
        coupure = zone_debut + point_le_plus_calme(signal[zone_debut: debut + longueur_max], fenetre)
        return {"fin_parole": coupure, "coupure": coupure, "reprise": coupure}

    def diffuser(self, signal: np.ndarray, temps_reel: bool = False) -> Iterator[Dict[str, Any]]:
        """Messages d'affichage dans l'ordre du temps : un par passe, avec ``texte_confirme``,
        ``texte_provisoire``, ``horodatage`` (instant d'affichage, en s d'audio) et ``est_clos``."""
        r, sr = self.reglages, FREQUENCE_ATTENDUE
        fenetre = int(r["fenetre_energie_s"] * sr)
        confirmateur = ConfirmateurDeuxPasses()
        self.evenements = []
        self._dernier_ar, self._dernier_fr = None, ""
        debut, t, numero = 0, 0.0, 1
        duree = len(signal) / sr
        depart = time.perf_counter()

        def passe(fin: int, est_clos: bool, t_passe: float):
            a = time.perf_counter()
            textes = self._traduire(signal[debut:fin])
            calcul = time.perf_counter() - a
            if r["mode"] == "deux_passes" or est_clos:
                msg = confirmateur.mettre_a_jour(f"seg_{numero:04d}", textes["fr"], est_clos, t_passe + calcul)
            else:
                msg = {"id_segment": f"seg_{numero:04d}", "texte_confirme": "", "texte_provisoire": "",
                       "horodatage": round(t_passe + calcul, 3), "est_clos": False}
            msg.update(ar=textes["ar"], calcul_s=round(calcul, 3), debut_s=round(debut / sr, 2), fin_s=round(fin / sr, 2))
            return msg, calcul

        while True:
            t = min(duree, t + r["intervalle_passe_s"])
            if temps_reel:
                time.sleep(max(0.0, t - (time.perf_counter() - depart)))
            recu = int(t * sr)
            # Niveau de parole : centile des trames déjà reçues ; le silence est sous ce niveau moins seuil_silence_db,
            # et toujours sous plancher_silence_db (sinon un flux muet serait pris pour de la parole).
            niveaux = energies_db(signal[:recu], fenetre)
            relatif = (np.percentile(niveaux, r["centile_niveau_parole"]) if len(niveaux) else 0.0) - r["seuil_silence_db"]
            seuil_db = max(relatif, r["plancher_silence_db"])
            parole = energies_db(signal[debut:recu], fenetre) >= seuil_db
            fin_flux = recu >= len(signal)
            if not parole.any():
                debut = recu  # rien que du silence : on ne garde pas cet audio
                if fin_flux:
                    return
                continue
            # Le segment commence à la parole (moins une petite marge), pas dans le silence qui la précède.
            debut += max(0, int(np.argmax(parole)) - int(round(r["marge_debut_s"] / r["fenetre_energie_s"]))) * fenetre
            coupure = self._coupure(signal, debut, recu, seuil_db)
            if coupure is None and fin_flux:
                coupure = {"fin_parole": len(signal), "coupure": len(signal), "reprise": len(signal)}
            calcul = 0.0
            if coupure is not None:
                provisoires = list(confirmateur.historique_affichages)
                msg, calcul = passe(coupure["coupure"], True, t)
                self.evenements.append({
                    "id_segment": msg["id_segment"], "debut_s": msg["debut_s"], "fin_s": msg["fin_s"],
                    "fin_parole_s": round(coupure["fin_parole"] / sr, 3), "affichage_definitif_s": msg["horodatage"],
                    "passes_provisoires": provisoires,
                    "texte_final": msg["texte_confirme"], "ar": msg["ar"]})
                yield msg
                confirmateur.reinitialiser_segment()
                self._dernier_ar, self._dernier_fr = None, ""
                debut, numero = coupure["reprise"], numero + 1
                if fin_flux:
                    return
            elif (recu - debut) / sr >= r["duree_min_passe_s"] and r["mode"] == "deux_passes":
                msg, calcul = passe(recu, False, t)
                yield msg
            # Un calcul plus long que l'intervalle décale la passe suivante.
            t = min(duree, t + max(0.0, calcul - r["intervalle_passe_s"]))

    def bilan(self) -> Dict[str, Any]:
        """Retard (affichage définitif - fin de parole) et taux de réécriture sur les segments clos."""
        if not self.evenements:
            return {"nb_segments": 0}
        return calculer_retard_et_reecriture(self.evenements)


def wav_base64(signal: np.ndarray) -> str:
    """Signal float 16 kHz -> fichier WAV 16 bits encodé en base64 (pour une page HTML)."""
    import base64
    import io
    import wave

    tampon = io.BytesIO()
    with wave.open(tampon, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(FREQUENCE_ATTENDUE)
        wav.writeframes((np.clip(signal, -1.0, 1.0) * 32767).astype("<i2").tobytes())
    return base64.b64encode(tampon.getvalue()).decode("ascii")


def lecteur_html(signal: np.ndarray, messages: List[Dict[str, Any]], nb_lignes_figees: int = 3) -> str:
    """Lecteur audio et sous-titres pilotés par la position de lecture.

    Chaque message s'affiche quand ``audio.currentTime`` atteint son ``horodatage`` (instant
    où il serait apparu en direct, calcul compris) : le retard du système est conservé, et
    l'affichage suit lecture, pause et retour en arrière.
    """
    import json
    import uuid

    ident = "flux_" + uuid.uuid4().hex[:8]
    champs = ("horodatage", "est_clos", "texte_confirme", "texte_provisoire", "id_segment", "calcul_s")
    donnees = json.dumps([{k: m[k] for k in champs} for m in messages], ensure_ascii=True).replace("</", "<\\/")
    return f"""
<div id="{ident}" style="font-family:system-ui,sans-serif;max-width:900px">
  <audio controls style="width:100%" src="data:audio/wav;base64,{wav_base64(signal)}"></audio>
  <div style="background:#111827;border-radius:10px;padding:14px 18px;margin-top:8px;font-size:1.25rem;line-height:1.5;min-height:7em">
    <div class="figes" style="color:#d1d5db"></div>
    <div><span class="confirme" style="color:#ffffff;font-weight:600"></span>
         <span class="provisoire" style="color:#9ca3af"></span></div>
    <div class="etat" style="color:#6b7280;font-size:0.8rem;margin-top:8px"></div>
  </div>
</div>
<script>
(() => {{
  const racine = document.getElementById("{ident}");
  const audio = racine.querySelector("audio"), messages = {donnees};
  const figes = racine.querySelector(".figes"), confirme = racine.querySelector(".confirme");
  const provisoire = racine.querySelector(".provisoire"), etat = racine.querySelector(".etat");
  const ligne = (texte) => {{ const d = document.createElement("div"); d.textContent = texte; return d; }};
  let dernier = -2;
  const afficher = () => {{
    const t = audio.currentTime;
    let n = 0;
    while (n < messages.length && messages[n].horodatage <= t) n++;
    etat.textContent = `t = ${{t.toFixed(1)}} s` + (n ? ` · ${{messages[n - 1].id_segment}} · calcul ${{messages[n - 1].calcul_s.toFixed(1)}} s` : "");
    if (n - 1 === dernier) return;
    dernier = n - 1;
    const clos = messages.slice(0, n).filter((m) => m.est_clos).map((m) => m.texte_confirme);
    figes.replaceChildren(...clos.slice(-{nb_lignes_figees}).map(ligne));
    const courant = n && !messages[n - 1].est_clos ? messages[n - 1] : null;
    confirme.textContent = courant ? courant.texte_confirme : "";
    provisoire.textContent = courant ? courant.texte_provisoire : "";
  }};
  const boucle = () => {{ afficher(); requestAnimationFrame(boucle); }};
  boucle();
}})();
</script>
"""


def reglages_direct_s3(mode: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = config or charger_config_direct()
    reglages = dict(config["direct"], **config["direct_s3"])
    if mode:
        reglages["mode"] = mode
    return reglages


def construire_parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sous-titrage français en direct d'un fichier audio arabe avec S3.")
    parser.add_argument("--audio", type=str, help="Fichier audio (wav, flac, mp3...).")
    parser.add_argument("--mode", choices=MODES, help="Défaut : direct_s3.mode de configs/direct.yaml.")
    parser.add_argument("--temps-reel", action="store_true", help="Suivre le rythme de l'audio au lieu d'aller au plus vite.")
    parser.add_argument("--essai", action="store_true", help="Petits modèles aléatoires et bruit de 30 s.")
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    from transcrire_s3 import TranscripteurS3, charger_audio, transcripteur_factice

    args = construire_parseur().parse_args(arguments)
    reglages = reglages_direct_s3(args.mode)
    if args.essai:
        transcripteur = transcripteur_factice(reglages["section_s3"])
        rng = np.random.default_rng(0)
        signal = (0.1 * rng.standard_normal(30 * FREQUENCE_ATTENDUE)).astype(np.float32)
        signal[12 * FREQUENCE_ATTENDUE: 13 * FREQUENCE_ATTENDUE] *= 0.001  # une pause d'1 s
    else:
        if not args.audio:
            raise SystemExit("--audio est obligatoire hors mode --essai")
        transcripteur = TranscripteurS3.charger(reglages["section_s3"])
        signal = charger_audio(args.audio)
    flux = FluxS3(transcripteur, reglages)
    for msg in flux.diffuser(signal, temps_reel=args.temps_reel):
        etat = "CLOS" if msg["est_clos"] else "    "
        print(f"[{msg['horodatage']:7.2f} s] {etat} {msg['texte_confirme']} | {msg['texte_provisoire']}")
    bilan = flux.bilan()
    print(bilan)
    return bilan


if __name__ == "__main__":
    main()
