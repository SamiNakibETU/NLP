"""Lecture d'une source audio serveur (yt-dlp + ffmpeg) avec chien de garde et reprise automatique.

Un direct YouTube/HLS peut se figer sans que ffmpeg s'arrête (réseau, lien signé expiré). ``lire_avec_reprise`` lit la
sortie du processus dans un fil ; si rien n'arrive pendant ``delai_inactif_s`` ou si le processus se termine alors que la
source est un direct, il le tue, résout l'URL de nouveau et relance, jusqu'à ``max_reprises`` fois. Un fichier (``reprise=False``)
n'est lu qu'une fois. Testable sans ffmpeg : ``creer_processus`` et ``resoudre`` sont injectés.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable

PAQUET_OCTETS = 8000


def lire_avec_reprise(creer_processus: Callable, resoudre: Callable[[], str], pousser: Callable[[bytes], None],
                      arret: threading.Event, *, reprise: bool = True, delai_inactif_s: float = 15.0,
                      max_reprises: int = 30, pause_s: float = 2.0, vides_consecutifs_max: int = 3,
                      horloge: Callable[[], float] = time.monotonic, journal: Callable[[str], None] = lambda m: None) -> dict:
    """Pousse l'audio dans ``pousser`` jusqu'à ``arret`` ou à la fin de la source. Renvoie {"octets", "reprises", "motif_fin"}."""
    total, reprises, vides = 0, 0, 0
    while not arret.is_set():
        cible = resoudre()
        proc = creer_processus(cible)
        file: queue.Queue = queue.Queue()

        def lecteur(p=proc, f=file):
            try:
                while True:
                    b = p.stdout.read(PAQUET_OCTETS)
                    f.put(b)
                    if not b:
                        return
            except Exception:                                    # noqa: BLE001 : processus tué
                f.put(b"")

        threading.Thread(target=lecteur, daemon=True).start()
        recu, dernier, motif = 0, horloge(), "arret"
        while not arret.is_set():
            try:
                b = file.get(timeout=0.25)
            except queue.Empty:
                if horloge() - dernier > delai_inactif_s:
                    motif = "inactivite"
                    break
                continue
            if not b:
                motif = "fin_du_flux"
                break
            pousser(b)
            recu += len(b)
            dernier = horloge()
        total += recu
        try:
            proc.kill()
            proc.wait()
        except Exception:                                        # noqa: BLE001
            pass
        if arret.is_set() or not reprise:
            return {"octets": total, "reprises": reprises, "motif_fin": "arret" if arret.is_set() else motif}
        vides = vides + 1 if recu == 0 else 0
        if vides >= vides_consecutifs_max:
            return {"octets": total, "reprises": reprises, "motif_fin": "source_terminee"}
        reprises += 1
        if reprises > max_reprises:
            raise RuntimeError(f"la source a décroché {max_reprises} fois : abandon")
        journal(f"source {motif} après {recu} octets : reprise {reprises}")
        arret.wait(pause_s)
    return {"octets": total, "reprises": reprises, "motif_fin": "arret"}
