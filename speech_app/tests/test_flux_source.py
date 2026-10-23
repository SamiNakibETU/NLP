"""Chien de garde et reprise de la lecture d'une source serveur, avec de faux processus."""
import io
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flux_source import lire_avec_reprise


class FauxProc:
    """stdout rend `octets` puis, si `fige`, bloque jusqu'au kill (flux gelé), sinon rend b"" (fin du flux)."""

    def __init__(self, octets: bytes, fige: bool = False):
        self._buf, self._fige, self._tue = io.BytesIO(octets), fige, threading.Event()
        self.stdout = self

    def read(self, n):
        b = self._buf.read(n)
        if not b and self._fige:
            self._tue.wait(30)
        return b

    def kill(self):
        self._tue.set()

    def wait(self):
        return 0


def envelopper(proc):
    return proc


class TestReprise(unittest.TestCase):
    def test_flux_fige_est_relance(self):
        procs = [FauxProc(b"a" * 100, fige=True), FauxProc(b"b" * 50), FauxProc(b"")]
        crees, recus = [], []
        arret = threading.Event()

        def creer(cible):
            p = envelopper(procs[len(crees)])
            crees.append(cible)
            return p

        t0 = time.monotonic()
        res = lire_avec_reprise(creer, lambda: f"url{len(crees)}", recus.append, arret, delai_inactif_s=0.5, pause_s=0.05,
                                vides_consecutifs_max=1, journal=lambda m: None)
        self.assertLess(time.monotonic() - t0, 10)
        self.assertEqual(len(crees), 3)                 # URL résolue de nouveau à chaque reprise
        self.assertEqual(sum(len(b) for b in recus), 150)
        self.assertEqual(res["reprises"], 2)
        self.assertEqual(res["motif_fin"], "source_terminee")

    def test_fichier_lu_une_seule_fois(self):
        recus = []
        res = lire_avec_reprise(lambda c: envelopper(FauxProc(b"x" * 20)), lambda: "f.wav", recus.append,
                                threading.Event(), reprise=False)
        self.assertEqual(res["octets"], 20)
        self.assertEqual(res["reprises"], 0)

    def test_source_qui_ne_donne_plus_rien_s_arrete(self):
        res = lire_avec_reprise(lambda c: envelopper(FauxProc(b"")), lambda: "u", lambda b: None, threading.Event(),
                                pause_s=0.01, vides_consecutifs_max=3)
        self.assertEqual(res["motif_fin"], "source_terminee")


if __name__ == "__main__":
    unittest.main()
