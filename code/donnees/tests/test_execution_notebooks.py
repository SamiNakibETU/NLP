#!/usr/bin/env python3
"""Test d'exécution de bout en bout des 6 notebooks du Jeu D."""

import sys
import unittest
import asyncio
from pathlib import Path

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

import nbformat
from nbclient import NotebookClient

DOSSIER_NOTEBOOKS = Path(__file__).resolve().parents[3] / "notebooks"
NOTEBOOKS = [
    "d1-selection-segments.ipynb",
    "d2-comparaison-modeles.ipynb",
    "d3-traduction-paquets.ipynb",
    "d4-filtres.ipynb",
    "d5-tirage-controle.ipynb",
    "d6-debuts-de-phrase.ipynb",
]


class TestExecutionNotebooksD(unittest.TestCase):
    def test_tous_les_notebooks_executent_sans_erreur(self):
        root_dir = Path(__file__).resolve().parents[2]
        for nb_nom in NOTEBOOKS:
            nb_path = DOSSIER_NOTEBOOKS / nb_nom
            self.assertTrue(nb_path.exists(), f"Le notebook {nb_nom} doit exister.")
            with self.subTest(notebook=nb_nom):
                with open(nb_path, "r", encoding="utf-8") as f:
                    nb = nbformat.read(f, as_version=4)

                client = NotebookClient(nb, timeout=60, kernel_name="python3", resources={"metadata": {"path": str(root_dir)}})
                # Doit s'exécuter complètement sans lever d'exception
                client.execute()
                print(f"[OK] Notebook exécuté avec succès : {nb_nom}")


if __name__ == "__main__":
    unittest.main()
