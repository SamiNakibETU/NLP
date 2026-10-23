"""Jeu de données S3 : vecteurs d'encodeur précalculés et texte cible (étape 4).

``DatasetS3`` lit ``index.jsonl`` écrit par ``code/ecoute/encoder_s3.py`` et
renvoie ``(vecteurs (T, d_enc) en float32, texte)``. ``regrouper_lot`` renvoie
la liste des vecteurs et la liste des textes : ``SpeechLLMS3`` gère lui-même
le rembourrage et les longueurs.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetS3(Dataset):
    """``champ_texte`` : "texte_ar" au temps 1, "texte_fr" au temps 2."""

    def __init__(self, dossier: Union[str, Path], champ_texte: str = "texte_ar"):
        self.dossier = Path(dossier)
        chemin = self.dossier / "index.jsonl"
        if not chemin.is_file():
            raise FileNotFoundError(f"index des vecteurs introuvable : {chemin} (lancer code/ecoute/encoder_s3.py)")
        with open(chemin, "r", encoding="utf-8") as fichier:
            self.index: List[Dict[str, Any]] = [json.loads(ligne) for ligne in fichier if ligne.strip()]
        if not self.index:
            raise ValueError(f"index vide : {chemin}")
        dimensions = {ligne["d_enc"] for ligne in self.index}
        if len(dimensions) != 1:
            raise ValueError(f"{chemin} : plusieurs d_enc ({sorted(dimensions)})")
        absents = [ligne["id"] for ligne in self.index if champ_texte not in ligne]
        if absents:
            raise KeyError(f"{chemin} : champ '{champ_texte}' absent pour {len(absents)} segment(s), dont {absents[0]}")
        self.d_enc = dimensions.pop()
        self.champ_texte = champ_texte
        self._matrices: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _matrice(self, nom: str) -> np.ndarray:
        if nom not in self._matrices:
            self._matrices[nom] = np.load(self.dossier / nom, mmap_mode="r")
        return self._matrices[nom]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, str]:
        ligne = self.index[i]
        debut = ligne["ligne_debut"]
        vecteurs = self._matrice(ligne["fichier_npy"])[debut: debut + ligne["nb_vecteurs"]]
        return torch.from_numpy(vecteurs.astype(np.float32)), ligne[self.champ_texte]


def regrouper_lot(exemples: Sequence[Tuple[torch.Tensor, str]]) -> Tuple[List[torch.Tensor], List[str]]:
    return [v for v, _ in exemples], [t for _, t in exemples]
