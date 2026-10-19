#!/usr/bin/env python3
"""Générateur et vérificateur des 6 notebooks du Jeu D (D1 à D6).

Conforme à la consigne :
- Les notebooks sont de FINES INTERFACES qui importent les fonctions centrales de `pipeline_mgb2.py`.
- Aucune duplication de traitement dans les notebooks.
- Conservation d'un mode SIMULATION hors ligne et d'un mode ESSAI_REEL / COMPLET.
- Interdiction stricte de tout repli silencieux vers le mock.
"""

from pathlib import Path
import nbformat as nbf

DOSSIER_NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks"
DOSSIER_NOTEBOOKS.mkdir(exist_ok=True)


def creer_cellule_md(source):
    return nbf.v4.new_markdown_cell(source)


def creer_cellule_code(source):
    return nbf.v4.new_code_cell(source)


# ==============================================================================
# NOTEBOOK D1 : SÉLECTION DES SEGMENTS MGB-2
# ==============================================================================
def generer_nb_d1():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        creer_cellule_md(
            """# Étape D1 — Sélection des segments MGB-2 (Jeux D et A)

Ce notebook constitue la fine interface d'exécution pour l'étape D1.
Toute la logique de filtrage, de transcription ASR, de gestion des quotas et de reprise
est centralisée dans le module [`pipeline_mgb2.py`](file:///code/donnees/pipeline_mgb2.py).

### Modes d'exécution :
1. `MODE_REEL = False` (SIMULATION) : test hors ligne sans réseau ni GPU.
2. `MODE_REEL = True` (ESSAI RÉEL / COMPLET) : streaming officiel Kaldi MGB-2 et Whisper ASR réel.
"""
        )
    )

    cells.append(
        creer_cellule_code(
            """import os
import sys
from pathlib import Path

# Résolution des modules du projet
for p in [Path.cwd(), Path.cwd().parent, *Path.cwd().parents]:
    if (p / "code" / "donnees" / "pipeline_mgb2.py").exists():
        if str(p / "code" / "donnees") not in sys.path:
            sys.path.insert(0, str(p / "code" / "donnees"))
        break

from pipeline_mgb2 import executer_d1_selection
from seuils import charger_seuils

seuils = charger_seuils()
print("[INFO] Seuils officiels chargés depuis seuils.yaml.")

# Configuration de l'exécution
MODE_REEL = False               # False: simulation hors ligne | True: exécution réelle MGB-2 + Whisper
MAX_SEGMENTS = 20               # 20 pour essai réel, jusqu'à 50 000 en production Kaggle
MODELE_ASR = "openai/whisper-tiny"  # Whisper tiny pour essai local, large-v3 sur GPU Kaggle
REPARTITION = "alterne"         # 'alterne' pour essai (alimente D et A) | 'quota' pour grand volume

print(f"[D1 INTERFACE] Mode réel : {MODE_REEL} | Max segments : {MAX_SEGMENTS} | ASR : {MODELE_ASR}")
"""
        )
    )

    cells.append(
        creer_cellule_code(
            r"""# Exécution de D1 via pipeline_mgb2.py (source unique de vérité)
resultats = executer_d1_selection(
    mode_reel=MODE_REEL,
    max_segments=MAX_SEGMENTS,
    modele_ecoute=MODELE_ASR,
    repartition_mode=REPARTITION,
)

stats = resultats["stats"]
print("\n=== BILAN EXÉCUTION D1 ===")
print(f"  Entrée examinée      : {stats['entree']}")
print(f"  Déjà traités (reprise) : {stats['deja_traites']}")
print(f"  Retenus Jeu D        : {stats['gardes_d']}")
print(f"  Retenus Jeu A        : {stats['gardes_a']}")
print(f"  Rejetés              : {stats['rejets']}")
print(f"  Motifs de rejets     : {stats['rejets_par_motif']}")
print(f"  Durée audio totale   : {stats['duree_audio_totale_s']}s")
print(f"  Temps inférence ASR  : {stats['temps_inference_total_s']:.2f}s")
print(f"  Facteur Temps Réel   : {stats['facteur_temps_reel_rtf']}")
print(f"\nSorties :")
print(f"  - D sélection : {resultats['chemin_d']}")
print(f"  - A alignement: {resultats['chemin_a']}")
print(f"  - Rejets D1   : {resultats['chemin_rejets']}")
print(f"  - Métriques   : {resultats['chemin_stats']}")
"""
        )
    )

    nb.cells = cells
    return nb


# ==============================================================================
# NOTEBOOK D2 : COMPARAISON DES MODÈLES TRADUCTEURS
# ==============================================================================
def generer_nb_d2():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        creer_cellule_md(
            """# Étape D2 — Choix du modèle traducteur

Fine interface pour l'évaluation des 4 modèles candidats officiels.
Génère la grille en aveugle anonymisée (A, B, C, D) et la clé secrète associée.
Logique centralisée dans [`pipeline_mgb2.py`](file:///code/donnees/pipeline_mgb2.py).
"""
        )
    )

    cells.append(
        creer_cellule_code(
            r"""import sys
from pathlib import Path

for p in [Path.cwd(), Path.cwd().parent, *Path.cwd().parents]:
    if (p / "code" / "donnees" / "pipeline_mgb2.py").exists():
        if str(p / "code" / "donnees") not in sys.path:
            sys.path.insert(0, str(p / "code" / "donnees"))
        break

from pipeline_mgb2 import generer_grille_aveugle_d2, SPEC_MODELES_CANDIDATS

print("--- CANDIDATS OFFICIELS ET SPÉCIFICATIONS TECHNIQUES ---")
for nom, spec in SPEC_MODELES_CANDIDATS.items():
    print(f"  - {nom:28s} | {spec['parametres_milliards']}B | {spec['licence']} | Débit estimé: {spec['debit_tokens_sec_estime']} tok/s")

# Génération de la grille aveugle (mode simulation hors ligne)
segments_test = [{"id": f"d-{i:06d}", "ar_ecoute": f"نص تجريبي للاختبار {i}"} for i in range(1, 101)]
candidats = list(SPEC_MODELES_CANDIDATS.keys())

grille, cle = generer_grille_aveugle_d2(segments_test, candidats, graine=42, mode_reel=False)
print(f"\n[OK] Grille aveugle générée ({len(grille)} segments). Clé secrète prête ({len(cle)} fiches).")
"""
        )
    )

    nb.cells = cells
    return nb


# ==============================================================================
# NOTEBOOK D3 : TRADUCTION PAR PAQUETS DE 5 SEGMENTS
# ==============================================================================
def generer_nb_d3():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        creer_cellule_md(
            """# Étape D3 — Traduction par paquets de 5 segments

Interface pour la traduction par paquets avec injection des noms propres.
Logique centralisée dans [`pipeline_mgb2.py`](file:///code/donnees/pipeline_mgb2.py).
"""
        )
    )

    cells.append(
        creer_cellule_code(
            """import sys
from pathlib import Path

for p in [Path.cwd(), Path.cwd().parent, *Path.cwd().parents]:
    if (p / "code" / "donnees" / "pipeline_mgb2.py").exists():
        if str(p / "code" / "donnees") not in sys.path:
            sys.path.insert(0, str(p / "code" / "donnees"))
        break

from pipeline_mgb2 import executer_d3_traduction_paquets

chemin_noms = Path.cwd().parent / "donnees" / "noms" / "noms_v1.csv"
if not chemin_noms.exists():
    chemin_noms = Path("donnees/noms/noms_v1.csv")

segments_entree = [{"id": f"d-{i:06d}", "episode": f"ep_{i//5:02d}", "ar_ecoute": f"نص {i}"} for i in range(1, 26)]
fiches, rejets = executer_d3_traduction_paquets(segments_entree, chemin_noms, mode_reel=False)

print(f"[D3 INTERFACE] Fiches traduites : {len(fiches)} | Rejets paquet : {len(rejets)}")
"""
        )
    )

    nb.cells = cells
    return nb


# ==============================================================================
# NOTEBOOK D4 : FILTRAGE DES TRADUCTIONS ET COMETKIWI
# ==============================================================================
def generer_nb_d4():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        creer_cellule_md(
            """# Étape D4 — Filtrage automatique et COMETKiwi

Interface pour l'application des 7 filtres de nettoyage et du retrait des 20% pires segments COMETKiwi.
Logique centralisée dans [`pipeline_mgb2.py`](file:///code/donnees/pipeline_mgb2.py).
"""
        )
    )

    cells.append(
        creer_cellule_code(
            """import sys
from pathlib import Path

for p in [Path.cwd(), Path.cwd().parent, *Path.cwd().parents]:
    if (p / "code" / "donnees" / "pipeline_mgb2.py").exists():
        if str(p / "code" / "donnees") not in sys.path:
            sys.path.insert(0, str(p / "code" / "donnees"))
        break

from pipeline_mgb2 import executer_d4_filtres

segments_test = [
    {
        "id": f"d-{i:06d}",
        "ar_ecoute": f"قال الوزير رقم {i} إن المؤتمر سيعقد غدا",
        "fr": f"Le ministre numéro {i} a déclaré que la conférence aurait lieu demain.",
    }
    for i in range(1, 31)
]

gardes, rejets = executer_d4_filtres(segments_test, mode_reel=False)
print(f"[D4 INTERFACE] Segments gardés : {len(gardes)} | Rejets filtres : {len(rejets)}")
"""
        )
    )

    nb.cells = cells
    return nb


# ==============================================================================
# NOTEBOOK D5 : TIRAGE DU CONTRÔLE HUMAIN (200 SEGMENTS)
# ==============================================================================
def generer_nb_d5():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        creer_cellule_md(
            """# Étape D5 — Contrôle qualité humain (200 segments)

Tirage au sort déterministe de 200 segments avec graine 42 pour double notation humaine indépendante.
Règle d'acceptation : moins de 5% d'erreurs de sens.
"""
        )
    )

    cells.append(
        creer_cellule_code(
            """import sys
import random
from pathlib import Path

for p in [Path.cwd(), Path.cwd().parent, *Path.cwd().parents]:
    if (p / "code" / "donnees" / "seuils.py").exists():
        if str(p / "code" / "donnees") not in sys.path:
            sys.path.insert(0, str(p / "code" / "donnees"))
        break

from seuils import charger_seuils
seuils = charger_seuils()
NB_CONTROLE = seuils["constantes"]["segments_controle_humain"]  # 200
GRAINE = 42

rng = random.Random(GRAINE)
segments_dispo = [f"d-{i:06d}" for i in range(1, 1000)]
tirage = rng.sample(segments_dispo, NB_CONTROLE)

print(f"[D5 INTERFACE] {len(tirage)} segments tirés au sort pour double relecture qualité.")
"""
        )
    )

    nb.cells = cells
    return nb


# ==============================================================================
# NOTEBOOK D6 : DÉBUTS DE PHRASE POUR LE DIRECT
# ==============================================================================
def generer_nb_d6():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        creer_cellule_md(
            """# Étape D6 — Débuts de phrase pour le direct

Découpe à 30%, 50% et 70% pour l'apprentissage du direct et échantillon de 100 pour relecture humaine.
"""
        )
    )

    cells.append(
        creer_cellule_code(
            """import sys
from pathlib import Path

for p in [Path.cwd(), Path.cwd().parent, *Path.cwd().parents]:
    if (p / "code" / "donnees" / "seuils.py").exists():
        if str(p / "code" / "donnees") not in sys.path:
            sys.path.insert(0, str(p / "code" / "donnees"))
        break

from seuils import charger_seuils
seuils = charger_seuils()
COUPES = seuils["constantes"]["coupe_debuts_pct"]  # [30, 50, 70]
NB_RELECTURE = seuils["constantes"]["debuts_relecture"]  # 100

print(f"[D6 INTERFACE] Paliers de coupe : {COUPES} | Échantillon relecture : {NB_RELECTURE}")
"""
        )
    )

    nb.cells = cells
    return nb


def generer_tous_les_notebooks():
    nbf.write(generer_nb_d1(), str(DOSSIER_NOTEBOOKS / "d1-selection-segments.ipynb"))
    nbf.write(generer_nb_d2(), str(DOSSIER_NOTEBOOKS / "d2-comparaison-modeles.ipynb"))
    nbf.write(generer_nb_d3(), str(DOSSIER_NOTEBOOKS / "d3-traduction-paquets.ipynb"))
    nbf.write(generer_nb_d4(), str(DOSSIER_NOTEBOOKS / "d4-filtres.ipynb"))
    nbf.write(generer_nb_d5(), str(DOSSIER_NOTEBOOKS / "d5-tirage-controle.ipynb"))
    nbf.write(generer_nb_d6(), str(DOSSIER_NOTEBOOKS / "d6-debuts-de-phrase.ipynb"))
    print(f"[OK] 6 notebooks générés avec succès dans {DOSSIER_NOTEBOOKS}")


if __name__ == "__main__":
    generer_tous_les_notebooks()
