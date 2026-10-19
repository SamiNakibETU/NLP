"""Interface de démonstration pour le dépôt de fichier audio et l'export SRT (Étape 3).

Utilise Gradio si installé (`import gradio as gr`), ou démarre un serveur web FastAPI
avec une interface HTML complète permettant de déposer un fichier audio/vidéo et de
télécharger le fichier `.srt` généré par le système S0.
"""

import argparse
from pathlib import Path
import sys
import tempfile
from typing import Optional, Sequence

DOSSIER_DEMO = Path(__file__).resolve().parent
if str(DOSSIER_DEMO) not in sys.path:
    sys.path.insert(0, str(DOSSIER_DEMO))

from systeme_s0 import SystemeS0, charger_config_s0  # noqa: E402

try:
    import gradio as gr
except ImportError:
    gr = None


def traiter_depot_fichier(
    chemin_audio: str,
    mode_substitution: bool = False,
):
    """Fonction appelée par l'interface lorsqu'un utilisateur dépose un fichier audio."""
    if not chemin_audio:
        raise ValueError("Aucun fichier audio fourni.")
    s0 = SystemeS0(config=charger_config_s0(), mode_substitution=mode_substitution)
    dossier_tmp = Path(tempfile.mkdtemp(prefix="s0_demo_"))
    chemin_srt = dossier_tmp / (Path(chemin_audio).stem + ".srt")
    res = s0.sous_titrer_fichier(chemin_audio, chemin_srt_sortie=chemin_srt)
    resume = (
        f"Système : {res['systeme']} | Écoute : {res['modele_ecoute']} | "
        f"Traduction : {res['modele_traduction']} | "
        f"Durée audio : {res['duree_audio_s']} s | Temps calcul : {res['temps_calcul_s']} s (RTF = {res['rtf']})"
    )
    return res["texte_ar_complet"], res["texte_fr_complet"], res["srt"], str(chemin_srt), resume


def construire_interface_gradio(mode_substitution: bool = False):
    """Construit l'objet Blocks Gradio si la bibliothèque `gradio` est installée."""
    if gr is None:
        return None

    with gr.Blocks(title="Sous-titrage français de JT arabes — Démo S0") as demo:
        gr.Markdown(
            "# Sous-titrage français de journaux télévisés arabes (Mode différé — S0)\n"
            "Déposez un extrait audio en arabe pour obtenir la transcription arabe, "
            "la traduction française et le fichier de sous-titres `.srt`."
        )
        with gr.Row():
            entree_audio = gr.Audio(type="filepath", label="Fichier audio arabe (.wav, .flac, .mp3)")
            case_substitution = gr.Checkbox(
                value=mode_substitution,
                label="Mode essai hors ligne (sans téléchargement de modèle)",
            )
        bouton = gr.Button("Générer les sous-titres français (.srt)", variant="primary")
        sortie_resume = gr.Textbox(label="Informations d'exécution", interactive=False)
        with gr.Row():
            sortie_ar = gr.Textbox(label="Transcription arabe (écoute)", rtl=True, lines=4)
            sortie_fr = gr.Textbox(label="Traduction française", lines=4)
        sortie_srt_texte = gr.Code(label="Aperçu du fichier .srt", language=None, lines=10)
        sortie_srt_fichier = gr.File(label="Télécharger le fichier .srt")

        bouton.click(
            fn=traiter_depot_fichier,
            inputs=[entree_audio, case_substitution],
            outputs=[sortie_ar, sortie_fr, sortie_srt_texte, sortie_srt_fichier, sortie_resume],
        )
    return demo


def construire_application_fastapi(mode_substitution: bool = False):
    """Construit une application FastAPI de dépôt de fichier (repli si Gradio est absent)."""
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    app = FastAPI(title="Démo S0 — Sous-titrage français de JT arabes")

    page_html = """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Sous-titrage JT Arabe -> Français (S0)</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 820px; margin: 2rem auto; padding: 0 1rem; background: #0f172a; color: #f8fafc; }
    .card { background: #1e293b; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; }
    button { background: #3b82f6; color: white; border: none; padding: 0.7rem 1.2rem; border-radius: 6px; cursor: pointer; font-weight: 600; }
    pre { background: #020617; padding: 1rem; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; }
    .ar { direction: rtl; font-size: 1.15rem; }
  </style>
</head>
<body>
  <h1>Sous-titrage français de JT arabes — Démo S0</h1>
  <div class="card">
    <form id="form-upload" enctype="multipart/form-data">
      <p><label>Fichier audio (.wav, .flac) : <input type="file" name="fichier" accept=".wav,.flac,.ogg" required></label></p>
      <p><label><input type="checkbox" name="mode_substitution" value="true"> Mode essai hors ligne</label></p>
      <button type="submit">Traduire et exporter en .SRT</button>
    </form>
  </div>
  <div id="resultat" class="card" style="display:none;">
    <h3>Résultat</h3>
    <p id="infos"></p>
    <h4>Transcription arabe</h4>
    <pre id="texte_ar" class="ar"></pre>
    <h4>Sous-titres français (.srt)</h4>
    <pre id="srt"></pre>
  </div>
  <script>
    document.getElementById('form-upload').onsubmit = async (e) => {
      e.preventDefault();
      const fd = new FormData(e.target);
      const r = await fetch('/api/sous-titrer', { method: 'POST', body: fd });
      const data = await r.json();
      document.getElementById('resultat').style.display = 'block';
      document.getElementById('infos').textContent = `Durée: ${data.duree_audio_s}s | Calcul: ${data.temps_calcul_s}s | RTF: ${data.rtf}`;
      document.getElementById('texte_ar').textContent = data.texte_ar_complet;
      document.getElementById('srt').textContent = data.srt;
    };
  </script>
</body>
</html>"""

    @app.get("/", response_class=HTMLResponse)
    def accueil():
        return page_html

    @app.post("/api/sous-titrer")
    async def api_sous_titrer(
        fichier: UploadFile = File(...),
        mode_substitution_form: Optional[str] = Form(default=None, alias="mode_substitution"),
    ):
        suffixe = Path(fichier.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffixe) as tmp:
            contenu = await fichier.read()
            tmp.write(contenu)
            chemin_tmp = tmp.name

        utiliser_sub = mode_substitution or (mode_substitution_form == "true")
        s0 = SystemeS0(config=charger_config_s0(), mode_substitution=utiliser_sub)
        res = s0.sous_titrer_fichier(chemin_tmp)
        return JSONResponse(res)

    return app


def construire_parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lance l'interface web de dépôt de fichier pour le système S0 (Gradio ou FastAPI)."
    )
    parser.add_argument("--essai", action="store_true", help="Vérifie le fonctionnement de l'interface sur un extrait factice.")
    parser.add_argument("--port", type=int, default=7860, help="Port d'écoute du serveur web (défaut: 7860).")
    parser.add_argument("--hors-ligne", action="store_true", help="Active le mode substitution hors ligne par défaut.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = construire_parseur()
    args = parser.parse_args(argv)

    if args.essai:
        from systeme_s0 import executer_mode_essai_s0
        res_s0 = executer_mode_essai_s0()
        dossier_tmp = Path(res_s0["chemin_srt"]).parent
        chemin_wav = dossier_tmp / "extrait_essai_ar.wav"
        ar, fr, srt_txt, srt_path, resume = traiter_depot_fichier(str(chemin_wav), mode_substitution=True)
        print(f"[ESSAI DEMO OK] {resume} | SRT={srt_path} ({len(srt_txt)} caractères)")
        return 0

    if gr is not None:
        demo = construire_interface_gradio(mode_substitution=args.hors_ligne)
        demo.launch(server_port=args.port)
    else:
        import uvicorn
        app = construire_application_fastapi(mode_substitution=args.hors_ligne)
        uvicorn.run(app, host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
