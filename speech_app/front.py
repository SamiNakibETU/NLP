"""Interface simple : on dépose une vidéo ou un audio, on choisit le moteur, on récupère les sous-titres.

Aucun réseau, aucun flux de presse, aucun mode « édition figée » : seulement les moteurs chargés
(`engines` : nom -> Engine). Lancée par api.py ; l'ancienne interface complète reste dans app.py (PORTEPAROLE_FRONT=old).
"""
from __future__ import annotations

import html
import shutil
import tempfile
from pathlib import Path

import gradio as gr

import media
from engines import write_subtitles

TASKS = [("Traduire en français", "st"), ("Transcrire en arabe", "asr")]
CSS = """
.gradio-container {max-width: 1100px !important; margin: auto}
.arabic textarea {direction: rtl; font-size: 1.1rem}
.warn {border-left: 4px solid #b45309; padding: .6rem 1rem; background: #fef3c7; color: #451a03; border-radius: 6px}
"""
THEME = gr.themes.Soft(primary_hue=gr.themes.colors.emerald, neutral_hue=gr.themes.colors.slate)


def launch_kwargs() -> dict:
    return dict(theme=THEME, css=CSS)


def _stats(res) -> str:
    bits = [f"**{res.label}**", f"audio {res.duration:.0f} s"]
    if res.seconds is not None:
        bits.append(f"traité en {res.seconds:.0f} s (x{res.seconds / max(res.duration, 1e-6):.2f} du temps réel)")
    if res.peak_mem_gb:
        bits.append(f"pic mémoire GPU {res.peak_mem_gb:.1f} Go")
    return " · ".join(bits)


def build_front(engines: dict, real: bool = True) -> gr.Blocks:
    choices = [(e.label, name) for name, e in engines.items()]
    blurbs = "\n".join(f"- **{e.label}** : {e.blurb}" for e in engines.values())

    def prepare(file):
        if not file:
            raise gr.Error("Dépose d'abord une vidéo (mp4…) ou un audio (wav, mp3…).")
        folder = tempfile.mkdtemp(prefix="porteparole_")
        src = str(Path(folder) / Path(file).name)
        shutil.copy(file, src)
        try:
            return folder, src, media.to_wav(src, folder)
        except media.MediaError as exc:
            shutil.rmtree(folder, ignore_errors=True)
            raise gr.Error(str(exc))

    def run_one(file, engine, task, progress=gr.Progress()):
        folder, src, wav = prepare(file)
        try:
            res = engines[engine].run(wav, task, progress=progress)
        except Exception as exc:                          # noqa: BLE001 : afficher la cause à l'utilisateur
            raise gr.Error(f"{type(exc).__name__}: {exc}")
        files = write_subtitles(res, folder)             # [.srt, .vtt]
        # gradio 6.30 ne lit les sous-titres qu'à la création du composant : on renvoie un gr.Video complet
        video = (gr.Video(value=src, subtitles=files[1], label="Aperçu avec sous-titres", interactive=False, visible=True)
                 if media.is_video(src) else gr.update(value=None, visible=False))
        warn = f'<div class="warn">{html.escape(res.warning)}</div>' if res.warning else ""
        return (video, res.main_text, files, _stats(res), warn)

    def run_all(file, task, progress=gr.Progress()):
        folder, _src, wav = prepare(file)
        rows = []
        for i, (name, eng) in enumerate(engines.items()):
            progress(i / len(engines), desc=f"{eng.label}…")
            try:
                res = eng.run(wav, task)
                text = res.main_text.replace("|", "/").replace("\n", " ")
                rows.append(f"| {html.escape(eng.label)} | {res.seconds or 0:.0f} s | {html.escape(text)} |")
            except Exception as exc:                      # noqa: BLE001 : un moteur en échec ne bloque pas les autres
                rows.append(f"| {html.escape(eng.label)} | - | ERREUR : {html.escape(type(exc).__name__ + ': ' + str(exc))[:300]} |")
        shutil.rmtree(folder, ignore_errors=True)
        return "| Moteur | Temps | Résultat |\n|---|---|---|\n" + "\n".join(rows)

    with gr.Blocks(title="Traduction de parole arabe") as demo:
        gr.Markdown("# Sous-titres français pour la parole arabe\n"
                    "Dépose une vidéo ou un audio, choisis un moteur, récupère les sous-titres (.srt, .vtt).")
        if not real:
            gr.Markdown("> **Mode démonstration** : aucun modèle n'est chargé, les textes sont des exemples fixes.")
        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                file = gr.File(label="Vidéo ou audio", file_types=["video", "audio"], type="filepath")
                engine = gr.Radio(choices, value=choices[0][1] if choices else None, label="Moteur")
                task = gr.Radio(TASKS, value="st", label="Tâche")
                go = gr.Button("Traduire", variant="primary")
                go_all = gr.Button("Comparer tous les moteurs")
                with gr.Accordion("Les moteurs", open=False):
                    gr.Markdown(blurbs)
            with gr.Column(scale=6):
                warn = gr.HTML()
                video = gr.Video(label="Aperçu avec sous-titres", visible=False, interactive=False)
                text = gr.Textbox(label="Résultat", lines=8, buttons=["copy"], elem_classes="arabic")
                stats = gr.Markdown()
                files = gr.File(label="Sous-titres à télécharger", file_count="multiple", interactive=False)
                table = gr.Markdown()
        go.click(run_one, [file, engine, task], [video, text, files, stats, warn])
        go_all.click(run_all, [file, task], [table])
    return demo
