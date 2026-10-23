"""Interface simple : on dépose une vidéo ou un audio, on choisit le moteur, on récupère les sous-titres,
puis le rapport de session (qui parle, traduction relue, résumé daté).

Aucun réseau, aucun flux de presse, aucun mode « édition figée » : seulement les moteurs chargés
(`engines` : nom -> Engine). Lancée par api.py ; l'ancienne interface complète reste dans app.py (PORTEPAROLE_FRONT=old).
Chaque fichier traité devient une Session (sessions.py), comme une écoute en direct : le même rapport s'y applique.
Chaque appel de modèle passe sous le verrou GPU global (un seul groupe de modèles en VRAM à la fois).
"""
from __future__ import annotations

import html
import shutil
import tempfile
import time
from pathlib import Path

import gradio as gr

import media
import session_report
import tts
from engines import write_subtitles
from sessions import Session

TASKS = [("Traduire en français", "st"), ("Transcrire en arabe", "asr")]
CSS = """
.gradio-container {max-width: 1100px !important; margin: auto}
.arabic textarea {direction: rtl; font-size: 1.1rem}
.warn {border-left: 4px solid #b45309; padding: .6rem 1rem; background: #fef3c7; color: #451a03; border-radius: 6px}
.live-link {display: inline-block; padding: .55rem 1rem; border-radius: 8px; background: #047857; color: #fff !important;
  font-weight: 600; text-decoration: none; transition-property: background-color, scale; transition-duration: .15s}
.live-link:hover {background: #065f46}
.live-link:active {scale: .96}
.live-link:focus-visible {outline: 3px solid #064e3b; outline-offset: 2px}
@media (prefers-reduced-motion: reduce) {.live-link {transition: none}}
"""
THEME = gr.themes.Soft(primary_hue=gr.themes.colors.emerald, neutral_hue=gr.themes.colors.slate)
AGE_MAX_TEMP_S = 3600                                  # dossiers temporaires gardés une heure
MSG_COMPARER = ("Chaque moteur est chargé puis déchargé à tour de rôle (trois groupes de modèles). "
                "Compte plusieurs minutes sur un T4, et le premier clic télécharge les modèles.")


def launch_kwargs() -> dict:
    return dict(theme=THEME, css=CSS)


def purger_temporaires(age_max_s: float = AGE_MAX_TEMP_S) -> int:
    """Supprime les dossiers `porteparole_*` du répertoire temporaire plus vieux que `age_max_s`. Renvoie leur nombre.

    Ne touche ni aux sessions (speech_app/sessions/) ni à JOBS (api.py) : ils sont hors du périmètre de ce fichier.
    """
    n, limite = 0, time.time() - age_max_s
    for d in Path(tempfile.gettempdir()).glob("porteparole_*"):
        try:
            if d.is_dir() and d.stat().st_mtime < limite:
                shutil.rmtree(d, ignore_errors=True)
                n += 1
        except OSError:
            pass
    return n


def _stats(res) -> str:
    bits = [f"**{res.label}**", f"audio {res.duration:.0f} s"]
    if res.seconds is not None:
        bits.append(f"traité en {res.seconds:.0f} s (x{res.seconds / max(res.duration, 1e-6):.2f} du temps réel)")
    if res.peak_mem_gb:
        bits.append(f"pic mémoire GPU {res.peak_mem_gb:.1f} Go")
    return " · ".join(bits)


def creer_session(res, wav: str, src: str) -> Session:
    """Le fichier traité devient une Session (audio + segments), comme une écoute en direct."""
    session = Session.creer(source="fichier", moteur=res.engine, titre=Path(src).stem)
    session.ecrire_audio(session_report.lire_audio(Path(wav)))
    for i, seg in enumerate(res.segments):
        session.ajouter_segment({"id": i, "debut_s": seg.start, "fin_s": seg.end, "ar": seg.ar, "fr": seg.fr})
    return session


def _resume_rapport(rapport: dict) -> str:
    lex = rapport["lexique"]
    lignes = [f"**{rapport['n_locuteurs']} locuteur(s)**, identifiés par : {rapport['methode_diarisation']}.",
              f"Relecture : {rapport['methode_revision']}. Résumé : {rapport['methode_resume']}. "
              f"Calcul : {rapport['duree_calcul_s']:.0f} s."]
    if lex["n_entites"]:
        lignes.append(f"Noms propres du lexique repérés : {', '.join(lex['entites_detectees'][:12])}.")
    if rapport["avertissements"]:
        lignes.append("À vérifier :\n" + "\n".join(f"- {a}" for a in rapport["avertissements"]))
    return "\n\n".join(lignes)


def build_front(engines: dict, real: bool = True) -> gr.Blocks:
    choices = [(e.label, name) for name, e in engines.items()]
    blurbs = "\n".join(f"- **{e.label}** : {e.blurb}" for e in engines.values())
    pool = next((e.pool for e in engines.values() if hasattr(e, "pool")), None)

    def prepare(file):
        if not file:
            raise gr.Error("Dépose d'abord une vidéo (mp4…) ou un audio (wav, mp3…).")
        purger_temporaires()
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
            with session_report.verrou_gpu():            # un seul modèle en VRAM à la fois
                res = engines[engine].run(wav, task, progress=progress)
        except Exception as exc:                          # noqa: BLE001 : afficher la cause à l'utilisateur
            raise gr.Error(f"{type(exc).__name__}: {exc}")
        files = write_subtitles(res, folder)             # [.srt, .vtt]
        try:
            sid = creer_session(res, wav, src).sid
            etat = "Fichier traité. Ouvre l'onglet « Rapport de session » pour générer le rapport."
        except Exception as exc:                          # noqa: BLE001 : les sous-titres restent utilisables
            sid, etat = None, f"Le rapport n'est pas disponible pour ce fichier ({type(exc).__name__}: {exc})."
        # gradio 6.30 ne lit les sous-titres qu'à la création du composant : on renvoie un gr.Video complet
        video = (gr.Video(value=src, subtitles=files[1], label="Aperçu avec sous-titres", interactive=False, visible=True)
                 if media.is_video(src) else gr.update(value=None, visible=False))
        warn = f'<div class="warn">{html.escape(res.warning)}</div>' if res.warning else ""
        return (video, res.main_text, files, _stats(res), warn, sid, etat)

    def run_all(file, task, progress=gr.Progress()):
        folder, _src, wav = prepare(file)
        rows = []
        for i, (name, eng) in enumerate(engines.items()):
            progress(i / len(engines), desc=f"{eng.label}…")
            try:
                with session_report.verrou_gpu():
                    res = eng.run(wav, task)
                text = res.main_text.replace("|", "/").replace("\n", " ")
                rows.append(f"| {html.escape(eng.label)} | {res.seconds or 0:.0f} s | {html.escape(text)} |")
            except Exception as exc:                      # noqa: BLE001 : un moteur en échec ne bloque pas les autres
                rows.append(f"| {html.escape(eng.label)} | - | ERREUR : {html.escape(type(exc).__name__ + ': ' + str(exc))[:300]} |")
        shutil.rmtree(folder, ignore_errors=True)
        return "| Moteur | Temps | Résultat |\n|---|---|---|\n" + "\n".join(rows)

    def faire_rapport(sid, locuteurs, relire, arabe, progress=gr.Progress()):
        if not sid:
            raise gr.Error("Traduis d'abord un fichier : le rapport s'appuie sur son audio et ses segments.")
        options = {"diarisation": "auto" if locuteurs else "off", "revision": bool(relire),
                   "inclure_arabe": bool(arabe), "pool": pool,
                   "progress": lambda f, d="": progress(f, desc=d)}
        try:
            rapport = session_report.generer_rapport(Session.ouvrir(sid), options)   # prend le verrou GPU lui-même
        except Exception as exc:                          # noqa: BLE001
            raise gr.Error(f"Le rapport n'a pas pu être généré : {type(exc).__name__}: {exc}")
        dossier = Session.ouvrir(sid).dossier_rapport
        transcript = (dossier / "transcript.md").read_text(encoding="utf-8")
        resume = (dossier / "resume.md").read_text(encoding="utf-8")
        fichiers = [str(dossier / n) for n in rapport["fichiers"]]
        return transcript, resume, fichiers, _resume_rapport(rapport)

    def lire_resume(sid):
        chemin = Session.ouvrir(sid).dossier_rapport / "resume.md" if sid else None
        if not chemin or not chemin.is_file():
            raise gr.Error("Génère d'abord le rapport.")
        try:
            audio, _mime = tts.synthese(chemin.read_text(encoding="utf-8"))
        except tts.TTSIndisponible as exc:
            raise gr.Error(f"Lecture vocale indisponible : {exc}")
        return str(audio)

    with gr.Blocks(title="Traduction de parole arabe") as demo:
        gr.Markdown("# Sous-titres français pour la parole arabe\n"
                    "Dépose une vidéo ou un audio, choisis un moteur, récupère les sous-titres (.srt, .vtt), "
                    "puis un rapport avec qui parle et un résumé.")
        gr.HTML('<a class="live-link" href="/live">Écouter en direct (micro ou flux)</a>')
        if not real:
            gr.Markdown("> **Mode démonstration** : aucun modèle n'est chargé, les textes sont des exemples fixes.")
        sid = gr.State(None)
        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                file = gr.File(label="Vidéo ou audio", file_types=["video", "audio"], type="filepath")
                engine = gr.Radio(choices, value=choices[0][1] if choices else None, label="Moteur")
                task = gr.Radio(TASKS, value="st", label="Tâche")
                go = gr.Button("Traduire", variant="primary")
                go_all = gr.Button("Comparer tous les moteurs")
                gr.Markdown(f"<small>{MSG_COMPARER}</small>")
                with gr.Accordion("Les moteurs", open=False):
                    gr.Markdown(blurbs)
            with gr.Column(scale=6):
                with gr.Tabs():
                    with gr.Tab("Sous-titres"):
                        warn = gr.HTML()
                        video = gr.Video(label="Aperçu avec sous-titres", visible=False, interactive=False)
                        text = gr.Textbox(label="Résultat", lines=8, buttons=["copy"], elem_classes="arabic")
                        stats = gr.Markdown()
                        files = gr.File(label="Sous-titres à télécharger", file_count="multiple", interactive=False)
                        table = gr.Markdown()
                    with gr.Tab("Rapport de session"):
                        gr.Markdown("Le rapport indique qui parle, relit la traduction et résume le contenu. "
                                    "Chaque point du résumé porte l'horodatage du passage source. "
                                    "Il utilise le GPU : compte quelques minutes.")
                        with gr.Row():
                            opt_loc = gr.Checkbox(True, label="Identifier les locuteurs")
                            opt_rel = gr.Checkbox(True, label="Relire la traduction avec un modèle de langage")
                            opt_ar = gr.Checkbox(False, label="Joindre le texte arabe")
                        go_rapport = gr.Button("Générer le rapport (locuteurs + résumé)", variant="primary")
                        etat_rapport = gr.Markdown()
                        resume_md = gr.Markdown(container=True)
                        voix = gr.Button("Lire le résumé à voix haute")
                        audio = gr.Audio(label="Lecture du résumé", type="filepath", interactive=False)
                        transcript_md = gr.Markdown(container=True)
                        fichiers_rapport = gr.File(label="Rapport à télécharger", file_count="multiple", interactive=False)
        go.click(run_one, [file, engine, task], [video, text, files, stats, warn, sid, etat_rapport])
        go_all.click(run_all, [file, task], [table])
        go_rapport.click(faire_rapport, [sid, opt_loc, opt_rel, opt_ar],
                         [transcript_md, resume_md, fichiers_rapport, etat_rapport])
        voix.click(lire_resume, [sid], [audio])
    return demo
