"""Le Porte-Parole : presse arabe en direct, traduite en français, et vidéos sous-titrées par le modèle.

Lancer :  python app.py            (puis ouvrir http://127.0.0.1:7860)

Variables d'environnement :
  PORTEPAROLE_S3_REPO     racine du dépôt de l'équipe (branche benchmarking) : charge le système S3
  PORTEPAROLE_S3_CKPT     projecteur S3 à utiliser (défaut : modeles/s3_temps1_50k/projecteur_meilleur.pt du dépôt)
  HF_TOKEN                jeton Hugging Face (encodeur Cohere à accès contrôlé)
  PORTEPAROLE_CODE        dossier du code du modèle (speech_xattn), pour charger les poids entraînés
  PORTEPAROLE_CKPT_PROJ   checkpoint du projecteur (last.pt)
  PORTEPAROLE_CKPT_XATTN  checkpoint de la cross-attention (last.pt)
  PORTEPAROLE_MT=off      désactive la traduction NLLB des articles
  SPEECH_LOGS             dossier avec log.jsonl / val.jsonl pour l'onglet Coulisses
Sans checkpoint, les moteurs de démonstration (texte fictif, signalé) servent à mettre au point l'interface ;
la traduction de vidéos en ligne est alors refusée plutôt que simulée.
"""
from __future__ import annotations

import datetime as dt
import os
import tempfile
import threading
import time
from pathlib import Path

import gradio as gr

import clip
import feeds
import plots
import reader
import render
import snapshot as snapmod
import textmt
from engines import build_engines, write_subtitles
from textmt import BackgroundTranslator, default_translator

ROOT = Path(__file__).parent
SNAP = snapmod.Snapshot.load(os.environ["PORTEPAROLE_SNAPSHOT"]) if os.environ.get("PORTEPAROLE_SNAPSHOT") else None
FROZEN = SNAP is not None   # édition figée : aucun réseau, aucun modèle
TRANSLATOR = textmt.Translator(None, None) if FROZEN else default_translator()
BG = BackgroundTranslator(TRANSLATOR)
_last = {}
COUNTRIES = ["Tous"] + [s.country for s in feeds.SOURCES]
TASKS = [("Traduire en français", "st"), ("Transcrire en arabe", "asr")]


def load_engines():
    """Moteurs réels selon les variables d'environnement, sinon moteurs de démonstration.
    PORTEPAROLE_S3_REPO : racine du dépôt de l'équipe (S3) ; PORTEPAROLE_CODE + CKPT_* : mes projecteurs."""
    engines = {}
    if os.environ.get("PORTEPAROLE_ENGINES") or os.environ.get("PORTEPAROLE_NLP_REPO"):
        import lazy_engines            # S0, Seamless, S3 chargés à la demande (un seul en mémoire)
        engines.update(lazy_engines.engines_from_env())
    elif os.environ.get("PORTEPAROLE_S3_REPO"):
        import s3_engine
        engines.update(s3_engine.load_s3_engines(os.environ["PORTEPAROLE_S3_REPO"],
                                                 os.environ.get("PORTEPAROLE_S3_CKPT") or None))
    code = os.environ.get("PORTEPAROLE_CODE")
    if code:
        import model_engine
        for kind, var in (("proj", "PORTEPAROLE_CKPT_PROJ"), ("xattn", "PORTEPAROLE_CKPT_XATTN")):
            if os.environ.get(var):
                engines[kind] = model_engine.load_engine(kind, code, os.environ[var])
    return (engines, True) if engines else (build_engines(demo=True), False)


ENGINES, REAL = ({}, False) if FROZEN else load_engines()
MODE = (f"Édition figée du {SNAP.created:%d/%m/%Y}" if FROZEN else "Modèle entraîné" if REAL else "Démonstration")
CHOICES = [(f"{e.label} ({e.status.lower()})", name) for name, e in ENGINES.items()]
DEFAULT_ENGINE = next(iter(ENGINES), None)

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?family=Aref+Ruqaa:wght@400;700&family=Newsreader:ital,opsz,wght@0,6..72,400..700;1,6..72,400..700'
         '&family=Noto+Naskh+Arabic:wght@400..700&display=swap" rel="stylesheet"><meta name="color-scheme" content="light">')
FORCE_LIGHT = "() => { document.documentElement.classList.remove('dark'); document.body.classList.remove('dark'); }"

PAPER, CARD, INK, TRAD = "#ece9e0", "#f5f3ec", "#1b2340", "#0e5a4f"
THEME = gr.themes.Base(primary_hue=gr.themes.colors.emerald, neutral_hue=gr.themes.colors.stone,
                       font=[gr.themes.GoogleFont("Newsreader"), "Georgia", "serif"]).set(
    body_background_fill=PAPER, body_background_fill_dark=PAPER, body_text_color=INK, body_text_color_dark=INK,
    background_fill_primary=CARD, background_fill_primary_dark=CARD, background_fill_secondary=PAPER,
    background_fill_secondary_dark=PAPER, block_background_fill="transparent", block_background_fill_dark="transparent",
    block_border_color="#b7b2a3", block_border_color_dark="#b7b2a3", block_radius="0px", block_shadow="none",
    border_color_primary="#b7b2a3", border_color_primary_dark="#b7b2a3", input_background_fill=CARD,
    input_background_fill_dark=CARD, input_radius="0px", button_large_radius="0px", button_small_radius="0px",
    button_primary_background_fill=TRAD, button_primary_background_fill_dark=TRAD,
    button_primary_background_fill_hover="#123d36", button_primary_background_fill_hover_dark="#123d36",
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff", color_accent=TRAD, slider_color=TRAD,
    checkbox_background_color_selected=TRAD, checkbox_background_color_selected_dark=TRAD,
)


# un clic sur un titre (data-open) prévient Python, qui charge l'article
PRESS_JS = """
element.addEventListener('click', (e) => {
  const b = e.target.closest('[data-open]');
  if (b) trigger('click', {id: b.dataset.open});
});
"""


# clic sur un horodatage (data-seek) : place le lecteur YouTube à cet instant
SEEK_JS = """
element.addEventListener('click', (e) => {
  const b = e.target.closest('[data-seek]');
  const f = document.querySelector('.embed iframe');
  if (!b || !f) return;
  const send = (func, args) => f.contentWindow.postMessage(JSON.stringify({event: 'command', func, args}), '*');
  send('seekTo', [parseFloat(b.dataset.seek), true]);
  send('playVideo', []);
});
"""


def launch_kwargs() -> dict:
    return dict(theme=THEME, css=(ROOT / "style.css").read_text(encoding="utf-8"), head=FONTS, js=FORCE_LIGHT)


# ---------------------------------------------------------------- presse
def _press_html(country, data):
    pending = BG.pending
    if not TRANSLATOR.enabled:
        how = "traduction automatique indisponible sur cette machine"
    elif pending:
        how = f"traduction en cours, encore {pending} textes"
    else:
        how = "traduit par NLLB-200"
    status = f"Actualisé à {dt.datetime.now():%H:%M}, {how}."
    _last["sig"] = (country, len(TRANSLATOR.cache), pending)
    return render.press(data, TRANSLATOR, None if country == "Tous" else country, status)


def _frozen_press(country):
    data = {sid: (items, None) for sid, items in SNAP.articles_by_source.items()}
    status = f"Édition figée du {SNAP.created:%d/%m à %H:%M}. Les articles ne se mettent plus à jour."
    return render.press(data, SNAP.translator, None if country == "Tous" else country, status)


def on_press(country, force=False):
    """Affiche tout de suite ce qui est en cache ; la traduction des textes manquants se fait en arrière-plan."""
    if FROZEN:
        return _frozen_press(country)
    data = feeds.fetch_all("article", 4, force)
    items = [it for its, _ in data.values() for it in its]
    BG.enqueue([it.title for it in items] + [it.summary for it in items if it.summary])
    return _press_html(country, data)


def on_tick(country):
    """Appelé toutes les 8 s : ne remet la page à jour que si de nouvelles traductions sont arrivées."""
    if FROZEN:
        return gr.skip()
    data = feeds.fetch_all("article", 4, False)  # cache mémoire : pas de requête réseau
    if _last.get("sig") == (country, len(TRANSLATOR.cache), BG.pending):
        return gr.skip()
    return _press_html(country, data)


def on_press_refresh(country):
    return on_press(country, force=True)


REFRESH_EVERY = 600   # secondes entre deux relectures des flux
PREFETCH = 12         # nombre d'articles dont on garde le texte complet en mémoire (le clic est alors instantané)
_started = False


def refresh_once() -> int:
    """Relit les flux, met en file la traduction des titres et résumés, charge le texte complet des premiers articles."""
    data = feeds.fetch_all("article", 4, True)
    items = [it for its, _ in data.values() for it in its]
    BG.enqueue([it.title for it in items] + [it.summary for it in items if it.summary])
    for it in items[:PREFETCH]:
        reader.fetch_article(it)
    return len(items)


def start_background() -> None:
    """Fil de fond lancé au démarrage du serveur : quand quelqu'un ouvre le site, tout est déjà prêt."""
    global _started
    if _started:
        return
    _started = True

    def loop():
        while True:
            try:
                refresh_once()
            except Exception as exc:  # réseau coupé, flux cassé : on réessaie au tour suivant
                print(f"actualisation en arrière-plan impossible : {type(exc).__name__}: {exc}", flush=True)
            time.sleep(REFRESH_EVERY)

    threading.Thread(target=loop, daemon=True, name="porteparole-refresh").start()


def _frozen_article(uid: str) -> str:
    got = SNAP.article(uid)
    if got:
        art, fr_title, fr = got
        return render.reader_page(art, fr_title, fr, "done")
    item = SNAP.find_item(uid)
    if item is None:
        return render.notice("Article introuvable", "Cet article n'est pas dans l'édition figée.")
    summary = [item.summary] if item.summary else [item.title]
    art = reader.Article(item, summary, "résumé", "texte complet non inclus dans cette édition")
    fr = [SNAP.translator.get(summary[0])]
    return render.reader_page(art, SNAP.translator.get(item.title), fr, "done")


def on_open(evt: gr.EventData):
    """Clic sur un article : texte complet d'abord en arabe, puis traduction paragraphe par paragraphe."""
    show = (gr.update(visible=False), gr.update(visible=True))
    if FROZEN:
        yield (*show, _frozen_article(getattr(evt, "id", "") or ""))
        return
    item = feeds.find(getattr(evt, "id", "") or "")
    if item is None:
        yield (*show, render.notice("Article introuvable", "Actualise la page Presse puis réessaie."))
        return
    yield (*show, render.empty_state("Chargement de l'article", "Récupération du texte complet."))
    art = reader.fetch_article(item)
    state = "pending" if TRANSLATOR.enabled else "off"
    fr_title, fr = TRANSLATOR.get(item.title), [None] * len(art.paragraphs)
    yield (*show, render.reader_page(art, fr_title, fr, state))
    if not TRANSLATOR.enabled:
        return
    fr_title = fr_title or TRANSLATOR.translate_paragraph(item.title)
    yield (*show, render.reader_page(art, fr_title, fr, state))
    for i, p in enumerate(art.paragraphs):
        fr[i] = TRANSLATOR.translate_paragraph(p)
        yield (*show, render.reader_page(art, fr_title, fr, "pending" if i + 1 < len(fr) else "done"))


def on_back():
    return gr.update(visible=True), gr.update(visible=False)


# ---------------------------------------------------------------- vidéos
def on_channel(country):
    src = next((s for s in feeds.SOURCES if s.country == country), feeds.SOURCES[0])
    if FROZEN:
        items, err = SNAP.videos_by_source.get(src.id, []), None
        done = SNAP.translated_video_ids()
        gallery = [(i.image, ("[traduite] " if i.video_id in done else "") + i.title) for i in items]
        return gallery, items, None, render.video_original(None), "", ""
    items, err = feeds.fetch(src, "video", 10)
    gallery = [(i.image, i.title) for i in items]
    msg = render.notice("Flux vidéo indisponible", err) if err and not items else ""
    return gallery, items, None, render.video_original(None), "", msg


def on_pick(items, evt: gr.SelectData):
    item = items[evt.index] if items and 0 <= evt.index < len(items) else None
    return item, render.video_original(item.video_id if item else None), render.video_caption(item)


def on_translate(item, engine, seconds, progress=gr.Progress()):
    none = (None, "", None)
    if not item:
        return (*none[:1], render.notice("Aucune vidéo choisie", "Sélectionne d'abord une vignette."), None)
    if FROZEN:
        res = SNAP.video_result(item.video_id)
        if res is None:
            n = len(SNAP.translated_video_ids())
            return (None, render.notice("Vidéo non traduite dans cette édition",
                                        f"Seules {n} vidéos ont été traduites avant que l'édition soit figée : elles portent la mention « traduite »."), None)
        return None, render.timeline(res, seek=True), write_subtitles(res, tempfile.mkdtemp(prefix="porteparole_"))
    if not REAL:
        return (None, render.notice(
            "Aucun modèle entraîné chargé",
            "La traduction de vidéos exige un checkpoint (PORTEPAROLE_CKPT_PROJ ou PORTEPAROLE_CKPT_XATTN). "
            "Sans lui, l'interface ne simule pas de traduction sur de vraies vidéos."), None)
    if (missing := clip.missing_tools()):
        return (None, render.notice("Outils manquants", "Installe : " + ", ".join(missing) + "."), None)
    folder = tempfile.mkdtemp(prefix="porteparole_")
    progress(0.05, desc="Récupération de l'extrait")
    try:
        video, wav = clip.fetch_clip(item.video_id, folder, int(seconds))
    except Exception as exc:
        return (None, render.notice("Extrait indisponible", f"{type(exc).__name__}: {exc}"), None)
    res = ENGINES[engine].run(wav, "st", progress=lambda f, desc="": progress(0.1 + 0.9 * f, desc=desc))
    files = write_subtitles(res, folder)
    return (video, files[1]), render.timeline(res, seek=True), files


# ---------------------------------------------------------------- atelier, comparaison, coulisses
def need_audio(path):
    if not path:
        raise gr.Error("Dépose ou enregistre d'abord un audio.")


def on_atelier(path, engine, task, show_conf, progress=gr.Progress()):
    need_audio(path)
    res = ENGINES[engine].run(path, task, progress=progress)
    files = write_subtitles(res, tempfile.mkdtemp(prefix="porteparole_"))
    return render.article(res, show_conf), render.timeline(res), files, res


def on_conf(res, show_conf):
    return render.article(res, show_conf) if res else render.empty_state("Rien à afficher", "Lance d'abord un traitement.")


def on_compare(path, task, show_conf):
    need_audio(path)
    results = [e.run(path, task) if e.available else None for e in ENGINES.values()]
    return render.compare(results, [e.label for e in ENGINES.values()], show_conf), results


def on_compare_conf(results, show_conf):
    labels = [e.label for e in ENGINES.values()]
    return render.compare(results or [None] * len(labels), labels, show_conf)


def on_logs(folder):
    train, val, demo = plots.load_logs(folder or ".")
    steps = [r["step"] for r in val]
    last = val[-1] if val else {}
    note = ("Aucun journal dans ce dossier : graphique d'illustration." if demo
            else f"{len(train)} pas d'entraînement, {len(val)} validations.")
    return (plots.loss_figure(train, val, demo), gr.update(choices=steps, value=steps[-1] if steps else None),
            render.samples(last.get("samples", []), demo), f'<p class="status">{note}</p>', (train, val, demo))


def on_step(step, state):
    if not state:
        return render.samples([], True)
    _, val, demo = state
    row = next((r for r in val if r["step"] == step), None)
    return render.samples(row.get("samples", []) if row else [], demo)


# ---------------------------------------------------------------- interface
def build_app() -> gr.Blocks:
    with gr.Blocks(title="Le Porte-Parole") as demo:
        gr.HTML(render.masthead(MODE), padding=False)

        with gr.Tabs():
            with gr.Tab("Presse"):
                with gr.Column() as list_col:
                    with gr.Row():
                        country = gr.Radio(COUNTRIES, value="Tous", label="Pays", scale=5)
                        refresh = gr.Button("Actualiser", variant="secondary", scale=1, visible=not FROZEN)
                    press = gr.HTML(render.empty_state("Chargement de la presse", "Les cinq journaux arrivent."),
                                    js_on_load=PRESS_JS, padding=False)
                with gr.Column(visible=False) as reader_col:
                    back = gr.Button("Retour à la presse", variant="secondary")
                    article_view = gr.HTML("", padding=False)
                country.change(on_press, country, press, show_progress="minimal")
                demo.load(on_press, country, press)
                if not FROZEN:
                    refresh.click(on_press_refresh, country, press)
                    gr.Timer(600).tick(on_press_refresh, country, press, show_progress="hidden")
                    gr.Timer(8).tick(on_tick, country, press, show_progress="hidden")
                press.click(on_open, None, [list_col, reader_col, article_view], show_progress="hidden")
                back.click(on_back, None, [list_col, reader_col])

            with gr.Tab("Vidéos"):
                gr.HTML('<p class="status">' + ("Les vidéos marquées « traduite » ont été sous-titrées en français par le modèle. Le clic sur une minute du texte "
                                                 "place la vidéo à cet instant." if FROZEN else
                                                 "Les dernières vidéos des chaînes officielles. Choisis-en une, puis demande sa traduction.") + '</p>', padding=False)
                vsrc = gr.Radio(COUNTRIES[1:], value=COUNTRIES[1], label="Chaîne")
                vmsg = gr.HTML("", padding=False)
                gal = gr.Gallery(columns=5, height=210, object_fit="cover", allow_preview=False, show_label=False)
                items_state, pick_state = gr.State([]), gr.State(None)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=6, min_width=340):
                        gr.HTML('<h2 class="headline" style="font-size:1.3rem!important">Version d\'origine</h2>', padding=False)
                        orig = gr.HTML(render.video_original(None), padding=False)
                        cap = gr.HTML("", padding=False)
                    with gr.Column(scale=6, min_width=340):
                        gr.HTML('<h2 class="headline" style="font-size:1.3rem!important">Version traduite en français</h2>', padding=False)
                        vengine = gr.Dropdown(CHOICES, value=DEFAULT_ENGINE, label="Moteur", visible=not FROZEN)
                        secs = gr.Slider(30, clip.MAX_SECONDS, value=60, step=10, label="Durée traduite, en secondes", visible=not FROZEN)
                        vgo = gr.Button("Afficher la traduction française" if FROZEN else "Traduire cet extrait en français",
                                        variant="primary", elem_classes="go")
                        vout = gr.Video(label="Extrait sous-titré", interactive=False, visible=not FROZEN)
                vtl = gr.HTML("", js_on_load=SEEK_JS, padding=False)
                vfiles = gr.File(label="Sous-titres .srt et .vtt", file_count="multiple", interactive=False)
                vsrc.change(on_channel, vsrc, [gal, items_state, pick_state, orig, cap, vmsg])
                demo.load(on_channel, vsrc, [gal, items_state, pick_state, orig, cap, vmsg])
                gal.select(on_pick, items_state, [pick_state, orig, cap])
                vgo.click(on_translate, [pick_state, vengine, secs], [vout, vtl, vfiles])

            with gr.Tab("Atelier", visible=not FROZEN):
                gr.HTML('<p class="status">Ton propre enregistrement, passé dans le modèle.</p>', padding=False)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=300):
                        audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Enregistrement")
                        engine = gr.Dropdown(CHOICES, value=DEFAULT_ENGINE, label="Moteur")
                        task = gr.Radio(TASKS, value="st", label="Tâche")
                        conf = gr.Checkbox(False, label="Souligner les passages incertains")
                        go = gr.Button("Traduire", variant="primary", elem_classes="go")
                        files = gr.File(label="Sous-titres .srt et .vtt", file_count="multiple", interactive=False)
                    with gr.Column(scale=8, min_width=380):
                        out = gr.HTML(render.empty_state("Aucun enregistrement", "Dépose un fichier ou parle au micro."), padding=False)
                        tl = gr.HTML("", padding=False)
                state = gr.State(None)
                go.click(on_atelier, [audio, engine, task, conf], [out, tl, files, state])
                conf.change(on_conf, [state, conf], out)

            with gr.Tab("Comparer", visible=not FROZEN):
                gr.HTML('<p class="status">Le même enregistrement dans chaque chaîne chargée.</p>', padding=False)
                with gr.Row():
                    audio2 = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Enregistrement", scale=3)
                    with gr.Column(scale=2):
                        task2 = gr.Radio(TASKS, value="st", label="Tâche")
                        conf2 = gr.Checkbox(False, label="Souligner les passages incertains")
                        go2 = gr.Button("Comparer", variant="primary", elem_classes="go")
                out2 = gr.HTML(on_compare_conf(None, False), padding=False)
                state2 = gr.State(None)
                go2.click(on_compare, [audio2, task2, conf2], [out2, state2])
                conf2.change(on_compare_conf, [state2, conf2], out2)

            with gr.Tab("Coulisses", visible=(not FROZEN) or SNAP.logs_dir is not None):
                gr.HTML('<p class="status">Le journal de bord de l\'entraînement, lu dans log.jsonl et val.jsonl.</p>', padding=False)
                with gr.Row():
                    folder = gr.Textbox(str(SNAP.logs_dir) if FROZEN and SNAP.logs_dir else os.environ.get("SPEECH_LOGS", "logs"),
                                        label="Dossier du journal", scale=4, visible=not FROZEN)
                    load = gr.Button("Ouvrir le journal", variant="secondary", scale=1, visible=not FROZEN)
                note = gr.HTML("", padding=False)
                plot = gr.Plot(label="Perte", show_label=False)
                step = gr.Dropdown([], label="Étape de validation", interactive=True)
                tries = gr.HTML(render.samples([], False), padding=False)
                logs_state = gr.State(None)
                load.click(on_logs, folder, [plot, step, tries, note, logs_state])
                step.change(on_step, [step, logs_state], tries)
                demo.load(on_logs, folder, [plot, step, tries, note, logs_state])

            with gr.Tab("Colophon"):
                gr.HTML(render.COLOPHON, padding=False)

        gr.HTML('<div class="foot">Le Porte-Parole, projet de recherche, usage non commercial.</div>', padding=False)
    return demo


if __name__ == "__main__":
    if not FROZEN:
        start_background()
    # PORTEPAROLE_SHARE=1 : lien public temporaire (Colab) ; sinon http://127.0.0.1:7860
    build_app().launch(share=os.environ.get("PORTEPAROLE_SHARE") == "1", **launch_kwargs())
