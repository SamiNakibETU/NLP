"""Construction du HTML de l'interface. Tout texte venant d'un flux ou d'un modèle est échappé ; les liens sont filtrés."""
from __future__ import annotations

import datetime as dt
from html import escape
from urllib.parse import urlparse

from engines import Result
from feeds import BY_ID, SOURCES, Item, ago

APP_NAME = "Le Porte-Parole"
TAGLINE = "La presse arabe du jour, lue en français"
MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
          "novembre", "décembre"]
DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
TASK_LABEL = {"st": "Traduction vers le français", "asr": "Transcription en arabe"}


def long_date(d: dt.date | None = None) -> str:
    d = d or dt.date.today()
    return f"{DAYS[d.weekday()]} {d.day} {MONTHS[d.month - 1]} {d.year}"


def fmt_clock(t: float) -> str:
    t = int(round(t))
    return f"{t // 60:02d}:{t % 60:02d}"


def safe_url(url: str | None) -> str:
    """Seuls http et https sont acceptés (les flux sont des données non fiables)."""
    try:
        u = urlparse(url or "")
    except ValueError:
        return "#"
    return url if u.scheme in ("http", "https") and u.netloc else "#"


def masthead(mode_label: str) -> str:
    return f"""
<header class="masthead">
  <div class="dateline"><span>{escape(long_date())}</span><span>Édition expérimentale</span><span>{escape(mode_label)}</span></div>
  <div class="title"><h1>{escape(APP_NAME)}</h1><span class="title-ar" lang="ar" dir="rtl">الناطق</span></div>
  <p class="tagline">{escape(TAGLINE)}</p>
</header>"""


def stamp(text: str = "Texte fictif") -> str:
    return f'<div class="stamp" role="note">{escape(text)}</div>'


def empty_state(title: str, hint: str) -> str:
    ghost = "".join(f'<i style="width:{w}%"></i>' for w in (92, 88, 95, 70, 90, 84, 60))
    return (f'<section class="paper empty"><h2 class="headline">{escape(title)}</h2>'
            f'<p class="byline">{escape(hint)}</p><div class="ghost">{ghost}</div></section>')


# ---------------------------------------------------------------- presse
def open_btn(item: Item, text: str, cls: str = "open") -> str:
    return f'<button type="button" class="{cls}" data-open="{escape(item.uid)}">{escape(text)}</button>'


def story(item: Item, fr_title: str | None, fr_summary: str | None, lead: bool = False, translating: bool = False) -> str:
    def fr_col() -> str:
        if fr_title is None:
            msg = "Traduction en cours" if translating else "Traduction non disponible."
            return f'<p class="pending">{msg}</p>'
        s = f"<p>{escape(fr_summary)}</p>" if fr_summary else ""
        return f"<h3>{open_btn(item, fr_title)}</h3>{s}"

    summary = f"<p>{escape(item.summary)}</p>" if item.summary else ""
    return f"""<article class="story{' lead' if lead else ''}">
  <div class="fr" lang="fr">{fr_col()}</div>
  <div class="ar" dir="rtl" lang="ar"><h3>{open_btn(item, item.title)}</h3>{summary}</div>
  <footer><span>{escape(ago(item.published))}</span>{open_btn(item, "Lire l'article complet", "open-link")}
    <a href="{escape(safe_url(item.link))}" target="_blank" rel="noopener noreferrer">Lire dans le journal</a></footer>
</article>"""


def press(data: dict[str, tuple[list[Item], str | None]], tr, only: str | None, status: str) -> str:
    """data : source -> (entrées, erreur) ; tr : Translator (get(texte) -> français ou None)."""
    blocks, first = [], True
    for src in SOURCES:
        if only and src.country != only:
            continue
        items, err = data.get(src.id, ([], None))
        rows = []
        for it in items:
            rows.append(story(it, tr.get(it.title), tr.get(it.summary) if it.summary else None, lead=first and only is None,
                              translating=tr.enabled))
            first = False
        note = (f'<p class="warn" role="status">Flux indisponible ({escape(err.split(":")[0])}) : '
                f'{"dernière copie affichée." if items else "rien à afficher."}</p>') if err else ""
        blocks.append(f"""<section class="journal">
  <header><div><h2>{escape(src.name_fr)}</h2><span>{escape(src.country)}</span></div>
    <div class="ar-name" lang="ar" dir="rtl">{escape(src.name_ar)}</div></header>
  {note}{"".join(rows) or '<p class="pending">Aucun article.</p>'}
</section>""")
    return f'<p class="status">{escape(status)}</p>{"".join(blocks)}'


def reader_page(art, fr_title: str | None, fr_paras: list[str | None], state: str) -> str:
    """Article complet en deux colonnes, paragraphe par paragraphe. state : "pending", "done" ou "off"."""
    item = art.item
    src = BY_ID[item.source]

    def fr_cell(text: str | None) -> str:
        if text is not None:
            return f"<p>{escape(text)}</p>"
        msg = "Traduction en cours" if state == "pending" else "Traduction non disponible."
        return f'<p class="pending">{msg}</p>'

    origin = {"flux": "Texte repris du flux du journal.", "page": "Texte extrait de la page de l'article.",
              "résumé": "Seul le résumé a pu être récupéré" + (f" ({escape(art.error)})." if art.error else ".")}[art.origin]
    rows = "".join(
        f'<div class="pair para"><div class="fr" lang="fr">{fr_cell(fr_paras[i] if i < len(fr_paras) else None)}</div>'
        f'<div class="ar" dir="rtl" lang="ar"><p>{escape(p)}</p></div></div>' for i, p in enumerate(art.paragraphs))
    head_fr = escape(fr_title) if fr_title else '<span class="pending">Traduction en cours</span>' if state == "pending" else ""
    return f"""<article class="reader">
  <p class="reader-meta">{escape(src.name_fr)}, {escape(src.country)}. {escape(ago(item.published))}</p>
  <div class="pair head"><h2 class="fr" lang="fr">{head_fr}</h2><h2 class="ar" dir="rtl" lang="ar">{escape(item.title)}</h2></div>
  <p class="origin">{origin} Les droits appartiennent au journal.
    <a href="{escape(safe_url(item.link))}" target="_blank" rel="noopener noreferrer">Lire sur le site de {escape(src.name_fr)}</a></p>
  {rows}
</article>"""


# ---------------------------------------------------------------- vidéos
def video_original(video_id: str | None) -> str:
    if not video_id:
        return empty_state("Choisis une vidéo", "Sélectionne une vignette ci-dessus.")
    vid = escape(video_id)
    return (f'<div class="embed"><iframe src="https://www.youtube-nocookie.com/embed/{vid}?enablejsapi=1" title="Vidéo d\'origine" '
            f'loading="lazy" allow="encrypted-media; picture-in-picture" allowfullscreen referrerpolicy="strict-origin-when-cross-origin">'
            f'</iframe></div>')


def video_caption(item: Item | None) -> str:
    if not item:
        return ""
    src = BY_ID[item.source]
    return (f'<div class="vcap"><h3 lang="ar" dir="rtl">{escape(item.title)}</h3>'
            f'<p>{escape(src.name_fr)}, {escape(src.country)}. {escape(ago(item.published))}. '
            f'<a href="{escape(safe_url(item.link))}" target="_blank" rel="noopener noreferrer">Voir sur YouTube</a></p></div>')


def notice(title: str, body: str) -> str:
    return f'<section class="paper notice" role="status"><h3>{escape(title)}</h3><p>{escape(body)}</p></section>'


# ---------------------------------------------------------------- atelier, comparaison, journal d'entraînement
def _words_html(text: str, confidences: list[float] | None, show_conf: bool) -> str:
    words = text.split()
    if not (show_conf and confidences and len(confidences) == len(words)):
        return escape(text)
    out = []
    for w, c in zip(words, confidences):
        cls = "low" if c < 0.6 else "mid" if c < 0.8 else ""
        out.append(f'<span class="w {cls}" title="confiance {c:.0%}">{escape(w)}</span>' if cls else escape(w))
    return " ".join(out)


def _paragraphs(res: Result, show_conf: bool) -> str:
    texts = [(s.ar if res.task == "asr" else s.fr) or "" for s in res.segments]
    conf, i, paras = res.confidences, 0, []
    for k in range(0, len(texts), 2):
        chunk = " ".join(t for t in texts[k:k + 2] if t)
        n = len(chunk.split())
        paras.append(_words_html(chunk, conf[i:i + n] if conf else None, show_conf))
        i += n
    rtl = ' dir="rtl" lang="ar"' if res.task == "asr" else ' lang="fr"'
    return "".join(f"<p{rtl}>{p}</p>" for p in paras if p)


def meta_line(res: Result, with_label: bool = True) -> str:
    bits = ([f"<b>{escape(res.label)}</b>"] if with_label else []) + [f"{fmt_clock(res.duration)} d'audio"]
    if res.audio_tokens:
        bits.append(f"{res.audio_tokens} tokens audio")
    if res.seconds is not None:
        bits.append(f"{res.seconds:.1f} s de calcul")
    if res.peak_mem_gb is not None:
        bits.append(f"{res.peak_mem_gb:.1f} Go de GPU")
    return "".join(f"<span>{b}</span>" for b in bits)


def original_box(res: Result) -> str:
    if res.task == "asr":
        return ""
    if res.produces_arabic and res.ar:
        return f'<aside class="original"><h4>Transcription arabe intermédiaire</h4><p dir="rtl" lang="ar">{escape(res.ar)}</p></aside>'
    return ('<aside class="original"><h4>Texte arabe</h4><p class="pending">Cette chaîne ne passe pas par une transcription '
            "arabe : la traduction part directement de la parole.</p></aside>")


def warn_html(res: Result) -> str:
    return f'<p class="warn" role="alert">{escape(res.warning)}</p>' if getattr(res, "warning", None) else ""


def article(res: Result, show_conf: bool = False) -> str:
    title = res.headline or " ".join(res.main_text.split()[:9])
    return f"""<article class="paper">
  {stamp() if res.is_demo else ""}
  <h2 class="headline">{escape(title)}</h2>
  <p class="byline">{TASK_LABEL[res.task]}{meta_line(res)}</p>
  {warn_html(res)}
  <div class="body cols">{_paragraphs(res, show_conf)}</div>
  {original_box(res)}
</article>"""


def compare(results: list[Result | None], labels: list[str], show_conf: bool = False) -> str:
    cols = []
    for res, label in zip(results, labels):
        if res is None:
            cols.append(f'<section class="col off"><h3>{escape(label)}</h3><p class="pending">Moteur non branché.</p></section>')
            continue
        cols.append(f"""<section class="col">{stamp() if res.is_demo else ""}
  <h3>{escape(res.label)}</h3><p class="byline">{meta_line(res, False)}</p>{warn_html(res)}
  <div class="body">{_paragraphs(res, show_conf)}</div>{original_box(res)}</section>""")
    return f'<div class="compare">{"".join(cols)}</div>'


def timeline(res: Result, seek: bool = False) -> str:
    if not res.segments:
        return empty_state("Aucun sous-titre", "Le moteur n'a rien produit pour cet audio.")
    total = max(res.duration, 1e-6)
    bar = "".join(f'<i style="left:{s.start / total * 100:.2f}%;width:{(s.end - s.start) / total * 100:.2f}%"></i>'
                  for s in res.segments)
    rows = []
    for s in res.segments:
        main = s.ar if res.task == "asr" else s.fr
        side = s.ar if (res.task == "st" and s.ar) else ""
        clock = (f'<button type="button" class="seek" data-seek="{s.start:.2f}">{fmt_clock(s.start)}</button>' if seek
                 else f"<time>{fmt_clock(s.start)}</time>")
        rows.append(f"""<div class="tl-row">{clock}
  <span class="tx"{' dir="rtl" lang="ar"' if res.task == "asr" else ' lang="fr"'}>{escape(main or "")}</span>
  <span class="ar" dir="rtl" lang="ar">{escape(side)}</span></div>""")
    return f"""<article class="paper">{stamp() if res.is_demo else ""}
  <h2 class="headline">Sous-titres, {escape(res.label.lower())}</h2>
  {warn_html(res)}
  <div class="tl-bar" aria-hidden="true">{bar}</div>
  <div class="tl">{"".join(rows)}</div></article>"""


def samples(rows: list[dict], demo: bool) -> str:
    if not rows:
        return '<p class="pending">Aucun exemple de validation dans ce journal.</p>'
    cells = "".join(
        f'<div class="try"><h4>Référence</h4><p dir="rtl" lang="ar">{escape(r.get("ref", ""))}</p>'
        f'<h4>Hypothèse du modèle</h4><p dir="rtl" lang="ar">{escape(r.get("hyp", ""))}</p></div>' for r in rows)
    return f'<div class="paper tries">{stamp("Données factices") if demo else ""}{cells}</div>'


COLOPHON = """
<section class="paper colophon">
  <h2 class="headline">Comment ce journal est fait</h2>
  <div class="body cols">
    <p><b>La presse.</b> Les titres et résumés viennent des flux publics de cinq journaux (Hespress, Echorouk, An-Nahar,
    Asharq Al-Awsat, Al Jazeera). On ne copie pas les articles : chaque entrée renvoie vers le journal. Leur traduction
    est faite par NLLB-200, un modèle de traduction de texte.</p>
    <p><b>Les vidéos.</b> L'audio d'une vidéo passe dans la chaîne du projet : un encodeur Whisper gelé, un petit module
    entraîné (projecteur ou cross-attention) et Qwen2.5-1.5B, qui écrit les sous-titres français sans passer par du
    texte arabe.</p>
    <p><b>Limites.</b> La traduction vidéo n'apparaît que si un checkpoint entraîné est chargé. Sans lui, l'interface
    montre des textes fictifs, signalés comme tels. Récupérer l'audio d'une vidéo en ligne contrevient aux conditions du
    site : c'est un outil d'essai local, sans republication.</p>
    <p><b>Données.</b> Le corpus MGB2 (QCRI) est réservé à la recherche non commerciale : cette application n'en contient
    aucun extrait.</p>
  </div>
</section>"""
