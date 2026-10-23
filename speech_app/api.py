"""API HTTP + interface Gradio dans un seul serveur : `uvicorn api:app --host 0.0.0.0 --port 7860`.

    GET  /api/health                    état du service
    GET  /api/engines                   moteurs disponibles (s0, seamless, s3) et moteur chargé
    POST /api/translate                 fichier audio ou vidéo + engine=... [&task=st|asr] [&format=json|srt|vtt] [&wait=false]
    GET  /api/jobs/{id}                 état et résultat d'un travail lancé avec wait=false
    GET  /api/jobs/{id}/subtitles.srt   sous-titres d'un travail terminé (ou .vtt)
    GET  /live                          sous-titres en direct (micro, onglet, lien YouTube/HLS, fichier) : voir live.py
    GET  /  /video  /rapport  /sessions pages de l'interface : voir pages.py
    GET  /atelier                       interface Gradio (comparaison des moteurs, démonstration technique)

Les modèles sont chargés au premier appel du moteur demandé ; un seul reste en mémoire (voir lazy_engines.py).
Un jeton facultatif protège l'API : variable PORTEPAROLE_API_TOKEN, en-tête `Authorization: Bearer <jeton>`.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gradio as gr
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse

import app as ui
import front
import live
import media
import pages
import session_report
from engines import Result, to_srt, to_vtt

TOKEN = os.environ.get("PORTEPAROLE_API_TOKEN") or None
MAX_UPLOAD_MB = int(os.environ.get("PORTEPAROLE_MAX_UPLOAD_MB", "500"))
POOL = ThreadPoolExecutor(max_workers=1)      # un seul traitement GPU à la fois
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

api = FastAPI(title="Le Porte-Parole : traduction de parole arabe vers le français")


def check_token(request: Request) -> None:
    if TOKEN and request.headers.get("authorization", "") != f"Bearer {TOKEN}":
        raise HTTPException(401, "Jeton manquant ou invalide")


def result_json(res: Result) -> dict:
    return {
        "engine": res.engine, "label": res.label, "task": res.task, "duration_s": round(res.duration, 2),
        "seconds": round(res.seconds, 2) if res.seconds is not None else None,
        "rtf": round(res.seconds / res.duration, 3) if res.seconds and res.duration else None,
        "peak_mem_gb": round(res.peak_mem_gb, 2) if res.peak_mem_gb else None,
        "produces_arabic": res.produces_arabic, "warning": res.warning,
        "text": res.main_text,
        "segments": [{"start": round(s.start, 2), "end": round(s.end, 2), "fr": s.fr, "ar": s.ar} for s in res.segments],
        "srt": to_srt(res), "vtt": to_vtt(res),
    }


def to_wav(src: str, folder: str) -> str:
    """Audio ou vidéo -> wav mono 16 kHz (ffmpeg), erreurs converties en réponses HTTP."""
    try:
        return media.to_wav(src, folder)
    except media.MediaError as exc:
        raise HTTPException(500 if "n'est pas installé" in str(exc) else 400, str(exc))


def run_engine(engine: str, wav: str, task: str, titre: str = "") -> dict:
    """Lance le moteur sous le verrou GPU global, puis garde audio et segments dans une Session (champ `sid`)."""
    with session_report.verrou_gpu():
        res = ui.ENGINES[engine].run(wav, task)
    out = result_json(res)
    try:
        out["sid"] = front.creer_session(res, wav, titre).sid
    except Exception:                                         # noqa: BLE001 : les sous-titres restent utilisables sans rapport
        out["sid"] = None
    return out


def execute(job_id: str, engine: str, wav: str, task: str, folder: str, titre: str = "") -> None:
    with JOBS_LOCK:
        JOBS[job_id]["status"] = "running"
    try:
        out = run_engine(engine, wav, task, titre)
        with JOBS_LOCK:
            JOBS[job_id].update(status="done", result=out)
    except Exception as exc:                                  # noqa: BLE001 : on renvoie l'erreur au client
        with JOBS_LOCK:
            JOBS[job_id].update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(folder, ignore_errors=True)


@api.get("/api/health")
def health():
    return {"status": "ok", "mode": ui.MODE, "engines": len(ui.ENGINES)}


@api.get("/api/engines", dependencies=[Depends(check_token)])
def engines():
    loaded = getattr(next((e.pool for e in ui.ENGINES.values() if hasattr(e, "pool")), None), "loaded", None)
    return {"loaded_group": loaded, "engines": [
        {"name": n, "label": e.label, "description": e.blurb, "produces_arabic": e.produces_arabic,
         "status": e.status, "group": getattr(e, "group", None)} for n, e in ui.ENGINES.items()]}


@api.post("/api/translate", dependencies=[Depends(check_token)])
async def translate(file: UploadFile = File(...), engine: str = Form(None), task: str = Form("st"),
                    format: str = Form("json"), wait: str = Form("true")):
    engine = engine or ui.DEFAULT_ENGINE
    if engine not in ui.ENGINES:
        raise HTTPException(404, f"Moteur inconnu : {engine}. Disponibles : {', '.join(ui.ENGINES)}")
    if task not in ("st", "asr") or format not in ("json", "srt", "vtt"):
        raise HTTPException(422, "task : st|asr ; format : json|srt|vtt")
    folder = tempfile.mkdtemp(prefix="porteparole_api_")
    src = Path(folder) / (Path(file.filename or "entree").name or "entree")
    size = 0
    with open(src, "wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                shutil.rmtree(folder, ignore_errors=True)
                raise HTTPException(413, f"Fichier trop gros (max {MAX_UPLOAD_MB} Mo)")
            out.write(chunk)
    try:
        wav = to_wav(str(src), folder)
    except HTTPException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {"id": job_id, "engine": engine, "task": task, "status": "queued"}
    fut = POOL.submit(execute, job_id, engine, wav, task, folder, src.name)
    if wait.lower() in ("false", "0", "no", "non"):
        return {"job_id": job_id, "status": "queued", "poll": f"/api/jobs/{job_id}"}
    await __import__("asyncio").get_running_loop().run_in_executor(None, fut.result)
    job = JOBS[job_id]
    if job["status"] == "error":
        raise HTTPException(500, job["error"])
    res = job["result"]
    if format == "srt":
        return PlainTextResponse(res["srt"], media_type="application/x-subrip")
    if format == "vtt":
        return PlainTextResponse(res["vtt"], media_type="text/vtt")
    return {"job_id": job_id, **res}


@api.get("/api/jobs/{job_id}", dependencies=[Depends(check_token)])
def job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Travail inconnu")
    return JOBS[job_id]


@api.get("/api/jobs/{job_id}/subtitles.{ext}", dependencies=[Depends(check_token)])
def job_subtitles(job_id: str, ext: str):
    j = JOBS.get(job_id)
    if not j or j["status"] != "done" or ext not in ("srt", "vtt"):
        raise HTTPException(404, "Sous-titres indisponibles")
    return PlainTextResponse(j["result"][ext])


# Pages (accueil, vidéo, rapport) et fichiers statiques
api.include_router(pages.router)
# Direct : /live, /ws/live, /ws/regarder/{sid}, /api/live/*  (à monter avant l'interface Gradio)
api.include_router(live.router)
try:                                                        # rapport de fin de session, écrit à part
    import session_api
    api.include_router(session_api.router)
except ImportError:
    pass

if os.environ.get("PORTEPAROLE_FRONT") == "old":          # ancienne interface (presse, vidéos, atelier…)
    if not ui.FROZEN:
        ui.start_background()
    app = gr.mount_gradio_app(api, ui.build_app(), path="/atelier", **ui.launch_kwargs())
else:                                                       # interface simple : on dépose un mp4 ou un audio
    app = gr.mount_gradio_app(api, front.build_front(ui.ENGINES, ui.REAL), path="/atelier", **front.launch_kwargs())
