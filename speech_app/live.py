"""Sous-titrage en direct : audio reçu par WebSocket, tiré d'une URL (YouTube live, HLS) ou rejoué depuis un fichier.

    GET  /live                         la page (mobile d'abord) ; /live?sid=... pour regarder une session en cours
    WS   /ws/live                      envoyer l'audio : PCM 16 kHz mono int16 en binaire (paquets de ~250 ms) ;
                                       texte « stop » ou {"type":"stop"} pour finir. Reçoit les mêmes messages que les spectateurs.
    WS   /ws/regarder/{sid}            spectateur : l'historique puis le direct (autant d'écrans qu'on veut)
    POST /api/live/url                 {"url": "...", "titre": "..."} : le serveur tire l'audio (yt-dlp + ffmpeg)
    POST /api/live/fichier             fichier audio/vidéo ; /api/live/{sid}/demarrer le rejoue en temps réel
    POST /api/live/{sid}/arreter       termine la session (audio et segments écrits, bilan renvoyé)
    GET  /api/live/{sid}/etat · /media · /api/live/statut · POST /api/live/precharger

Messages JSON : {type: provisoire|definitif|etat|fin|ping, id_segment, fr, ar, debut_s, fin_s, retard_s}.
Un message ``definitif`` remplace le ``provisoire`` de même ``id_segment`` (gris -> blanc).

Un seul ASR en mémoire (S3) plus Opus-MT pour le texte provisoire ; tout appel modèle passe par GPU_LOCK
(lazy_engines.py). Chaque segment clos est écrit dans la session (sessions.py) pour le rapport de fin.
Jeton facultatif : PORTEPAROLE_API_TOKEN, en-tête Bearer ou paramètre ?token=.
Moteur factice sans GPU : PORTEPAROLE_LIVE=factice.

    python live.py --factice --port 8765     serveur autonome de démonstration
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

import lazy_engines
from lazy_engines import GPU_LOCK
from sessions import Session

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
RACINE = HERE.parent
PAQUET_OCTETS = 8000          # 250 ms de PCM 16 kHz mono int16
MAX_UPLOAD_MB = int(os.environ.get("PORTEPAROLE_MAX_UPLOAD_MB", "500"))
MAX_SESSIONS_GARDEES = 20
ASSETS = {"live.js": "text/javascript", "live.css": "text/css", "pcm-worklet.js": "text/javascript"}

router = APIRouter()
SESSIONS: dict[str, "SessionLive"] = {}
_SESSIONS_LOCK = threading.Lock()
_CONFIG: dict = {"fournisseur": None}      # tests : configurer(fournisseur=...)
_MOTEUR_FACTICE = None
TEMPS_REEL = True       # les sources serveur (fichier, replay) vont au rythme de l'audio ; les tests le coupent
RELANCES_DIRECT = 5     # un lien en direct qui s'interrompt est rouvert (yt-dlp) jusqu'à 5 fois


# ---------------------------------------------------------------- moteur
def configurer(fournisseur=None, factice: bool | None = None) -> None:
    """``fournisseur()`` -> moteur de direct (appelé sous GPU_LOCK). ``factice=True`` : moteur simulé."""
    global _MOTEUR_FACTICE
    _CONFIG["fournisseur"] = fournisseur
    if factice:
        import live_moteurs
        delai = float(os.environ.get("PORTEPAROLE_LIVE_DELAI", "0.05"))
        _MOTEUR_FACTICE = live_moteurs.MoteurFactice(delai_s=delai)
        _CONFIG["fournisseur"] = lambda: _MOTEUR_FACTICE


def fournisseur_moteur():
    """Le fournisseur configuré, ou celui de l'environnement ; HTTPException 503 si aucun moteur n'est disponible."""
    if _CONFIG["fournisseur"]:
        return _CONFIG["fournisseur"]
    if os.environ.get("PORTEPAROLE_LIVE") == "factice":
        configurer(factice=True)
        return _CONFIG["fournisseur"]
    repo = os.environ.get("PORTEPAROLE_NLP_REPO") or os.environ.get("PORTEPAROLE_S3_REPO")
    if not repo:
        raise HTTPException(503, "Moteur de direct indisponible : définir PORTEPAROLE_NLP_REPO (S3 + Opus-MT), "
                                 "ou PORTEPAROLE_LIVE=factice pour une démonstration sans modèle.")
    ckpt = os.environ.get("PORTEPAROLE_S3_CKPT") or None
    return lambda: lazy_engines.build_direct_engine(repo, ckpt)


def _est_factice() -> bool:
    return _MOTEUR_FACTICE is not None and _CONFIG["fournisseur"] is not None


class _Proxy:
    """Le flux passe par lui : le moteur est redemandé au Pool à chaque appel (un autre groupe a pu le décharger)."""

    def __init__(self, fournisseur):
        self.fournisseur = fournisseur

    def transcrire(self, signal):
        return self.fournisseur().transcrire(signal)

    def traduire_provisoire(self, ar):
        return self.fournisseur().traduire_provisoire(ar)

    def traduire_definitif(self, ar, contexte):
        return self.fournisseur().traduire_definitif(ar, contexte)


def _importer_flux():
    """Import tardif : le flux importe transcrire_s3, qui place code/demo (un autre app.py) devant sys.path."""
    for d in (RACINE / "code" / "direct",):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    import flux_incremental
    if str(HERE) in sys.path:
        sys.path.remove(str(HERE))
    sys.path.insert(0, str(HERE))
    return flux_incremental


def _jeton_ok(entetes, params) -> bool:
    jeton = os.environ.get("PORTEPAROLE_API_TOKEN") or None
    return not jeton or entetes.get("authorization", "") == f"Bearer {jeton}" or params.get("token") == jeton


def verifier(request: Request) -> None:
    if not _jeton_ok(request.headers, request.query_params):
        raise HTTPException(401, "Jeton manquant ou invalide")


# ---------------------------------------------------------------- diffusion
class Diffusion:
    """Salle de diffusion d'une session : les messages vont à tous les abonnés ; un nouvel arrivant reçoit l'historique."""

    def __init__(self):
        self._lock = threading.Lock()
        self._abonnes: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self.definitifs: list[dict] = []
        self.courant: dict | None = None
        self.etat: dict | None = None
        self.fin: dict | None = None

    def publier(self, msg: dict) -> None:
        with self._lock:
            t = msg.get("type")
            if t == "definitif":
                self.definitifs.append(msg)
                self.courant = None
            elif t == "provisoire":
                self.courant = msg
            elif t == "etat":
                self.etat = msg
            elif t == "fin":
                self.fin, self.courant = msg, None
            abonnes = list(self._abonnes)
        for boucle, q in abonnes:
            try:
                boucle.call_soon_threadsafe(q.put_nowait, msg)
            except RuntimeError:          # boucle fermée
                pass

    def abonner(self) -> tuple[asyncio.Queue, list[dict]]:
        """File du spectateur + instantané (état, définitifs, provisoire courant, fin) pris sous le même verrou."""
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._abonnes.append((asyncio.get_running_loop(), q))
            instantane = [m for m in [self.etat, *self.definitifs, self.courant, self.fin] if m]
        return q, instantane

    def desabonner(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._abonnes = [(b, f) for b, f in self._abonnes if f is not q]


# ---------------------------------------------------------------- session en direct
class SessionLive:
    def __init__(self, source: str, titre: str = "", jeter_avant_pret: bool = False):
        self.fournisseur = fournisseur_moteur()
        self.source = source
        self.jeter_avant_pret = jeter_avant_pret
        moteur_nom = _MOTEUR_FACTICE.nom if _est_factice() else "S3 + Opus-MT"
        self.session = Session.creer(source, moteur_nom, titre)
        self.sid = self.session.sid
        self.diffusion = Diffusion()
        self.file: queue.Queue = queue.Queue()
        self.pret = threading.Event()
        self.arret = threading.Event()
        self.termine = threading.Event()
        self.flux = None
        self.media: Path | None = None
        self.bilan: dict = {}
        self.erreur: str | None = None
        self._dernier_etat = 0.0
        self._etat = "chargement"
        self.sous_processus: subprocess.Popen | None = None
        self.lance = False                 # une source serveur a été lancée
        self.thread = threading.Thread(target=self._boucle, name=f"live-{self.sid}", daemon=True)
        with _SESSIONS_LOCK:
            SESSIONS[self.sid] = self
            for ancien in [s for s, v in SESSIONS.items() if v.termine.is_set()][:-MAX_SESSIONS_GARDEES]:
                SESSIONS.pop(ancien, None)
        self._publier_etat("chargement", "Chargement des modèles…")
        self.thread.start()

    # -- entrée audio
    def pousser(self, octets: bytes) -> None:
        if not self.termine.is_set():
            self.file.put(octets)

    def arreter(self) -> None:
        self.arret.set()
        self.file.put(None)

    # -- état
    def _publier_etat(self, etat: str, message: str = "") -> None:
        self._etat = etat
        flux = self.flux
        evenements = list(flux.evenements) if flux else []
        retards = [e["affichage_definitif_s"] - e["fin_parole_s"] for e in evenements]
        self.diffusion.publier({
            "type": "etat", "etat": etat, "message": message, "sid": self.sid, "source": self.source,
            "niveau_db": round(flux.niveau, 1) if flux else None, "duree_s": round(flux.duree_s, 1) if flux else 0.0,
            "nb_segments": len(evenements),
            "retard_moyen_s": round(sum(retards) / len(retards), 2) if retards else None,
            "retard_dernier_s": round(retards[-1], 2) if retards else None,
            "factice": _est_factice(),
        })
        self._dernier_etat = time.monotonic()

    def _diffuser(self, messages: list[dict]) -> None:
        for m in messages:
            if m["type"] == "definitif" and not m.get("vide"):
                self.session.ajouter_segment({"id": m["id_segment"], "debut_s": m["debut_s"], "fin_s": m["fin_s"],
                                              "ar": m["ar"], "fr": m["fr"]})
            self.diffusion.publier(m)

    # -- fil de travail
    def _boucle(self) -> None:
        try:
            flux_mod = _importer_flux()
            with GPU_LOCK:
                self.fournisseur()                                   # charge S3 + Opus-MT (ou ne fait rien s'ils le sont)
            self.flux = flux_mod.FluxIncremental(_Proxy(self.fournisseur), flux_mod.reglages_direct_live())
            if self.jeter_avant_pret:
                self._vider_file()
            self.pret.set()
            self._publier_etat("pret", "Modèles prêts")
            fini = False
            while not fini:                                          # se termine à l'arrivée du marqueur None, après tout l'audio reçu
                try:
                    item = self.file.get(timeout=0.5)
                except queue.Empty:
                    continue
                lot = [item]
                while item is not None:                              # rattrape le retard : tout ce qui attend d'un coup
                    try:
                        item = self.file.get_nowait()
                    except queue.Empty:
                        break
                    lot.append(item)
                for b in lot:
                    if b is None:
                        fini = True
                    else:
                        self.flux.ajouter_pcm16(b)
                with GPU_LOCK:
                    messages = self.flux.traiter()
                self._diffuser(messages)
                if messages or time.monotonic() - self._dernier_etat > 0.5:
                    self._publier_etat("en_cours")
            with GPU_LOCK:
                self._diffuser(self.flux.terminer())
        except Exception as exc:                                     # noqa: BLE001 : on la montre aux écrans
            self.erreur = f"{type(exc).__name__}: {exc}"
            self._publier_etat("erreur", self.erreur)
            self.pret.set()
        finally:
            self._cloturer()

    def _vider_file(self) -> None:
        """Jette l'audio arrivé pendant le chargement (en direct on ne rattrape pas), mais garde un éventuel marqueur d'arrêt."""
        arret = False
        try:
            while True:
                arret = self.file.get_nowait() is None or arret
        except queue.Empty:
            pass
        if arret:
            self.file.put(None)

    def _cloturer(self) -> None:
        self.arret.set()
        if self.sous_processus and self.sous_processus.poll() is None:
            self.sous_processus.kill()
        try:
            if self.flux is not None and self.flux.n:
                self.session.ecrire_audio(self.flux.signal())
                self.bilan = self.flux.bilan()
                meta = self.session.meta()
                meta["bilan_direct"] = self.bilan
                self.session.ecrire_meta(meta)
        except Exception as exc:                                     # noqa: BLE001
            self.erreur = self.erreur or f"écriture de la session : {exc}"
        self.termine.set()
        self.diffusion.publier({"type": "fin", "sid": self.sid, "bilan": self.bilan, "erreur": self.erreur,
                                "duree_s": round(self.flux.duree_s, 1) if self.flux else 0.0})
        self._publier_etat("erreur" if self.erreur else "termine", self.erreur or "Session terminée")


# ---------------------------------------------------------------- sources côté serveur
def commande_ffmpeg(entree: str, temps_reel: bool = True) -> list[str]:
    """ffmpeg -> PCM 16 kHz mono int16 sur la sortie standard. ``-re`` : au rythme de l'audio (fichier, replay)."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
    if temps_reel:
        cmd.append("-re")
    if entree.startswith(("http://", "https://")):              # coupure réseau : ffmpeg se reconnecte seul
        cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]
    if ".m3u8" in entree:
        cmd += ["-live_start_index", "-1"]                          # direct HLS : on part du bord du direct
    return cmd + ["-i", entree, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"]


def est_flux_direct(cible: str) -> bool:
    """Playlist HLS/DASH ou flux RTMP : la source avance d'elle-même au rythme du direct (pas de -re)."""
    c = cible.lower()
    return any(m in c for m in (".m3u8", ".mpd", "/manifest/", "hls_playlist")) or c.startswith(("rtmp", "rtsp", "srt:"))


def _binaire_ytdlp() -> list[str]:
    exe = shutil.which("yt-dlp")
    return [exe] if exe else [sys.executable, "-m", "yt_dlp"]


def resoudre_url(url: str) -> str:
    """URL de page (YouTube...) -> URL du flux audio via yt-dlp ; une URL de flux direct est renvoyée telle quelle."""
    directe = any(m in url.lower() for m in (".m3u8", ".mpd", ".mp3", ".aac", ".m4a", ".wav", ".mp4", ".ts"))
    if directe or url.lower().startswith(("rtmp:", "rtsp:", "rtmps:")):
        return url
    r = subprocess.run(_binaire_ytdlp() + ["-g", "-f", "bestaudio/best", "--no-playlist", url],
                       capture_output=True, text=True, timeout=120)
    lignes = [l for l in r.stdout.splitlines() if l.strip()]
    if r.returncode or not lignes:
        raise RuntimeError("yt-dlp n'a pas pu ouvrir ce lien : " + (r.stderr.strip().splitlines() or ["échec"])[-1][:300])
    return lignes[0]


def _alimenter(live: SessionLive, entree: str | Path, resoudre: bool, temps_reel: bool = True) -> None:
    """Fil : lance ffmpeg sur ``entree`` et pousse l'audio dans la session, jusqu'à la fin du flux ou à l'arrêt."""
    try:
        live.pret.wait()                                             # en direct, on n'accumule pas d'audio pendant le chargement
        if live.arret.is_set():
            return
        # Un lien (direct YouTube, HLS) arrive déjà au rythme du direct : -re le ferait dériver derrière la playlist
        # jusqu'à ce que ses morceaux expirent. -re ne sert qu'à rejouer un fichier.
        relances = RELANCES_DIRECT if resoudre else 0
        recu = 0
        while True:
            cible = resoudre_url(str(entree)) if resoudre else str(entree)
            rythme = temps_reel and not est_flux_direct(cible)
            with tempfile.TemporaryFile() as err:
                proc = subprocess.Popen(commande_ffmpeg(cible, rythme), stdout=subprocess.PIPE, stderr=err)
                live.sous_processus = proc
                while not live.arret.is_set():
                    octets = proc.stdout.read(PAQUET_OCTETS)
                    if not octets:
                        break
                    recu += len(octets)
                    live.pousser(octets)
                proc.kill()
                proc.wait()
                if recu == 0 and not live.arret.is_set():
                    err.seek(0)
                    raise RuntimeError("aucun audio reçu : " + err.read().decode("utf-8", "replace").strip()[-300:])
            if live.arret.is_set() or relances == 0:
                break
            relances -= 1                                            # le direct s'est interrompu : on rouvre le lien
            live._publier_etat("en_cours", "Flux interrompu, reconnexion…")
    except FileNotFoundError:
        live.erreur = "ffmpeg n'est pas installé sur le serveur"
    except Exception as exc:                                         # noqa: BLE001
        live.erreur = f"{type(exc).__name__}: {exc}"
    finally:
        if live.erreur:
            live._publier_etat("erreur", live.erreur)
        live.arreter()


def _lancer_source(live: SessionLive, entree, resoudre: bool) -> None:
    threading.Thread(target=_alimenter, args=(live, entree, resoudre, TEMPS_REEL), daemon=True,
                     name=f"source-{live.sid}").start()


# ---------------------------------------------------------------- pages
def _page(nom: str) -> HTMLResponse:
    chemin = STATIC / nom
    if not chemin.is_file():
        raise HTTPException(404, f"{nom} introuvable")
    return HTMLResponse(chemin.read_text(encoding="utf-8"))


@router.get("/live", include_in_schema=False)
def page_live():
    return _page("live.html")


@router.get("/live/assets/{nom}", include_in_schema=False)
def asset(nom: str):
    if nom not in ASSETS:
        raise HTTPException(404)
    return FileResponse(STATIC / nom, media_type=ASSETS[nom], headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------- API
def _session(sid: str) -> SessionLive:
    live = SESSIONS.get(sid)
    if not live:
        raise HTTPException(404, "Session inconnue (ou le serveur a redémarré)")
    return live


@router.get("/api/live/statut", dependencies=[Depends(verifier)])
def statut():
    try:
        fournisseur_moteur()
        moteur = True
        raison = None
    except HTTPException as exc:
        moteur, raison = False, exc.detail
    return {"moteur_disponible": moteur, "raison": raison, "groupe_charge": lazy_engines.POOL.loaded,
            "factice": os.environ.get("PORTEPAROLE_LIVE") == "factice" or _est_factice(),
            "lora": os.environ.get("PORTEPAROLE_LORA") or None,
            "ffmpeg": shutil.which("ffmpeg") is not None, "yt_dlp": shutil.which("yt-dlp") is not None,
            "sessions": [{"sid": s, "source": v.source, "etat": v._etat} for s, v in SESSIONS.items()
                         if not v.termine.is_set()]}


@router.post("/api/live/precharger", dependencies=[Depends(verifier)])
def precharger():
    """Charge S3 et Opus-MT en arrière-plan (à appeler avant la démonstration)."""
    fournisseur = fournisseur_moteur()

    def charger():
        with GPU_LOCK:
            fournisseur()
    threading.Thread(target=charger, daemon=True, name="precharge-live").start()
    return {"etat": "chargement lancé"}


@router.post("/api/live/url", dependencies=[Depends(verifier)])
async def demarrer_url(corps: dict):
    url = str(corps.get("url", "")).strip()
    if not url.lower().startswith(("http://", "https://", "rtmp://", "rtsp://", "rtmps://")):
        raise HTTPException(422, "Il faut un lien http(s) : page YouTube, flux HLS (.m3u8)...")
    live = SessionLive(f"url:{url[:200]}", str(corps.get("titre", "")))
    _lancer_source(live, url, resoudre=True)
    return {"sid": live.sid, "etat": "chargement"}


@router.post("/api/live/fichier", dependencies=[Depends(verifier)])
async def deposer_fichier(file: UploadFile = File(...), titre: str = Form("")):
    live = SessionLive("fichier", titre or (file.filename or ""))
    dossier = live.session.dossier / "media"
    dossier.mkdir(exist_ok=True)
    nom = Path(file.filename or "media").name or "media"
    chemin = dossier / nom
    taille = 0
    with open(chemin, "wb") as sortie:
        while bloc := await file.read(1 << 20):
            taille += len(bloc)
            if taille > MAX_UPLOAD_MB << 20:
                live.arreter()
                raise HTTPException(413, f"Fichier trop gros (max {MAX_UPLOAD_MB} Mo)")
            sortie.write(bloc)
    live.media = chemin
    return {"sid": live.sid, "media": f"/api/live/{live.sid}/media", "nom": nom}


@router.post("/api/live/{sid}/demarrer", dependencies=[Depends(verifier)])
def demarrer_fichier(sid: str):
    live = _session(sid)
    if not live.media:
        raise HTTPException(409, "Cette session n'a pas de fichier à rejouer")
    if not live.lance and not live.arret.is_set():
        live.lance = True
        _lancer_source(live, live.media, resoudre=False)
    return {"sid": sid, "etat": live._etat}


@router.get("/api/live/{sid}/media", dependencies=[Depends(verifier)])
def media(sid: str):
    live = _session(sid)
    if not live.media or not live.media.is_file():
        raise HTTPException(404)
    return FileResponse(live.media)


@router.get("/api/live/{sid}/etat", dependencies=[Depends(verifier)])
def etat(sid: str):
    live = _session(sid)
    return {"sid": sid, "etat": live._etat, "termine": live.termine.is_set(), "bilan": live.bilan, "erreur": live.erreur,
            "segments": len(live.diffusion.definitifs)}


@router.post("/api/live/{sid}/arreter", dependencies=[Depends(verifier)])
async def arreter(sid: str):
    live = _session(sid)
    live.arreter()
    await asyncio.get_running_loop().run_in_executor(None, live.termine.wait, 120)
    return {"sid": sid, "termine": live.termine.is_set(), "bilan": live.bilan, "erreur": live.erreur,
            "segments": len(live.diffusion.definitifs)}


# ---------------------------------------------------------------- WebSocket
async def _transmettre(ws: WebSocket, q: asyncio.Queue) -> None:
    """File de la salle -> socket ; un ping toutes les 15 s détecte les écrans partis."""
    while True:
        try:
            msg = await asyncio.wait_for(q.get(), 15)
        except asyncio.TimeoutError:
            msg = {"type": "ping"}
        await ws.send_text(json.dumps(msg, ensure_ascii=False))


@router.websocket("/ws/live")
async def ws_envoyer(ws: WebSocket):
    if not _jeton_ok(ws.headers, ws.query_params):
        await ws.close(code=4401)
        return
    await ws.accept()
    source = ws.query_params.get("source", "micro")
    try:
        live = SessionLive(source if source in ("micro", "onglet") else "micro", ws.query_params.get("titre", ""),
                           jeter_avant_pret=True)
    except HTTPException as exc:
        await ws.send_text(json.dumps({"type": "etat", "etat": "erreur", "message": exc.detail}, ensure_ascii=False))
        await ws.close()
        return
    q, instantane = live.diffusion.abonner()
    for m in instantane:
        await ws.send_text(json.dumps(m, ensure_ascii=False))
    envoi = asyncio.create_task(_transmettre(ws, q))
    connecte = True
    try:
        while True:
            m = await ws.receive()
            if m["type"] == "websocket.disconnect":
                connecte = False
                break
            if m.get("bytes"):
                if live._etat != "chargement":
                    live.pousser(m["bytes"])
            elif m.get("text"):
                texte = m["text"].strip()
                if texte == "stop" or '"stop"' in texte:
                    break
    except WebSocketDisconnect:
        connecte = False
    finally:
        live.arreter()
        envoi.cancel()
        await asyncio.gather(envoi, return_exceptions=True)
        if connecte:                                     # la fin de session (bilan) est encore due à celui qui a envoyé « stop »
            await asyncio.get_running_loop().run_in_executor(None, live.termine.wait, 120)
            try:
                while not q.empty():
                    await ws.send_text(json.dumps(q.get_nowait(), ensure_ascii=False))
                await ws.close()
            except (WebSocketDisconnect, RuntimeError):
                pass
        live.diffusion.desabonner(q)


@router.websocket("/ws/regarder/{sid}")
async def ws_regarder(ws: WebSocket, sid: str):
    if not _jeton_ok(ws.headers, ws.query_params):
        await ws.close(code=4401)
        return
    await ws.accept()
    live = SESSIONS.get(sid)
    if live is None:                                     # session déjà écrite sur disque : on rejoue ses segments
        try:
            s = Session.ouvrir(sid)
        except FileNotFoundError:
            await ws.send_text(json.dumps({"type": "etat", "etat": "erreur", "message": "Session inconnue"}))
            await ws.close()
            return
        for seg in s.segments():
            await ws.send_text(json.dumps({"type": "definitif", "id_segment": seg["id"], "fr": seg["fr"], "ar": seg["ar"],
                                           "debut_s": seg["debut_s"], "fin_s": seg["fin_s"], "retard_s": None}))
        await ws.send_text(json.dumps({"type": "fin", "sid": sid, "bilan": s.meta().get("bilan_direct", {})}))
        await ws.close()
        return
    q, instantane = live.diffusion.abonner()
    try:
        for m in instantane:
            await ws.send_text(json.dumps(m, ensure_ascii=False))
        envoi = asyncio.create_task(_transmettre(ws, q))
        while True:                                       # on lit pour détecter la déconnexion
            m = await ws.receive()
            if m["type"] == "websocket.disconnect":
                break
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        live.diffusion.desabonner(q)
        try:
            envoi.cancel()
        except UnboundLocalError:
            pass


# ---------------------------------------------------------------- autonome
def creer_app(factice: bool = False):
    from fastapi import FastAPI
    if factice:
        configurer(factice=True)
    app = FastAPI(title="Le Porte-Parole : direct")
    app.include_router(router)
    try:                                                # rapport de fin de session, écrit par ailleurs
        import session_api
        app.include_router(session_api.router)
    except ImportError:
        pass
    return app


if __name__ == "__main__":
    import argparse
    import uvicorn

    p = argparse.ArgumentParser(description="Serveur de direct autonome (sans l'interface Gradio).")
    p.add_argument("--factice", action="store_true", help="moteur simulé, sans GPU")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    a = p.parse_args()
    uvicorn.run(creer_app(a.factice), host=a.host, port=a.port)
