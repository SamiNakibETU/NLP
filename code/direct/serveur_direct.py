"""Serveur FastAPI avec WebSocket par spectateur et page web mobile pour le sous-titrage en direct."""

import argparse
import asyncio
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

DOSSIER_DIRECT = Path(__file__).resolve().parent
if str(DOSSIER_DIRECT) not in sys.path:
    sys.path.insert(0, str(DOSSIER_DIRECT))

from flux_simule import simuler_flux_direct  # noqa: E402


PAGE_HTML_DIRECT = """<!DOCTYPE html>
<html lang="fr" dir="ltr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sous-titrage en direct (Arabe -> Français)</title>
  <style>
    body {
      margin: 0;
      padding: 16px;
      background: #090d16;
      color: #ffffff;
      font-family: system-ui, -apple-system, sans-serif;
      display: flex;
      flex-direction: column;
      justify-content: flex-end;
      min-height: 92vh;
      direction: ltr;
      text-align: left;
    }
    .bandeau-sous-titres {
      background: rgba(17, 24, 39, 0.92);
      border-radius: 10px;
      padding: 14px 18px;
      font-size: clamp(1.1rem, 4vw, 1.65rem);
      line-height: 1.45;
      /* Deux lignes au plus */
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      max-height: 2.9em;
      direction: ltr;
    }
    .confirme {
      color: #ffffff;
      font-weight: 600;
    }
    .provisoire {
      color: #9ca3af;
      font-weight: 400;
    }
    .meta {
      margin-top: 8px;
      font-size: 0.8rem;
      color: #6b7280;
    }
  </style>
</head>
<body>
  <div class="bandeau-sous-titres" id="zone-st" dir="ltr">
    <span class="confirme" id="txt-confirme">En attente du flux direct...</span>
    <span class="provisoire" id="txt-provisoire"></span>
  </div>
  <div class="meta" id="meta-info">Segment : - | Horodatage : 0.0 s</div>

  <script>
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${window.location.host}/ws/direct`);
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      document.getElementById("txt-confirme").textContent = msg.texte_confirme || "";
      document.getElementById("txt-provisoire").textContent = msg.texte_provisoire ? (" " + msg.texte_provisoire) : "";
      document.getElementById("meta-info").textContent = `Segment : ${msg.id_segment} | Horodatage : ${msg.horodatage} s`;
    };
  </script>
</body>
</html>
"""


def construire_serveur_direct() -> FastAPI:
    """Construit l'application FastAPI gérant la page mobile et un WebSocket par spectateur."""
    app = FastAPI(title="Sous-titrage en direct Arabe -> Français")
    spectateurs: List[WebSocket] = []

    @app.get("/", response_class=HTMLResponse)
    def page_accueil() -> str:
        return PAGE_HTML_DIRECT

    @app.websocket("/ws/direct")
    async def websocket_spectateur(ws: WebSocket) -> None:
        await ws.accept()
        spectateurs.append(ws)
        try:
            # Envoie un premier message d'état structuré dès la connexion
            await ws.send_json(
                {
                    "id_segment": "seg_0001",
                    "texte_confirme": "Bonsoir et bienvenue",
                    "texte_provisoire": "dans ce journal télévisé.",
                    "horodatage": 1.0,
                }
            )
            while True:
                data = await ws.receive_text()
                if data == "stop":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            if ws in spectateurs:
                spectateurs.remove(ws)

    return app


def construire_parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serveur FastAPI + WebSocket pour le mode direct.")
    parser.add_argument("--essai", action="store_true", help="Vérifie la création du serveur et la simulation 600 s.")
    parser.add_argument("--hote", type=str, default="127.0.0.1", help="Adresse d'écoute.")
    parser.add_argument("--port", type=int, default=8000, help="Port HTTP/WebSocket.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = construire_parseur()
    args = parser.parse_args(argv)
    if args.essai:
        app = construire_serveur_direct()
        sim = simuler_flux_direct(duree_totale_s=600.0)
        print(
            f"[SERVEUR DIRECT OK] Routes={len(app.routes)} | "
            f"Simulation 600s : {sim['nb_segments_clos']} segments, retard={sim['retard_moyen_s']}s"
        )
        return 0

    import uvicorn

    app = construire_serveur_direct()
    uvicorn.run(app, host=args.hote, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
