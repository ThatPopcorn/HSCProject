#!/usr/bin/env python3
"""
server.py — AI Interface backend

  pip install fastapi uvicorn
  python server.py

Then open:  http://localhost:8000
WebSocket:  ws://localhost:8000/ws  (already the default in the UI)
"""

import asyncio
import json
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

app  = FastAPI()
HTML = Path(__file__).parent / "ai-interface.html"


# ─────────────────────────────────────────────────────────────────────────────
#  PLUG YOUR MODEL IN HERE
#
#  get_response() is the only function you need to replace.
#  It receives the user's text (string) and must return a dict with:
#
#    content    (str)  — the reply text that will be spoken / displayed
#    expression (str)  — face to show while speaking (optional, default "neutral")
#
#  Valid expressions:
#    neutral  happy  sad  surprised  thinking  sleeping  speaking  error
#
#  This function is called in a thread executor so blocking calls (model
#  inference, API requests, etc.) are fine — they won't stall the event loop.
# ─────────────────────────────────────────────────────────────────────────────

def get_response(text: str) -> dict:
    """Replace the body of this function with your AI model logic."""

    # ── Example: simple echo ─────────────────────────────────────────────────
    return {
        "content":    f"You said: {text}",
        "expression": "happy",
    }

    # ── Example: OpenAI ──────────────────────────────────────────────────────
    # from openai import OpenAI
    # client = OpenAI()
    # reply = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[{"role": "user", "content": text}],
    # )
    # return {
    #     "content":    reply.choices[0].message.content,
    #     "expression": "neutral",
    # }

    # ── Example: local Ollama ─────────────────────────────────────────────────
    # import requests
    # r = requests.post("http://localhost:11434/api/generate", json={
    #     "model": "llama3", "prompt": text, "stream": False,
    # })
    # return {
    #     "content":    r.json()["response"],
    #     "expression": "neutral",
    # }


# ─────────────────────────────────────────────────────────────────────────────
#  HTTP — serve the frontend
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    if not HTML.exists():
        return HTMLResponse("<h1>ai-interface.html not found next to server.py</h1>", status_code=404)
    return HTMLResponse(HTML.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
#  WEBSOCKET — handle messages from the UI
#
#  Incoming message format (from frontend):
#    { "type": "text", "content": "..." }
#
#  Outgoing message formats (to frontend):
#    { "type": "response",   "content": "...", "expression": "..." }
#    { "type": "expression", "expression": "..." }
#    { "type": "speak",      "content": "..." }
#    { "type": "log",        "message": "...", "level": "info|warn|error|ok|debug" }
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    addr = f"{websocket.client.host}:{websocket.client.port}"
    log.info(f"[ws] connected  {addr}")

    async def send(data: dict):
        await websocket.send_text(json.dumps(data))

    async def ui_log(msg: str, level: str = "info"):
        await send({"type": "log", "message": msg, "level": level})

    await ui_log("Backend connected", "ok")

    try:
        while True:
            raw = await websocket.receive_text()

            # Parse
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                log.warning(f"[ws] bad JSON: {raw[:80]}")
                await ui_log(f"Bad JSON received", "error")
                continue

            msg_type = msg.get("type", "")
            content  = msg.get("content", "").strip()
            log.info(f"[ws] ← {msg_type}: {content[:80]}")

            # Route
            if msg_type == "text":
                if not content:
                    continue

                # Show "thinking" while the model runs
                await send({"type": "expression", "expression": "thinking"})

                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, get_response, content
                    )

                    # Validate result shape before sending
                    if not isinstance(result, dict) or "content" not in result:
                        raise ValueError(f"get_response() must return dict with 'content' key, got: {result!r}")

                    await send({"type": "response", **result})
                    log.info(f"[ws] → {result.get('content','')[:80]}")

                except Exception as e:
                    log.error(f"[ws] model error: {e}")
                    await send({"type": "expression", "expression": "error"})
                    await ui_log(f"Model error: {e}", "error")

            else:
                log.debug(f"[ws] unknown type: {msg_type}")
                await ui_log(f"Unknown message type: '{msg_type}'", "warn")

    except WebSocketDisconnect:
        log.info(f"[ws] disconnected  {addr}")
    except Exception as e:
        log.error(f"[ws] unexpected error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting AI Interface server")
    log.info("Open  →  http://localhost:8000")
    log.info("WS    →  ws://localhost:8000/ws")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
