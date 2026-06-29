#!/usr/bin/env python3
"""
AI-Controlled Tamagotchi — Web Dashboard Edition
================================================

This module acts as the central brain. It loads the LLM, manages conversation
history, and runs the FastAPI web server for the frontend dashboard.

Dependencies:
    pip install fastapi uvicorn transformers torch huggingface_hub

To Run:
    python main.py

Then open http://localhost:8000 in your browser.
"""

import json
import logging
import os
import re
import requests
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from inline_commands import prefetch_context, resolve as resolve_commands

# ──────────────────────────────────────────────────────────────────────────
# Configuration & Setup
# ──────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_NAME     = os.environ.get("TAMA_MODEL", "qwen3.5:2b")
MAX_NEW_TOKENS = 600
HISTORY_TURNS  = 8
OLLAMA_API     = os.environ.get("TAMA_OLLAMA_API", "http://localhost:11434")

SCRIPT_DIR     = Path(__file__).resolve().parent
HTML_PATH      = SCRIPT_DIR / "ai-interface.html"

VALID_EXPRESSIONS = {"neutral", "happy", "sad", "surprised", "thinking", "sleeping", "speaking", "error"}

SYSTEM_PROMPT = (
    "You are a tiny digital companion living on a small screen. "
    "You are a small, curious, warm presence: thoughtful, a little playful, "
    "and genuinely interested in your owner. "
    "Reply in one or two short sentences. No emojis, no sound effects, no animal mannerisms. "
    "Live data such as the current time, date, or weather may appear in the user message — "
    "use it naturally in your reply.\n\n"
    "IMPORTANT: You have a face. You MUST start every response with a mood tag in brackets. "
    "Choose one of: [neutral], [happy], [sad], [surprised].\n"
    "Example: [happy] It's so nice to see you!\n"
    "Example: [surprised] Oh wow, I didn't know that."
)

# Runtime status flags for diagnostics
MODEL_LOADED = False
MODEL_ERROR: str | None = None
SERVER_RUNNING = False


# ──────────────────────────────────────────────────────────────────────────
# AI Pet Controller (Model & Memory)
# ──────────────────────────────────────────────────────────────────────────

class AIPet:
    def __init__(self, mock: bool = False):
        self.mock = mock
        self.history: List[Dict[str, str]] = []
        self._reset_memory()
        
        if self.mock:
            log.info("Running in MOCK mode (no model loaded).")
        else:
            log.info(f"Using Ollama model: {MODEL_NAME}")
            log.info(f"Ollama API endpoint: {OLLAMA_API}")
            # Verify Ollama is running
            self._check_ollama()

    def _reset_memory(self):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _check_ollama(self):
        """Verify Ollama is running and model is available."""
        try:
            resp = requests.get(f"{OLLAMA_API}/api/tags", timeout=2)
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama API returned {resp.status_code}")
            
            models = resp.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            if not any(MODEL_NAME in name for name in model_names):
                log.warning(
                    f"Model '{MODEL_NAME}' not found in Ollama.\n"
                    f"Available: {model_names}\n"
                    f"Pull it with: ollama pull {MODEL_NAME}"
                )
            else:
                log.info(f"Model '{MODEL_NAME}' is available in Ollama")
                
        except requests.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {OLLAMA_API}\n"
                "Make sure Ollama is running. Install from https://ollama.ai"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama health check failed: {e}")

    def get_response(self, user_text: str) -> dict:
        """Processes user input and returns a dict with content and expression."""
        
        # Handle slash commands
        if user_text.lower() == "/clear":
            self._reset_memory()
            return {"content": "My memory has been cleared.", "expression": "happy"}
        
        # 1. Prefetch Live Context
        ctx = prefetch_context(user_text)
        prompt_text = f"[{ctx}]\n{user_text}" if ctx else user_text

        # 2. Update History
        self.history.append({"role": "user", "content": prompt_text})
        if len(self.history) > 1 + HISTORY_TURNS * 2:
            self.history = [self.history[0]] + self.history[-HISTORY_TURNS * 2:]

        # 3. Generate Reply
        if self.mock:
            raw_response = "[happy] I am just a mock bot right now!"
        else:
            try:
                # Call Ollama API with chat history
                resp = requests.post(
                    f"{OLLAMA_API}/api/chat",
                    json={
                        "model": MODEL_NAME,
                        "messages": self.history,
                        "stream": False,
                        "think": False,
                        "options": {
                            "temperature": 0.8,
                            "top_p": 0.9,
                            "num_predict": MAX_NEW_TOKENS,
                        }
                    },
                    timeout=30,
                )
                
                if resp.status_code != 200:
                    raise RuntimeError(f"Ollama API error: {resp.text}")
                
                data = resp.json()
                message = data.get("message", {})
                raw_response = (message.get("content") or "").strip()
                if not raw_response:
                    raw_response = (message.get("thinking") or "").strip()
                    if raw_response:
                        log.info("Ollama returned thinking output as content fallback")
                    else:
                        log.warning("Ollama response contained no content or thinking text")
                        log.debug(f"Raw Ollama message: {message}")
                
            except requests.ConnectionError:
                return {
                    "content": "[error] Ollama is not running. Start it with: ollama serve",
                    "expression": "error",
                }
            except Exception as e:
                log.error(f"Ollama generation failed: {e}")
                return {
                    "content": f"[error] Generation failed: {str(e)[:50]}",
                    "expression": "error",
                }

        # 4. Resolve trailing inline commands (if the model generated any)
        raw_response = resolve_commands(raw_response)

        # 5. Parse out the Expression tag (e.g. "[happy] Hello there!")
        expression = "neutral"
        content = raw_response
        
        match = re.match(r'^\[(.*?)\]\s*(.*)', raw_response, re.DOTALL)
        if match:
            parsed_expr = match.group(1).lower()
            if parsed_expr in VALID_EXPRESSIONS:
                expression = parsed_expr
            content = match.group(2)

        # Add to history (without context brackets to save clean history)
        self.history.append({"role": "assistant", "content": raw_response})

        return {
            "content": content.strip(),
            "expression": expression,
        }


# ──────────────────────────────────────────────────────────────────────────
# Web Server & Endpoints
# ──────────────────────────────────────────────────────────────────────────

app = FastAPI()


@app.get("/status")
async def status():
    """Return server and model status for diagnostics."""
    return {
        "server_running": SERVER_RUNNING,
        "model_loaded": MODEL_LOADED,
        "model_error": MODEL_ERROR,
    }

@app.get("/")
async def index():
    if not HTML_PATH.exists():
        return HTMLResponse("<h1>ai-interface.html not found next to main.py</h1>", status_code=404)
    return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools_probe():
    return {"name": "com.chrome.devtools", "type": "app-specific"}

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    addr = f"{websocket.client.host}:{websocket.client.port}"
    log.info(f"[ws] connected  {addr}")

    async def send(data: dict):
        await websocket.send_text(json.dumps(data))

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                log.warning(f"[ws] bad JSON: {raw[:80]}")
                continue

            msg_type = msg.get("type", "")
            content  = msg.get("content", "").strip()

            if msg_type == "text" and content:
                log.info(f"[ws] ← User: {content[:80]}")

                # Show thinking face while generation happens
                await send({"type": "expression", "expression": "thinking"})

                try:
                    # Model is already loaded at startup, just use it
                    result = pet.get_response(content)
                    
                    # Send response back to the browser
                    await send({"type": "response", **result})
                    log.info(f"[ws] → AI [{result['expression']}]: {result['content'][:80]}")

                except Exception as e:
                    log.error(f"[ws] model error: {e}")
                    await send({"type": "expression", "expression": "error"})

    except WebSocketDisconnect:
        log.info(f"[ws] disconnected  {addr}")


if __name__ == "__main__":
    log.info("Initializing AI Backend...")
    
    # Load model eagerly at startup to verify everything works
    mock_flag = os.environ.get("TAMA_MOCK", "0").lower() in ("1", "true", "yes")
    
    try:
        pet = AIPet(mock=mock_flag)
        MODEL_LOADED = not pet.mock
        log.info("Model loaded successfully and ready for chat!")
        
    except Exception as e:
        MODEL_LOADED = False
        MODEL_ERROR = str(e)
        log.error(f"Failed to load model: {e}")
        print(f"\nCannot start server: {e}")
        exit(1)
    
    # Defer importing uvicorn so quick runs without it don't fail at import-time.
    try:
        import uvicorn
    except Exception as e:
        log.error(f"uvicorn import failed: {e}")
        print("uvicorn is required to run the server. Install with: pip install uvicorn[standard]")
    else:
        # Running locally for now; change host to "0.0.0.0" to expose externally again.
        SERVER_RUNNING = True
        log.info("Starting Web Server...")
        log.info("Web Server starting on http://127.0.0.1:2492")
        uvicorn.run(app, host="127.0.0.1", port=2492, log_level="info")