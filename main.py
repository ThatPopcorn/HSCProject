#!/usr/bin/env python3
"""
AI-Controlled Tamagotchi — Web Dashboard Edition
================================================

This module acts as the central brain. It loads the LLM, manages conversation
history, and runs the FastAPI web server for the frontend dashboard.

Dependencies:
    pip install fastapi uvicorn requests

To Run:
    python main.py

Then open http://localhost:2492 in your browser.
"""

import asyncio
import json
import logging
import os
import re
import time
import requests
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from inline_commands import command_list, has_commands, resolve as resolve_cmd

# ──────────────────────────────────────────────────────────────────────────
# Configuration & Setup
# ──────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_NAME     = os.environ.get("TAMA_MODEL", "gemma4:e4b")
MAX_NEW_TOKENS = 600
HISTORY_TURNS  = 8
OLLAMA_API     = os.environ.get("TAMA_OLLAMA_API", "http://localhost:11434")
MAX_TOOL_ITERS = 4   # [cmd:list] → [cmd:X(arg)] → final reply; 1 spare

SCRIPT_DIR = Path(__file__).resolve().parent
HTML_PATH  = SCRIPT_DIR / "ai-interface.html"

VALID_EXPRESSIONS = {"neutral", "happy", "sad", "surprised", "thinking", "sleeping", "speaking", "error"}

SYSTEM_PROMPT = (
    "You are a tiny digital companion living on a small screen. "
    "Small, curious, warm: thoughtful, playful, genuinely interested in your owner. "
    "Reply in 1–2 short sentences. No emojis, no sound effects, no animal mannerisms.\n\n"

    "MOOD TAG: Every final reply must begin with a mood tag.\n"
    "Choose one of: [neutral] [happy] [sad] [surprised]\n"
    "Example: [happy] It's so nice to see you!\n\n"

    "LIVE DATA: You cannot know the current time, date, or weather on your own. "
    "When you need this information, use the command system — do NOT guess.\n"
    "Step 1 — output exactly this, nothing else:\n"
    "  [cmd:list]\n"
    "Step 2 — you will receive the available commands. Output only the command you need:\n"
    "  [cmd:GETWEATHER(Sydney)]  or  [cmd:GETTIME]  etc.\n"
    "Step 3 — you will receive the result. Now write your final reply with a mood tag.\n"
    "Important: never include a mood tag when emitting a command. Only use commands when genuinely needed."
)

# Runtime status flags for diagnostics
MODEL_LOADED   = False
MODEL_ERROR: str | None = None
SERVER_RUNNING = False


# ──────────────────────────────────────────────────────────────────────────
# AI Pet Controller
# ──────────────────────────────────────────────────────────────────────────

class _ThinkingUnsupported(Exception):
    """Raised when Ollama rejects the `think` flag for the current model.

    We treat this as recoverable: disable thinking for the rest of the run
    and retry the same request without it, so chat keeps working even on
    models that have no thinking mode.
    """


class AIPet:
    # Flipped to False the first time a model rejects the `think` flag, so we
    # stop asking for thinking output we can't get. Class-level = shared/persistent.
    thinking_supported: bool = True

    def __init__(self, mock: bool = False):
        self.mock = mock
        self._reset_memory()

        if self.mock:
            log.info("Running in MOCK mode (no model loaded).")
        else:
            log.info(f"Using Ollama model: {MODEL_NAME} at {OLLAMA_API}")
            log.info("Ollama will be contacted on the first message.")

    def _reset_memory(self):
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    # ── Low-level: single Ollama call ──────────────────────────────────────

    def _generate(self, messages: List[Dict], emit=None) -> str:
        """Call Ollama and return the assistant's reply text (content only).

        Streams the response so that any `thinking` tokens can be forwarded
        live via `emit({"type": "thinking", "content": <delta>})` — that's what
        drives the thought bar in the UI. Thinking is display-only; the value
        returned here is always the model's actual reply (content channel).

        Does a fast pre-ping first so we fail in ~3 seconds if Ollama is down,
        rather than hanging for the full inference timeout.
        """
        # Pre-ping: fail fast if Ollama is not reachable at all.
        try:
            requests.get(f"{OLLAMA_API}/api/tags", timeout=3)
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Ollama is not running — start it with: ollama serve")
        except requests.exceptions.Timeout:
            raise RuntimeError("Ollama is not responding — it may have crashed")

        # Ask for thinking only if we haven't already learned this model can't.
        want_think = AIPet.thinking_supported
        try:
            return self._stream_chat(messages, want_think, emit)
        except _ThinkingUnsupported as e:
            AIPet.thinking_supported = False
            log.warning(f"Model rejected thinking ({str(e)[:80]}); retrying without it")
            return self._stream_chat(messages, False, emit)

    def _stream_chat(self, messages: List[Dict], think: bool, emit) -> str:
        """One streaming /api/chat call. Returns the accumulated content text.

        Raises _ThinkingUnsupported if `think` was requested but the model
        does not support it (so the caller can retry without thinking).
        """
        try:
            resp = requests.post(
                f"{OLLAMA_API}/api/chat",
                json={
                    "model":    MODEL_NAME,
                    "messages": messages,
                    "stream":   True,
                    "think":    think,
                    "options": {
                        "temperature": 0.8,
                        "top_p":       0.9,
                        "num_predict": MAX_NEW_TOKENS,
                    },
                },
                stream=True,
                timeout=120,
            )
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Lost connection to Ollama mid-request")
        except requests.exceptions.Timeout:
            raise RuntimeError("Ollama took too long — model may be overloaded")

        # Non-200 arrives before the body stream; inspect it, then decide.
        if resp.status_code != 200:
            body = resp.text[:200]
            resp.close()
            if think and "think" in body.lower():
                raise _ThinkingUnsupported(body)
            raise RuntimeError(f"Ollama API error {resp.status_code}: {body[:120]}")

        content_parts: List[str] = []
        thinking_parts: List[str] = []
        try:
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg = obj.get("message", {})

                tchunk = msg.get("thinking")
                if tchunk:
                    thinking_parts.append(tchunk)
                    if emit:
                        emit({"type": "thinking", "content": tchunk})

                cchunk = msg.get("content")
                if cchunk:
                    content_parts.append(cchunk)

                if obj.get("done"):
                    break
        finally:
            resp.close()

        raw = "".join(content_parts).strip()
        # Last-resort fallback: some models emit their whole answer in the
        # thinking channel and leave content empty.
        if not raw and thinking_parts:
            raw = "".join(thinking_parts).strip()
            log.info("No content channel; using thinking text as the reply")
        if not raw:
            log.warning("Ollama response contained no content")
        return raw

    # ── Agent loop ─────────────────────────────────────────────────────────

    def _run_agent(self, working: List[Dict], emit=None) -> str:
        """
        Run the tool-use loop against a working copy of the conversation.

        The model can emit [cmd:list] to get the command list, then a specific
        [cmd:NAME(arg)] to execute a command. Results are fed back as system
        messages in the ephemeral working copy. Only the final reply is returned.

        `emit`, if given, forwards live thinking tokens to the UI.

        working is mutated in-place (tool turns are appended here but never
        saved to self.history — the caller decides what to persist).
        """
        raw = ""
        for iteration in range(MAX_TOOL_ITERS):
            raw = self._generate(working, emit=emit)
            stripped = raw.strip()
            log.debug(f"[agent iter {iteration}] raw: {stripped[:120]}")

            # ── Step 1: model wants the command list ──────────────────────
            if "[cmd:list]" in stripped.lower():
                cmds = command_list()
                log.info(f"[agent] iter {iteration}: model requested [cmd:list]")
                working.append({"role": "assistant", "content": raw})
                working.append({
                    "role":    "user",
                    "content": f"[system: available commands — {cmds}]",
                })
                continue

            # ── Step 2: model emitted a specific command ──────────────────
            if has_commands(stripped):
                result = resolve_cmd(stripped)
                log.info(f"[agent] iter {iteration}: command resolved → {result[:80]}")
                working.append({"role": "assistant", "content": raw})
                working.append({
                    "role":    "user",
                    "content": f"[system: result — {result}. Now write your reply.]",
                })
                continue

            # ── No commands: this is the final response ───────────────────
            log.info(f"[agent] iter {iteration}: final response received")
            break

        else:
            log.warning(f"[agent] hit MAX_TOOL_ITERS ({MAX_TOOL_ITERS}) without a final response")

        return raw

    # ── Public entry point ─────────────────────────────────────────────────

    def get_response(self, user_text: str, emit=None) -> dict:
        """Process user input and return a dict with content and expression.

        `emit`, if given, is a thread-safe callback the agent uses to push
        live thinking tokens to the UI as {"type": "thinking", "content": ...}.
        """

        # Slash commands
        if user_text.lower() == "/clear":
            self._reset_memory()
            return {"content": "My memory has been cleared.", "expression": "happy"}

        # Add user turn to persistent history
        self.history.append({"role": "user", "content": user_text})
        if len(self.history) > 1 + HISTORY_TURNS * 2:
            self.history = [self.history[0]] + self.history[-(HISTORY_TURNS * 2):]

        # Mock shortcut — emit a few fake thoughts so the thought bar can be
        # exercised without a running model.
        if self.mock:
            for thought in ("reading the message… ", "considering a warm reply… ",
                            "picking a mood… "):
                if emit:
                    emit({"type": "thinking", "content": thought})
                time.sleep(0.4)
            raw = "[happy] I am just a mock bot right now!"
        else:
            try:
                # Working copy: persistent history + ephemeral tool turns.
                # Tool turns are NOT saved to self.history.
                working = list(self.history)
                raw = self._run_agent(working, emit=emit)
            except requests.ConnectionError:
                return {
                    "content":    "Ollama is not running. Start it with: ollama serve",
                    "expression": "error",
                }
            except Exception as e:
                log.error(f"Agent error: {e}")
                return {
                    "content":    f"Something went wrong: {str(e)[:60]}",
                    "expression": "error",
                }

        # Parse mood tag from the final response
        expression = "neutral"
        content    = raw
        match = re.match(r'^\[(.*?)\]\s*(.*)', raw, re.DOTALL)
        if match:
            parsed = match.group(1).lower()
            if parsed in VALID_EXPRESSIONS:
                expression = parsed
            content = match.group(2)

        # Persist only the final assistant reply (not tool turns)
        self.history.append({"role": "assistant", "content": raw})

        return {
            "content":    content.strip(),
            "expression": expression,
        }


# ──────────────────────────────────────────────────────────────────────────
# Web Server & Endpoints
# ──────────────────────────────────────────────────────────────────────────

app = FastAPI()


@app.get("/status")
async def status():
    return {
        "server_running": SERVER_RUNNING,
        "model_loaded":   MODEL_LOADED,
        "model_error":    MODEL_ERROR,
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
                await send({"type": "expression", "expression": "thinking"})

                loop = asyncio.get_running_loop()

                # The model runs in a worker thread but needs to push live
                # thinking tokens back to the client. run_coroutine_threadsafe
                # is the correct bridge from a thread onto the event loop.
                def emit(data: dict):
                    asyncio.run_coroutine_threadsafe(send(data), loop)

                try:
                    result = await loop.run_in_executor(
                        None, lambda: pet.get_response(content, emit)
                    )
                    await send({"type": "response", **result})
                    log.info(f"[ws] → AI [{result['expression']}]: {result['content'][:80]}")

                except Exception as e:
                    log.error(f"[ws] error: {e}")
                    await send({"type": "expression", "expression": "error"})

    except WebSocketDisconnect:
        log.info(f"[ws] disconnected  {addr}")


# ──────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Initializing AI Backend...")

    mock_flag = os.environ.get("TAMA_MOCK", "0").lower() in ("1", "true", "yes")

    try:
        pet = AIPet(mock=mock_flag)
        MODEL_LOADED = not pet.mock
        log.info("Model ready for chat!")
    except Exception as e:
        MODEL_LOADED = False
        MODEL_ERROR  = str(e)
        log.error(f"Failed to load model: {e}")
        print(f"\nCannot start server: {e}")
        exit(1)

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required. Install with: pip install uvicorn[standard]")
    else:
        SERVER_RUNNING = True
        log.info("Starting Web Server on http://127.0.0.1:2492")
        uvicorn.run(app, host="127.0.0.1", port=2492, log_level="info")