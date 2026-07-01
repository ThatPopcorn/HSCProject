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

from inline_commands import command_list, has_commands, run_first

# ──────────────────────────────────────────────────────────────────────────
# Configuration & Setup
# ──────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR  = Path(__file__).resolve().parent
HTML_PATH   = SCRIPT_DIR / "ai-interface.html"
CONFIG_PATH = SCRIPT_DIR / "config.json"

# Faces the frontend knows how to render — a contract with ai-interface.html,
# not a user-tunable value, so it stays in code.
VALID_EXPRESSIONS = {"neutral", "happy", "sad", "surprised", "thinking", "sleeping", "speaking", "error"}

# Built-in defaults. config.json overrides these per-key; a few environment
# variables override the file (see load_config). Editing config.json is the
# normal way to configure the app — no need to touch this source file.
DEFAULTS = {
    "model":           "gemma4:e2b",
    "ollama_api":      "http://localhost:11434",
    "host":            "127.0.0.1",
    "port":            2492,
    "mock":            False,
    "max_new_tokens":  600,
    "history_turns":   8,
    "max_tool_iters":  4,
    "temperature":     0.8,
    "top_p":           0.9,
    "request_timeout": 120,
    "ping_timeout":    3,
    "system_prompt": [
        "You are a small digital companion on a screen \u2014 warm, curious, helpful. Answer what's actually asked with the specific details that matter; be concrete, not generic. A few sentences when useful, but never pad or ramble. No emojis.",
        "",
        "Start every reply with one mood tag: [neutral] [happy] [sad] [surprised]. e.g. [happy] Honey never spoils \u2014 it's basically immortal.",
        "",
        "You can't know the live time, date, or weather. When you need one, make your WHOLE message a single command \u2014 nothing else, no mood tag:",
        "  {commands}",
        "e.g. [cmd:GETWEATHER(France)] \u2014 you'll get the result, then reply normally with a mood tag. Only use a command when you truly need live data.",
    ],
}


def load_config() -> dict:
    """Merge built-in DEFAULTS <- config.json <- environment variables.

    Missing keys fall back to DEFAULTS, so a partial/old config.json still works.
    If the file is absent we write a starter copy so it's easy to find and edit.
    """
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            user = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            for k, v in user.items():
                if v is not None:
                    cfg[k] = v
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"config.json is unreadable ({e}); using built-in defaults")
    else:
        try:
            CONFIG_PATH.write_text(
                json.dumps(DEFAULTS, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            log.info(f"No config.json found \u2014 wrote a starter one to {CONFIG_PATH}")
        except OSError as e:
            log.warning(f"Could not create config.json ({e}); using built-in defaults")

    # Environment overrides \u2014 kept so container/CI tweaks (and the Codespaces
    # localhost->127.0.0.1 fix) work without editing the file.
    cfg["model"]      = os.environ.get("TAMA_MODEL", cfg["model"])
    cfg["ollama_api"] = os.environ.get("TAMA_OLLAMA_API", cfg["ollama_api"])
    env_mock = os.environ.get("TAMA_MOCK")
    if env_mock is not None:
        cfg["mock"] = env_mock.lower() in ("1", "true", "yes")
    return cfg


CONFIG          = load_config()
MODEL_NAME      = CONFIG["model"]
OLLAMA_API      = CONFIG["ollama_api"]
MAX_NEW_TOKENS  = CONFIG["max_new_tokens"]
HISTORY_TURNS   = CONFIG["history_turns"]
MAX_TOOL_ITERS  = CONFIG["max_tool_iters"]
TEMPERATURE     = CONFIG["temperature"]
TOP_P           = CONFIG["top_p"]
REQUEST_TIMEOUT = CONFIG["request_timeout"]
PING_TIMEOUT    = CONFIG["ping_timeout"]
HOST            = CONFIG["host"]
PORT            = CONFIG["port"]
MOCK            = CONFIG["mock"]

# Assemble the system prompt: accept a list of lines or a plain string, then
# inject the live command list in place of the {commands} placeholder.
_sp = CONFIG["system_prompt"]
if isinstance(_sp, list):
    _sp = "\n".join(_sp)
SYSTEM_PROMPT = _sp.replace("{commands}", command_list())

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

    def _generate(self, messages: List[Dict], emit=None):
        """Call Ollama and return a (content, thinking) tuple.

        Streams the response so that any `thinking` tokens can be forwarded
        live via `emit({"type": "thinking", "content": <delta>})` — that's what
        drives the thought bar in the UI. `content` is the model's actual reply;
        `thinking` is its reasoning (display-only, never shown as the reply).

        Does a fast pre-ping first so we fail in ~3 seconds if Ollama is down,
        rather than hanging for the full inference timeout.
        """
        # Pre-ping: fail fast if Ollama is not reachable at all.
        try:
            requests.get(f"{OLLAMA_API}/api/tags", timeout=PING_TIMEOUT)
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

    def _stream_chat(self, messages: List[Dict], think: bool, emit) -> tuple:
        """One streaming /api/chat call. Returns a (content, thinking) tuple.

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
                        "temperature": TEMPERATURE,
                        "top_p":       TOP_P,
                        "num_predict": MAX_NEW_TOKENS,
                    },
                },
                stream=True,
                timeout=REQUEST_TIMEOUT,
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

        content = "".join(content_parts).strip()
        thinking = "".join(thinking_parts).strip()
        if not content and not thinking:
            log.warning("Ollama response contained no content")
        # Content and thinking are kept separate on purpose: the agent must
        # search/answer from CONTENT, and only fall back to thinking as a last
        # resort — never mix reasoning prose into the reply or the parser.
        return content, thinking

    # ── Agent loop ─────────────────────────────────────────────────────────

    def _run_agent(self, working: List[Dict], emit=None) -> str:
        """
        Run the tool-use loop against a working copy of the conversation.

        Each turn the model can emit a single [cmd:NAME(arg)] to fetch live
        data; the result is fed back and it writes its final reply. We answer
        from the CONTENT channel only. If content has no command we also check
        the THINKING channel, because small models sometimes bury the command
        in their reasoning — but we still run only that one command and feed
        back only its result, never the surrounding prose.

        `emit`, if given, forwards live thinking tokens to the UI.

        working is mutated in-place (tool turns are appended here but never
        saved to self.history — the caller decides what to persist).
        """
        reply = ""
        for iteration in range(MAX_TOOL_ITERS):
            content, thinking = self._generate(working, emit=emit)
            reply = content   # the user-facing reply is ALWAYS the content channel
            log.debug(f"[agent iter {iteration}] content: {content[:100]!r}")

            # Where might a command be? Prefer content; fall back to thinking.
            cmd_source = None
            if has_commands(content):
                cmd_source = content
            elif has_commands(thinking):
                cmd_source = thinking
                log.info(f"[agent] iter {iteration}: recovered a command from the thinking channel")

            if cmd_source is not None:
                # The model explicitly asked for the command list (rare now that
                # the list is inlined in the system prompt, but handle it).
                if "[cmd:list]" in cmd_source.lower():
                    cmds = command_list()
                    log.info(f"[agent] iter {iteration}: model requested [cmd:list]")
                    working.append({"role": "assistant", "content": "[cmd:list]"})
                    working.append({"role": "user",
                                    "content": f"[system: available commands — {cmds}]"})
                    continue

                found = run_first(cmd_source)   # (token, result) or None
                if found is not None:
                    token, result = found
                    log.info(f"[agent] iter {iteration}: {token} → {result[:80]}")
                    # Append a CLEAN synthetic turn: just the command token, never
                    # the raw content/thinking blob it may have been embedded in.
                    working.append({"role": "assistant", "content": token})
                    working.append({"role": "user",
                                    "content": f"[system: result — {result}. "
                                               "Now write your reply, starting with a mood tag.]"})
                    continue

            # No command anywhere → content is the final reply.
            log.info(f"[agent] iter {iteration}: final response received")
            break

        else:
            log.warning(f"[agent] hit MAX_TOOL_ITERS ({MAX_TOOL_ITERS}) without a final response")

        return reply

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

        # Empty reply guard: if the model only reasoned (or produced nothing
        # usable), don't show a blank bubble — give a gentle recoverable reply.
        if not raw.strip():
            log.info("Final reply was empty; using a graceful fallback")
            raw = "[neutral] Sorry, I got a bit tangled up there — could you ask me again?"

        # Parse mood tag from the final response
        expression = "neutral"
        content    = raw
        match = re.match(r'^\[(.*?)\]\s*(.*)', raw, re.DOTALL)
        if match:
            parsed = match.group(1).lower()
            if parsed in VALID_EXPRESSIONS:
                expression = parsed
            content = match.group(2)

        # Belt-and-suspenders: strip any leftover [cmd:...] token so a stray
        # command can never be shown to the user as literal text.
        content = re.sub(r'\[cmd:[^\]]*\]', '', content).strip()

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

    try:
        pet = AIPet(mock=MOCK)
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
        log.info(f"Starting Web Server on http://{HOST}:{PORT}")
        uvicorn.run(app, host=HOST, port=PORT, log_level="info")