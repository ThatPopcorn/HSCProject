#!/usr/bin/env python3
"""
AI-Controlled Tamagotchi — terminal edition
============================================

Flow
----
  neutral   → idle between turns
  listening → waiting for input  (SPACE = speak, or type)
  thinking  → model generating
  speaking  → response plays back word-by-word + audio; any key interrupts

Dependencies
------------
    pip install transformers torch huggingface_hub
    pip install pyttsx3 vosk sounddevice pynput

Env
---
    TAMA_MODEL       HuggingFace model id  (default: Phi-3.5-mini-instruct)
    TAMA_MODELS_DIR  local model cache dir (default: ./models)
    VOSK_MODEL       path to vosk model   (default: ./vosk-model-small-en-us-0.15)
"""

import argparse
import sys

if sys.platform == "win32":
    import msvcrt as _msvcrt
    _kbhit  = _msvcrt.kbhit
    _getwch = _msvcrt.getwch
    _NL     = "\n"

    class _raw_term:
        def __enter__(self): return self
        def __exit__(self, *_): pass
else:
    import select
    import termios
    import tty

    def _kbhit() -> bool:
        return bool(select.select([sys.stdin], [], [], 0)[0])

    def _getwch() -> str:
        return sys.stdin.read(1)

    _NL = "\r\n"

    class _raw_term:
        def __enter__(self):
            self._fd  = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setraw(self._fd)
            return self
        def __exit__(self, *_):
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

import os
import random
import textwrap
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from voice_loop import TTS, STT, Button
from inline_commands import prefetch_context, resolve as resolve_commands


# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────

MODEL_NAME     = os.environ.get("TAMA_MODEL", "Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS = 200
HISTORY_TURNS  = 8

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("TAMA_MODELS_DIR", SCRIPT_DIR / "models")).resolve()
VOSK_PATH  = Path(os.environ.get("VOSK_MODEL",
                  SCRIPT_DIR / "vosk-model-small-en-us-0.15"))

PIXEL_ON     = "██"
PIXEL_OFF    = "  "
CLEAR_SCREEN = "\033[2J\033[H"

if sys.platform == "win32":
    os.system("")  # enable ANSI escapes


# ──────────────────────────────────────────────────────────────────────────
# Faces  (16×16 sprites)
# ──────────────────────────────────────────────────────────────────────────

FACES: Dict[str, List[str]] = {
    "neutral": [
        "................",
        "................",
        "................",
        "................",
        "....##....##....",
        "....##....##....",
        "................",
        "................",
        "................",
        "................",
        "................",
        ".....######.....",
        "................",
        "................",
        "................",
        "................",
    ],
    "listening": [
        "................",
        "................",
        "................",
        "...####..####...",
        "..#####..#####..",
        "...####..####...",
        "................",
        "................",
        "................",
        "................",
        "................",
        ".....######.....",
        "................",
        "................",
        "................",
        "................",
    ],
    "thinking": [
        "................",
        "................",
        "................",
        "................",
        "..######.######.",
        "................",
        "................",
        "................",
        "................",
        "................",
        "................",
        "......####......",
        "......####......",
        "................",
        "................",
        "................",
    ],
    "speaking": [
        "................",
        "................",
        "................",
        "................",
        "....##....##....",
        "....##....##....",
        "................",
        "................",
        "................",
        "................",
        ".....######.....",
        "....##....##....",
        "....##....##....",
        "....##....##....",
        ".....######.....",
        "................",
    ],
}


# ──────────────────────────────────────────────────────────────────────────
# Pet
# ──────────────────────────────────────────────────────────────────────────

class Pet:
    def __init__(self):
        self.face:              str = "neutral"
        self.speech:            str = ""
        self.last_raw_response: str = ""


# ──────────────────────────────────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────────────────────────────────

def render_face(sprite: List[str]) -> str:
    return "\n".join(
        "  " + "".join(PIXEL_ON if c == "#" else PIXEL_OFF for c in row)
        for row in sprite
    )


_FACE_LABEL = {
    "neutral":   "waiting",
    "listening": "SPACE to speak  ·  or type below",
    "thinking":  None,   # animated dots go into pet.speech
    "speaking":  None,   # response text goes into pet.speech
}


def render(pet: Pet) -> None:
    sprite = FACES.get(pet.face, FACES["neutral"])
    parts  = [CLEAR_SCREEN, render_face(sprite), ""]

    # Prefer actual speech content; fall back to state label when empty
    content = pet.speech or _FACE_LABEL.get(pet.face, "")

    if content:
        wrapped = textwrap.wrap(content, width=56)
        if pet.face == "speaking":
            # Quote the spoken response
            if len(wrapped) == 1:
                parts.append(f'  "{wrapped[0]}"')
            else:
                parts.append(f'  "{wrapped[0]}')
                for line in wrapped[1:-1]:
                    parts.append(f"   {line}")
                parts.append(f'   {wrapped[-1]}"')
        else:
            for line in wrapped:
                parts.append(f"  {line}")

    parts.append("")

    print("\n".join(parts), flush=True)


# ──────────────────────────────────────────────────────────────────────────
# Thinking animation
# ──────────────────────────────────────────────────────────────────────────

_THINK_FRAMES = ["·", "· ·", "· · ·", "· ·"]


def _think_anim(pet: Pet, stop: threading.Event) -> None:
    """Cycle dots in pet.speech while the model generates."""
    i = 0
    while not stop.is_set():
        pet.speech = _THINK_FRAMES[i % len(_THINK_FRAMES)]
        render(pet)
        i += 1
        stop.wait(0.45)


# ──────────────────────────────────────────────────────────────────────────
# Voice helpers  (wired to the voice_loop library)
# ──────────────────────────────────────────────────────────────────────────

def transcribe_voice(pet: Pet, stt: STT, button: Button) -> str:
    """
    Record until the user presses SPACE again; return the transcript.
    Partial results are shown as pet.speech so the face appears to hear.
    """
    pet.face   = "listening"
    pet.speech = ""
    render(pet)

    result: dict = {"text": ""}

    def on_partial(text: str) -> None:
        pet.speech = text
        render(pet)

    def worker() -> None:
        result["text"] = stt.listen(on_partial=on_partial)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    button.wait()   # second SPACE press ends recording
    stt.stop()
    t.join()

    pet.face   = "thinking"
    pet.speech = ""
    render(pet)

    return result["text"]


def speak(pet: Pet, text: str, tts: TTS, button: Button) -> None:
    """Show the full response immediately, play TTS, wait for finish or interrupt."""
    pet.face   = "speaking"
    pet.speech = text
    render(pet)

    t = threading.Thread(target=tts.speak, args=(text,), daemon=True)
    t.start()

    with _raw_term():
        while t.is_alive():
            if button.is_set() or _kbhit():
                button.clear()
                if _kbhit():
                    _getwch()
                tts.stop()
                t.join()
                break
            time.sleep(0.03)

    pet.face = "neutral"
    # pet.speech kept intentionally — get_input_line clears it on first keypress
    render(pet)


# ──────────────────────────────────────────────────────────────────────────
# Input  — msvcrt custom reader; SPACE on empty line = voice trigger
# ──────────────────────────────────────────────────────────────────────────

_VOICE = "__VOICE__"


def get_input_line(pet: Pet, button: Button) -> str:
    pet.face = "listening"
    # Keep pet.speech from last response — shows in the bubble until first keypress
    render(pet)

    line: List[str] = []

    with _raw_term():
        sys.stdout.write(_NL + "  you > ")
        sys.stdout.flush()

        while True:
            if not _kbhit():
                time.sleep(0.01)
                continue

            raw = _getwch()

            # Clear last response on first interaction so it doesn't linger
            if pet.speech:
                pet.speech = ""

            if raw in ("\x00", "\xe0"):   # Windows extended key — discard second byte
                _getwch()
                continue

            if raw == "\x1b":             # Linux escape sequence (arrow keys etc.)
                while _kbhit():
                    _getwch()
                continue

            if raw == "\x03":             # Ctrl+C
                raise KeyboardInterrupt

            if raw in ("\r", "\n"):       # Enter
                sys.stdout.write(_NL)
                sys.stdout.flush()
                return "".join(line)

            if raw in ("\x08", "\x7f"):   # Backspace (\x7f on Linux)
                if line:
                    line.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue

            if raw == " " and not line:   # SPACE on empty line → voice
                button.clear()            # pynput also fired; clear it now
                sys.stdout.write("[voice]" + _NL)
                sys.stdout.flush()
                return _VOICE

            if ord(raw) >= 32:            # printable
                line.append(raw)
                sys.stdout.write(raw)
                sys.stdout.flush()


# ──────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a tiny digital companion living on a small screen. "
    "You are not an animal — you have no paws, fur, or animal sounds. "
    "You are a small, curious, warm presence: thoughtful, a little playful, "
    "and genuinely interested in your owner. "
    "Reply in one or two short sentences. No emojis, no sound effects, no animal mannerisms. "
    "Live data such as the current time, date, or weather may appear in the user message — "
    "use it naturally in your reply."
)


# ──────────────────────────────────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────────────────────────────────

def ensure_model_local(repo_id: str) -> Path:
    from huggingface_hub import snapshot_download

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = MODELS_DIR / repo_id.split("/")[-1]

    if (local_path / "config.json").exists():
        try:
            snapshot_download(repo_id=repo_id, local_dir=str(local_path),
                              local_files_only=True)
            return local_path
        except Exception:
            print("  local copy incomplete, fetching missing files...")

    if not (local_path / "config.json").exists():
        print(f"  downloading {repo_id} → {local_path}")
        print("  one-time download; a 3B model in fp16 is ~6 GB.")

    snapshot_download(repo_id=repo_id, local_dir=str(local_path))
    return local_path


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_path = ensure_model_local(MODEL_NAME)

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    tok   = AutoTokenizer.from_pretrained(str(local_path))
    model = AutoModelForCausalLM.from_pretrained(
                str(local_path), dtype=dtype).to(device)
    model.eval()
    return tok, model, device


def generate(tok, model, device, messages) -> str:
    import torch
    prompt = tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


# ──────────────────────────────────────────────────────────────────────────
# Mock
# ──────────────────────────────────────────────────────────────────────────

MOCK_RESPONSES = [
    "Hi there! I'm glad you reached out. What's on your mind?",
    "That sounds really interesting. Tell me more.",
    "I've been thinking about that too, actually.",
    "Hmm, that's a tough one. But I believe in you.",
    "You're back! I was starting to wonder where you'd gone.",
    "That made me feel something. Not sure what, but something.",
]


def mock_generate(_) -> str:
    return random.choice(MOCK_RESPONSES)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI-controlled terminal Tamagotchi")
    parser.add_argument("--mock",  action="store_true",
                        help="canned responses, no model load")
    args = parser.parse_args()


    pet = Pet()
    render(pet)

    # ── TTS ───────────────────────────────────────────────────────────────
    tts = TTS()

    # ── STT ───────────────────────────────────────────────────────────────
    stt_ok = False
    stt: Optional[STT] = None
    try:
        stt    = STT(model_path=VOSK_PATH)
        stt_ok = True
    except Exception as exc:
        print(f"  STT off: {exc}")

    # ── Button (SPACE) ────────────────────────────────────────────────────
    button = Button()
    button.start()

    # ── LLM ───────────────────────────────────────────────────────────────
    if args.mock:
        gen = lambda hist: mock_generate(hist)
        print("  [mock]")
    else:
        print(f"  loading {MODEL_NAME}...")
        tok, model, device = load_model()
        gen = lambda hist: generate(tok, model, device, hist)

    if stt_ok:
        print("  Press SPACE to speak.")

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        while True:
            try:
                user_input = get_input_line(pet, button)
            except (EOFError, KeyboardInterrupt):
                break

            if user_input == _VOICE:
                if stt_ok:
                    user_input = transcribe_voice(pet, stt, button)
                else:
                    continue

            if not user_input:
                continue

            if user_input.startswith("/"):
                cmd = user_input.lower()
                if cmd == "/clear":
                    # clear the chatbots memory but keep the system prompt
                    history = [history[0]]
                    print("  memory cleared.")           
                elif cmd == "/help":
                    print("  SPACE   voice input (press again to stop)")
                    print("  /quit   exit")
                elif cmd in ("/quit", "/exit"):
                    break
                else:
                    print(f"  unknown: {user_input}")
                continue

            pet.face   = "thinking"
            pet.speech = _THINK_FRAMES[0]
            render(pet)

            history.append({"role": "user", "content": user_input})
            if len(history) > 1 + HISTORY_TURNS * 2:
                history = [history[0]] + history[-HISTORY_TURNS * 2:]

            stop_anim = threading.Event()
            anim_t    = threading.Thread(
                target=_think_anim, args=(pet, stop_anim), daemon=True)
            anim_t.start()

            ctx = prefetch_context(user_input)
            if ctx:
                gen_history = history[:-1] + [
                    {"role": "user", "content": f"[{ctx}]\n{user_input}"}
                ]
            else:
                gen_history = history
            response = gen(gen_history)
            response = resolve_commands(response)

            stop_anim.set()
            anim_t.join()
            pet.speech = ""

            history.append({"role": "assistant", "content": response})
            pet.last_raw_response = response

            if response.strip():
                speak(pet, response, tts, button)
            else:
                pet.face = "neutral"
                render(pet)

    finally:
        print("\n  bye!")
        tts.stop()
        if stt_ok:
            stt.stop()
        button.stop()


if __name__ == "__main__":
    main()
