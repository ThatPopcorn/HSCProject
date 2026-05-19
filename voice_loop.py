#!/usr/bin/env python3
"""
voice_loop.py — TTS + STT + button library with barge-in
=========================================================

Library for voice I/O with a physical (or simulated) button. Designed
to drop straight into the AI companion project, but useful for anything
that needs:

  * a process-killable TTS (so interruption is instant)
  * a streaming STT with live partials
  * a single-button input event
  * a conversational state machine that gives the user priority

Public surface
--------------
Classes:
    TTS         — speaks via a subprocess; stop() is an OS-level kill
    STT         — streams the mic into Vosk
    Button      — wraps pynput so callers just think in "press" events
    VoiceLoop   — full state machine: listen → respond → speak, with barge-in

Helpers:
    listen_once(stt, button, on_partial=None) -> Optional[str]
    confirm_with_countdown(button, seconds=3) -> bool

Setup
-----
    pip install vosk pyaudio pynput pyttsx3

    Vosk model (~40 MB) — download a small model from
    https://alphacephei.com/vosk/models and unzip next to this file as
    "vosk-model-small-en-us-0.15" (default expected name).

    Linux/Pi: sudo apt install espeak-ng   (fast native TTS)

Quick start
-----------
    from voice_loop import TTS, STT, Button, VoiceLoop

    def respond(text):
        return f"You said {text}"

    with Button() as button:
        VoiceLoop(TTS(), STT(), respond, button=button).run()

Running this file directly starts a basic conversational demo. See
test_voice.py for a more thorough test menu.
"""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import pyaudio # sudo apt-get install portaudio19-dev, then use pip install pyaudio
import vosk
from pynput import keyboard


# ─── config ───────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).resolve().parent
VOSK_MODEL  = SCRIPT_DIR / "vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000
BUTTON_KEY  = keyboard.Key.space


# ─── TTS ──────────────────────────────────────────────────────────────

def _build_tts_command(text: str) -> list:
    """
    Pick the most native TTS command available so startup is fast and
    .terminate() actually kills the audio. pyttsx3 is the cross-platform
    fallback (works on Windows out of the box) but we still run it as a
    subprocess — the OS process kill is what really stops the speech.
    """
    if sys.platform == "darwin" and shutil.which("say"):
        return ["say", text]
    if sys.platform.startswith("linux"):
        if shutil.which("espeak-ng"):
            return ["espeak-ng", text]
        if shutil.which("espeak"):
            return ["espeak", text]
    code = f"import pyttsx3; e=pyttsx3.init(); e.say({text!r}); e.runAndWait()"
    return [sys.executable, "-c", code]


class TTS:
    """
    Speaks via a subprocess so barge-in is a clean OS-level process kill.
    speak() blocks until either the audio finishes or stop() is called
    from another thread.
    """

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def speak(self, text: str) -> None:
        cmd = _build_tts_command(text)
        with self._lock:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            proc = self._proc
        proc.wait()
        with self._lock:
            if self._proc is proc:
                self._proc = None

    def stop(self) -> None:
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=0.3)
        except subprocess.TimeoutExpired:
            proc.kill()


# ─── STT ──────────────────────────────────────────────────────────────

class STT:
    """
    Streams microphone audio into a Vosk recognizer. listen() blocks and
    returns the final transcript when stop() is called from another thread.
    Partial results are pushed through on_partial so callers can show
    what's been heard so far.
    """

    def __init__(self, model_path: Path = VOSK_MODEL, sample_rate: int = SAMPLE_RATE):
        if not model_path.exists():
            raise FileNotFoundError(
                f"vosk model not found at {model_path}\n"
                "  download a small model from https://alphacephei.com/vosk/models\n"
                "  and unzip it next to this script."
            )
        if not model_path.is_dir():
            raise FileNotFoundError(f"{model_path} exists but is not a directory")

        # Windows' built-in zip extractor often produces a doubly-nested
        # folder (vosk-model-x/vosk-model-x/...). Detect and unwrap it.
        nested = model_path / model_path.name
        if nested.is_dir() and (nested / "am").is_dir():
            model_path = nested

        if not (model_path / "am").is_dir():
            contents = sorted(p.name for p in model_path.iterdir())[:10]
            raise FileNotFoundError(
                f"directory at {model_path} doesn't look like a vosk model\n"
                f"  contents: {contents}\n"
                "  a valid model folder contains subdirectories like 'am', 'conf', 'graph'.\n"
                "  if your zip extracted to a nested folder, move the inner\n"
                "  contents up one level (or re-extract with a tool like 7-Zip)."
            )

        vosk.SetLogLevel(-1)
        self.model = vosk.Model(str(model_path))
        self.sample_rate = sample_rate
        self._audio_q: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

    def _audio_cb(self, in_data, frame_count, time_info, status):
        self._audio_q.put(in_data)
        return (None, pyaudio.paContinue)

    def listen(self, on_partial: Optional[Callable[[str], None]] = None) -> str:
        self._stop_event.clear()
        while not self._audio_q.empty():
            try: self._audio_q.get_nowait()
            except queue.Empty: break

        rec = vosk.KaldiRecognizer(self.model, self.sample_rate)
        chunks: list = []

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=4000,
            stream_callback=self._audio_cb,
        )
        stream.start_stream()
        try:
            while not self._stop_event.is_set():
                try:
                    data = self._audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if rec.AcceptWaveform(data):
                    finished = json.loads(rec.Result()).get("text", "")
                    if finished:
                        chunks.append(finished)
                elif on_partial is not None:
                    partial = json.loads(rec.PartialResult()).get("partial", "")
                    on_partial(partial)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

        tail = json.loads(rec.FinalResult()).get("text", "")
        if tail:
            chunks.append(tail)
        return " ".join(chunks).strip()

    def stop(self) -> None:
        self._stop_event.set()


# ─── Button ───────────────────────────────────────────────────────────

class Button:
    """
    The single button on the device. On a PC it's the spacebar; on the
    Pi this whole class gets replaced with a GPIO interrupt callback
    that calls self._event.set() — nothing else has to change.

    Usage:
        with Button() as b:
            b.wait()                         # block until pressed
            b.wait(timeout=2.0)              # returns False on timeout
            if b.is_set(): ...               # non-blocking check
            b.clear()                        # discard any pending press
    """

    def __init__(self, key=BUTTON_KEY):
        self._key = key
        self._event = threading.Event()
        self._listener: Optional[keyboard.Listener] = None

    def start(self) -> "Button":
        if self._listener is None:
            self._listener = keyboard.Listener(on_press=self._on_key)
            self._listener.start()
        return self

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def __enter__(self) -> "Button":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def _on_key(self, key) -> None:
        if key == self._key:
            self._event.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until pressed (or timeout). Returns True if pressed."""
        pressed = self._event.wait(timeout=timeout)
        if pressed:
            self._event.clear()
        return pressed

    def is_set(self) -> bool:
        return self._event.is_set()

    def clear(self) -> None:
        self._event.clear()


# ─── helpers ──────────────────────────────────────────────────────────

def _default_partial(text: str) -> None:
    sys.stdout.write(f"\r  you > {text:<60}")
    sys.stdout.flush()


def listen_once(
    stt: STT,
    button: Button,
    on_partial: Optional[Callable[[str], None]] = None,
    quiet: bool = False,
) -> Optional[str]:
    """
    Open STT, stream partials, return final transcript when button pressed.
    Pass quiet=True to suppress terminal prints (e.g. when a UI handles display).
    Returns None if nothing was transcribed.
    """
    if on_partial is None and not quiet:
        on_partial = _default_partial

    result: dict = {"text": ""}
    def worker() -> None:
        result["text"] = stt.listen(on_partial=on_partial)

    if not quiet:
        sys.stdout.write("\r  you > (listening — press SPACE to stop)\n")
        sys.stdout.flush()

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    button.wait()
    stt.stop()
    t.join()

    transcript = result["text"]
    if not quiet:
        sys.stdout.write(f"\r  you > {transcript:<60}\n")
        sys.stdout.flush()
    return transcript or None


def confirm_with_countdown(button: Button, seconds: float = 3.0) -> bool:
    """
    Wait up to `seconds` for a button press, redrawing a live countdown.
    Returns True if the user pressed in time, False otherwise.
    """
    button.clear()
    start = time.time()
    while True:
        remaining = seconds - (time.time() - start)
        if remaining <= 0:
            sys.stdout.write("\r  (no confirmation)" + " " * 40 + "\n")
            sys.stdout.flush()
            return False
        if button.wait(timeout=min(0.1, remaining)):
            sys.stdout.write("\r  confirmed" + " " * 40 + "\n")
            sys.stdout.flush()
            return True
        sys.stdout.write(f"\r  press SPACE to confirm ({remaining:.1f}s)  ")
        sys.stdout.flush()


# ─── voice loop ───────────────────────────────────────────────────────

class VoiceLoop:
    """
    Wires TTS, STT, and the button into a state machine. The user owns
    it: a button press always moves the system toward listening.

    If you pass in an existing Button, VoiceLoop won't touch its
    lifecycle — caller is responsible for start()/stop(). Otherwise
    VoiceLoop creates and manages its own.

    Callbacks
    ---------
    on_state(state: str)       — fired on every state transition
    on_word(partial: str)      — fired each word during speaking (word-by-word)
    speak_wpm                  — display speed for on_word callbacks
    """

    IDLE      = "idle"
    LISTENING = "listening"
    THINKING  = "thinking"
    SPEAKING  = "speaking"

    def __init__(
        self,
        tts: TTS,
        stt: STT,
        respond: Callable[[str], str],
        button: Optional[Button] = None,
        speak_wpm: int = 200,
        on_state: Optional[Callable[[str], None]] = None,
        on_word:  Optional[Callable[[str], None]] = None,
    ):
        self.tts       = tts
        self.stt       = stt
        self.respond   = respond
        self._owns_button = button is None
        self.button    = button if button is not None else Button()
        self.state     = self.IDLE
        self.speak_wpm = speak_wpm
        self.on_state  = on_state
        self.on_word   = on_word

    def _set_state(self, state: str) -> None:
        self.state = state
        if self.on_state:
            self.on_state(state)

    def _speak_with_barge_in(self, text: str) -> bool:
        """
        Speak `text` word-by-word at speak_wpm while TTS audio plays in
        parallel. Fires on_word(partial) each step. Any button press
        terminates both the display loop and the TTS subprocess immediately.
        Returns True if interrupted, False if speech completed naturally.
        """
        self._set_state(self.SPEAKING)

        words = text.split()
        delay = 60.0 / self.speak_wpm

        t = threading.Thread(target=self.tts.speak, args=(text,), daemon=True)
        t.start()

        interrupted = False
        for i in range(len(words)):
            if self.button.is_set():
                self.button.clear()
                self.tts.stop()
                t.join()
                interrupted = True
                break
            partial = " ".join(words[: i + 1])
            if self.on_word:
                self.on_word(partial)
            else:
                sys.stdout.write(f"\r  pet > {partial}")
                sys.stdout.flush()
            time.sleep(delay)

        if not interrupted:
            # Hold so the reader can finish, then wait for TTS to complete
            time.sleep(1.5)
            while t.is_alive():
                if self.button.is_set():
                    self.button.clear()
                    self.tts.stop()
                    t.join()
                    interrupted = True
                    break
                time.sleep(0.03)

        if not self.on_word:
            print()  # newline after the \r terminal display

        return interrupted

    def run(self) -> None:
        if self._owns_button:
            self.button.start()

        print("Press SPACE to talk. Press SPACE again to stop or interrupt.")
        print("Ctrl-C to quit.\n")

        try:
            while True:
                self._set_state(self.IDLE)
                self.button.wait()

                # one turn = listen → think → speak.
                # barge-in during speak loops back into listen.
                while True:
                    self._set_state(self.LISTENING)
                    transcript = listen_once(self.stt, self.button,
                                             quiet=self.on_word is not None)
                    if not transcript:
                        break

                    self._set_state(self.THINKING)
                    reply = self.respond(transcript)

                    interrupted = self._speak_with_barge_in(reply)
                    if not interrupted:
                        break
        except KeyboardInterrupt:
            print()
        finally:
            self.tts.stop()
            self.stt.stop()
            if self._owns_button:
                self.button.stop()


# ─── basic demo (run this file directly) ──────────────────────────────

def _demo_respond(text: str) -> str:
    return f"You said: {text}. Try interrupting me — press space any time."


def main() -> None:
    tts = TTS()
    stt = STT()
    with Button() as button:
        VoiceLoop(tts, stt, _demo_respond, button=button).run()


if __name__ == "__main__":
    main()