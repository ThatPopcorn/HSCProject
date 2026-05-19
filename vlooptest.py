#!/usr/bin/env python3
"""
test_voice.py — interactive test menu for the voice loop

Three tests, run in isolation:

  1. TTS         — play a few phrases so you can judge voice quality
  2. Barge-in    — talk to a stub echo responder, interrupt at will
  3. Profile     — full 3-word-password setup, then verify by unlocking

Setup
-----
    pip install vosk sounddevice pynput pyttsx3 cryptography argon2-cffi word2number

    (See voice_loop.py for the Vosk model download.)

Usage
-----
    python test_voice.py
"""

import time

from voice_loop import TTS, STT, Button, VoiceLoop
from profilemanager import PROFILE_PATH, create_profile, unlock_profile


# ─── tests ────────────────────────────────────────────────────────────

def test_tts(tts: TTS, stt: STT, button: Button) -> None:
    """Play several phrases so you can listen to the voice."""
    phrases = [
        "Hello, this is a quick test of the text to speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "If you can hear me clearly, the voice synthesis is working.",
        "Numbers like one, two, three, and forty two.",
    ]
    print()
    print("  TTS test — playing four phrases. Listen for clarity.\n")
    for i, phrase in enumerate(phrases, 1):
        print(f"  ({i}/{len(phrases)}) pet > {phrase}")
        tts.speak(phrase)
        time.sleep(0.3)
    print("  done.\n")


def test_barge_in(tts: TTS, stt: STT, button: Button) -> None:
    """Echo conversation loop. Press SPACE to interrupt the pet mid-speech."""
    def echo(text: str) -> str:
        return (
            f"You said: {text}. "
            "I'm going to keep talking now so you have plenty of time "
            "to press the space bar and interrupt me. The user always "
            "wins. Try pressing space now if you can hear this part of "
            "the sentence."
        )

    print()
    print("  barge-in test:")
    print("    press SPACE to start talking")
    print("    press SPACE again to submit")
    print("    while pet is speaking, press SPACE to interrupt")
    print("    ctrl-c to return to the menu")
    print()
    VoiceLoop(tts, stt, echo, button=button).run()
    print()


def test_profile(tts: TTS, stt: STT, button: Button) -> None:
    """Run setup if no profile exists; otherwise verify unlock with the existing one."""
    print()
    if PROFILE_PATH.exists():
        print(f"  profile already exists at {PROFILE_PATH.name}")
        print("  testing unlock (delete the file manually to re-test setup)...\n")
        result = unlock_profile(tts, stt, button)
        if result is not None:
            print(f"\n  unlock OK: {result}\n")
        else:
            print("\n  unlock failed.\n")
        return

    print("  no profile yet — running setup...\n")
    profile = create_profile(tts, stt, button)
    print(f"\n  setup complete: {profile}")
    print(f"  encrypted blob written to {PROFILE_PATH.name}\n")
    print("  pick option 3 again to test unlock with your new password.\n")


# ─── menu ─────────────────────────────────────────────────────────────

TESTS = {
    "1": ("TTS — listen to the voice",            test_tts),
    "2": ("Barge-in — conversation + interrupt",  test_barge_in),
    "3": ("Profile — setup or unlock",            test_profile),
}


def main() -> None:
    print("loading TTS and STT...")
    tts = TTS()
    stt = STT()
    print("ready.\n")

    with Button() as button:
        try:
            while True:
                print("voice loop tests")
                for key, (label, _) in TESTS.items():
                    print(f"  {key}. {label}")
                print("  q. quit")
                try:
                    choice = input("  > ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                if choice in ("q", "quit", "exit"):
                    break
                if choice not in TESTS:
                    print(f"  unknown choice: {choice!r}\n")
                    continue

                _, fn = TESTS[choice]
                # discard any spurious spacebar from menu navigation
                button.clear()
                try:
                    fn(tts, stt, button)
                except KeyboardInterrupt:
                    print("\n  (interrupted, back to menu)\n")
        finally:
            tts.stop()
            stt.stop()

    print("bye.")


if __name__ == "__main__":
    main()