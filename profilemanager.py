#!/usr/bin/env python3
"""
profile.py — encrypted user profile for the AI companion
=========================================================

Voice-driven setup and unlock of a user profile stored encrypted at rest.

Setup flow (create_profile)
---------------------------
  1. Speak a 3-word password (readback + 3-second countdown to confirm)
  2. Speak a 1-word name (same confirmation pattern)
  3. Speak an age in words — "forty three" → 43 (same pattern)
  4. Derive an AES-256 key from the password using Argon2id
  5. Encrypt {name, age} with AES-GCM, write blob + salt to disk

Unlock flow (unlock_profile)
----------------------------
  1. Speak the password
  2. Derive a key with the stored salt
  3. AES-GCM decrypt — success proves the password was correct

Why no separate password hash?
  The encrypted profile IS the verification target. If decryption
  succeeds, the password was right. Storing a hash alongside would just
  give an attacker a second thing to attack and tells them nothing more
  than a failed decrypt already does.

Dependencies
------------
    pip install cryptography argon2-cffi word2number
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Optional, Tuple

from argon2.low_level import Type, hash_secret_raw
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from word2number import w2n

from voice_loop import (
    Button, STT, TTS,
    confirm_with_countdown, listen_once,
)


# ─── config ───────────────────────────────────────────────────────────

SCRIPT_DIR    = Path(__file__).resolve().parent
PROFILE_PATH  = SCRIPT_DIR / "profile.enc.json"
PASSWORD_LEN  = 3              # words in the password
CONFIRM_SECS  = 3.0            # countdown duration for confirmations

# Argon2id parameters. memory_cost is in KiB. 64 MiB is comfortable on a
# 4 GB Pi 5 and takes ~0.5–1 s per derivation on modern hardware — a
# noticeable delay during unlock, which is exactly what you want
# (slows brute force, imperceptible during legitimate use).
ARGON2_TIME        = 3
ARGON2_MEMORY      = 65536     # 64 MiB
ARGON2_PARALLELISM = 4
KEY_LEN            = 32        # AES-256
SALT_LEN           = 16
NONCE_LEN          = 12


# ─── crypto primitives ────────────────────────────────────────────────

def _derive_key(password: str, salt: bytes, kdf_params: Optional[dict] = None) -> bytes:
    """Derive a 256-bit key from password + salt using Argon2id."""
    p = kdf_params or {}
    return hash_secret_raw(
        secret=password.encode("utf-8"),
        salt=salt,
        time_cost=p.get("time", ARGON2_TIME),
        memory_cost=p.get("memory", ARGON2_MEMORY),
        parallelism=p.get("parallelism", ARGON2_PARALLELISM),
        hash_len=KEY_LEN,
        type=Type.ID,
    )


def encrypt_profile(profile: dict, password: str) -> dict:
    """Encrypt profile data with a password-derived key. Returns the blob to save."""
    salt = os.urandom(SALT_LEN)
    nonce = os.urandom(NONCE_LEN)
    key = _derive_key(password, salt)
    plaintext = json.dumps(profile).encode("utf-8")
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)
    return {
        "version": 1,
        "kdf": {
            "type": "argon2id",
            "time": ARGON2_TIME,
            "memory": ARGON2_MEMORY,
            "parallelism": ARGON2_PARALLELISM,
        },
        "salt": salt.hex(),
        "nonce": nonce.hex(),
        "ciphertext": ciphertext.hex(),
    }


def decrypt_profile(blob: dict, password: str) -> Optional[dict]:
    """Try to decrypt. Returns the profile dict, or None if the password was wrong."""
    salt = bytes.fromhex(blob["salt"])
    nonce = bytes.fromhex(blob["nonce"])
    ciphertext = bytes.fromhex(blob["ciphertext"])
    key = _derive_key(password, salt, blob.get("kdf"))
    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
    except Exception:
        return None
    return json.loads(plaintext.decode("utf-8"))


def save_profile(profile: dict, password: str, path: Path = PROFILE_PATH) -> None:
    blob = encrypt_profile(profile, password)
    path.write_text(json.dumps(blob, indent=2))


def load_profile(password: str, path: Path = PROFILE_PATH) -> Optional[dict]:
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    return decrypt_profile(blob, password)


# ─── parsing helpers ──────────────────────────────────────────────────

def parse_age(text: str) -> Optional[int]:
    """
    Convert spoken age text into an int. Returns None if it can't be
    parsed or the result is out of a plausible range.
    """
    if not text:
        return None
    cleaned = text.strip().lower().replace("-", " ")
    try:
        n = w2n.word_to_num(cleaned)
    except (ValueError, IndexError):
        return None
    if 0 < n < 150:
        return n
    return None


# ─── voice flow helpers ───────────────────────────────────────────────

def _say(tts: TTS, text: str) -> None:
    """Print + speak. One helper so the terminal trace stays consistent."""
    print(f"  pet > {text}")
    tts.speak(text)


# A validator takes the raw STT transcript and returns
#   (accepted: bool, message_if_rejected: str, value_if_accepted: Any)
Validator = Callable[[str], Tuple[bool, str, object]]


def _get_confirmed_input(
    tts: TTS,
    stt: STT,
    button: Button,
    prompt: str,
    *,
    readback_fmt: str,
    validate: Validator,
):
    """
    Ask the user for input via STT, then read back what was heard and
    require a button press within CONFIRM_SECS to accept. If validation
    fails or confirmation times out, loop and try again until accepted.
    """
    while True:
        _say(tts, prompt)
        _say(tts, "Press the button when you're ready to speak.")
        button.clear()
        button.wait()

        transcript = listen_once(stt, button) or ""

        ok, message, value = validate(transcript)
        if not ok:
            _say(tts, message or "I didn't catch that. Let's try again.")
            continue

        _say(
            tts,
            readback_fmt.format(value=value)
            + f" Press the button within {int(CONFIRM_SECS)} seconds to confirm.",
        )
        if confirm_with_countdown(button, CONFIRM_SECS):
            return value
        _say(tts, "Not confirmed. Let's try again.")


# ─── validators ───────────────────────────────────────────────────────

def _validate_password(text: str):
    words = text.split()
    if len(words) < PASSWORD_LEN:
        return (
            False,
            f"I needed {PASSWORD_LEN} words but heard {len(words)}. Let's try again.",
            None,
        )
    return True, "", " ".join(words[:PASSWORD_LEN])


def _validate_name(text: str):
    words = text.split()
    if not words:
        return False, "I didn't hear a name. Let's try again.", None
    return True, "", words[0].capitalize()


def _validate_age(text: str):
    age = parse_age(text)
    if age is None:
        return (
            False,
            "I couldn't understand that as an age. Try saying it like 'twenty five'.",
            None,
        )
    return True, "", age


# ─── public voice flows ───────────────────────────────────────────────

def create_profile(
    tts: TTS,
    stt: STT,
    button: Button,
    path: Path = PROFILE_PATH,
) -> dict:
    """
    Walk the user through first-time profile setup. Returns the profile
    dict that was saved. The password is NOT included in the return value.
    """
    _say(tts, "Welcome, new user. Let's set up your profile.")

    password = _get_confirmed_input(
        tts, stt, button,
        prompt=f"Choose a password of {PASSWORD_LEN} words you'll remember.",
        readback_fmt="I heard the words: {value}.",
        validate=_validate_password,
    )

    name = _get_confirmed_input(
        tts, stt, button,
        prompt="What's your name? One word, please.",
        readback_fmt="I heard your name as {value}.",
        validate=_validate_name,
    )

    age = _get_confirmed_input(
        tts, stt, button,
        prompt="How old are you? Say your age in words, like 'twenty five'.",
        readback_fmt="I heard {value}.",
        validate=_validate_age,
    )

    profile = {"name": name, "age": age}
    save_profile(profile, password, path)
    _say(tts, f"Profile saved. Welcome, {name}.")
    return profile


def unlock_profile(
    tts: TTS,
    stt: STT,
    button: Button,
    path: Path = PROFILE_PATH,
    max_attempts: int = 3,
) -> Optional[dict]:
    """
    Prompt for the password and try to decrypt the profile. Returns the
    profile dict on success, None if all attempts fail or no profile exists.
    """
    if not path.exists():
        _say(tts, "No profile found. Please run setup first.")
        return None

    blob = json.loads(path.read_text())

    for attempt in range(max_attempts):
        _say(
            tts,
            f"Say your {PASSWORD_LEN}-word password. Press the button when ready.",
        )
        button.clear()
        button.wait()

        transcript = listen_once(stt, button) or ""
        words = transcript.split()
        if len(words) < PASSWORD_LEN:
            _say(tts, f"I only heard {len(words)} words.")
            remaining = max_attempts - attempt - 1
            if remaining > 0:
                _say(tts, f"{remaining} attempts left.")
            continue

        password = " ".join(words[:PASSWORD_LEN])
        profile = decrypt_profile(blob, password)
        if profile is not None:
            _say(tts, f"Welcome back, {profile['name']}.")
            return profile

        remaining = max_attempts - attempt - 1
        if remaining > 0:
            _say(tts, f"That password didn't work. {remaining} attempts left.")
        else:
            _say(tts, "Out of attempts.")

    return None