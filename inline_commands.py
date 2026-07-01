"""
Live-data command module for the Tamagotchi bot.

Commands are registered in _REGISTRY and invoked by the agent loop in main.py.

The model emits [cmd:list] when it needs live data.
The system returns the command list, the model picks one and emits e.g. [cmd:GETWEATHER(Sydney)].
The system executes it, returns the result, and the model writes its final reply.

Public API:
  command_list() -> str          compact list of all commands for the model
  has_commands(text) -> bool     True if text contains any [cmd:...] token
  resolve(text) -> str           replace all [cmd:...] tokens with their results
"""

import datetime
import re
import urllib.parse
import urllib.request
from typing import Callable, Dict, Optional, Tuple

_REGISTRY: Dict[str, Tuple[Callable, str]] = {}


def _reg(name: str, desc: str):
    def deco(fn: Callable) -> Callable:
        _REGISTRY[name.upper()] = (fn, desc)
        return fn
    return deco


# ── Timezone map ────────────────────────────────────────────────────────────

_CITY_TZ: Dict[str, str] = {
    "SYDNEY":         "Australia/Sydney",
    "MELBOURNE":      "Australia/Melbourne",
    "BRISBANE":       "Australia/Brisbane",
    "ADELAIDE":       "Australia/Adelaide",
    "PERTH":          "Australia/Perth",
    "LONDON":         "Europe/London",
    "PARIS":          "Europe/Paris",
    "BERLIN":         "Europe/Berlin",
    "ROME":           "Europe/Rome",
    "AMSTERDAM":      "Europe/Amsterdam",
    "MOSCOW":         "Europe/Moscow",
    "TOKYO":          "Asia/Tokyo",
    "SEOUL":          "Asia/Seoul",
    "BEIJING":        "Asia/Shanghai",
    "SHANGHAI":       "Asia/Shanghai",
    "HONG KONG":      "Asia/Hong_Kong",
    "SINGAPORE":      "Asia/Singapore",
    "DUBAI":          "Asia/Dubai",
    "MUMBAI":         "Asia/Kolkata",
    "DELHI":          "Asia/Kolkata",
    "BANGKOK":        "Asia/Bangkok",
    "NEW YORK":       "America/New_York",
    "LOS ANGELES":    "America/Los_Angeles",
    "CHICAGO":        "America/Chicago",
    "TORONTO":        "America/Toronto",
    "SAO PAULO":      "America/Sao_Paulo",
    "BUENOS AIRES":   "America/Argentina/Buenos_Aires",
    "MEXICO CITY":    "America/Mexico_City",
    "JOHANNESBURG":   "Africa/Johannesburg",
    "CAIRO":          "Africa/Cairo",
    "NAIROBI":        "Africa/Nairobi",
}


def _city_key(raw: str) -> str:
    return raw.strip().upper().replace("_", " ")


def _fmt_time(dt: datetime.datetime) -> str:
    hour   = dt.hour % 12 or 12
    minute = dt.strftime("%M")
    ampm   = "AM" if dt.hour < 12 else "PM"
    return f"{hour}:{minute} {ampm}"


# ── Commands ────────────────────────────────────────────────────────────────

@_reg("GETTIME", "GETTIME or GETTIME(city) — local time or time in a named city")
def _gettime(arg: Optional[str] = None) -> str:
    if not arg:
        return _fmt_time(datetime.datetime.now())
    key     = _city_key(arg)
    tz_name = _CITY_TZ.get(key)
    if not tz_name:
        return f"(city not found: {arg})"
    try:
        from zoneinfo import ZoneInfo
        return _fmt_time(datetime.datetime.now(ZoneInfo(tz_name)))
    except ImportError:
        return "(install tzdata: pip install tzdata)"
    except Exception:
        return "(time unavailable)"


@_reg("GETDATE", "GETDATE — today's full date")
def _getdate(arg: Optional[str] = None) -> str:
    today = datetime.date.today()
    return today.strftime(f"%A, %B {today.day} %Y")


@_reg("GETWEATHER", "GETWEATHER(city) — current weather conditions for a city")
def _getweather(arg: Optional[str] = None) -> str:
    if not arg:
        return "(specify a city)"
    try:
        url = f"https://wttr.in/{urllib.parse.quote(arg.strip())}?format=%C,+%t"
        req = urllib.request.Request(url, headers={"User-Agent": "curl/7"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.read().decode().strip()
    except Exception:
        return "(weather unavailable)"


@_reg("GETDAY", "GETDAY — current day of the week")
def _getday(arg: Optional[str] = None) -> str:
    return datetime.date.today().strftime("%A")


# ── Public API ──────────────────────────────────────────────────────────────

def command_list() -> str:
    """Compact command reference to send to the model when it emits [cmd:list]."""
    return " | ".join(desc for _, (_, desc) in sorted(_REGISTRY.items()))


_CMD_RE = re.compile(r'\[cmd:([A-Za-z_]+)(?:\(([^)]*)\))?\]', re.IGNORECASE)


def has_commands(text: str) -> bool:
    """True if text contains any [cmd:...] token (use to detect tool calls)."""
    return bool(_CMD_RE.search(text))


def resolve(text: str) -> str:
    """Replace every [cmd:NAME] / [cmd:NAME(arg)] token in text with its result.

    NOTE: this rewrites the whole string. The agent loop should prefer
    run_first(), which returns only a single command's result — feeding a
    whole rewritten blob back to a small model tends to confuse it.
    """
    def _sub(m: re.Match) -> str:
        name  = m.group(1).upper()
        arg   = m.group(2)
        entry = _REGISTRY.get(name)
        if entry is None:
            return f"(unknown command: {name})"
        fn, _ = entry
        return fn(arg)
    return _CMD_RE.sub(_sub, text)


def run_first(text: str):
    """Execute ONLY the first [cmd:...] token found in text.

    Returns a (token, result) tuple — the literal token as matched, and just
    that command's result string — or None if no command token is present.
    Crucially this does NOT return the surrounding text, so reasoning prose
    that happens to contain a command can't leak back into the conversation.
    """
    m = _CMD_RE.search(text)
    if m is None:
        return None
    token = m.group(0)
    name  = m.group(1).upper()
    arg   = m.group(2)
    entry = _REGISTRY.get(name)
    if entry is None:
        return (token, f"(unknown command: {name})")
    fn, _ = entry
    return (token, fn(arg))


# A leading mood tag like "[happy] " that a bare command might be prefixed with.
_MOOD_RE = re.compile(r'^\s*\[[a-z]+\]\s*', re.IGNORECASE)


def detect_command(text: str):
    """Find a command in text and run it, returning (token, result) or None.

    Handles two forms:
      1. The strict  [cmd:NAME(arg)]  token anywhere in the text (via run_first).
      2. A BARE invocation — e.g. "GETWEATHER(Sydney)", "GETTIME", "GETWEATHER Sydney",
         or "cmd:GETWEATHER(Sydney)" — but ONLY when it is the entire reply (after an
         optional mood tag). Small models routinely drop the [cmd:...] wrapper; this
         recovers those. Requiring the whole reply to be the command avoids firing on
         a command name that merely appears inside a normal sentence.
    """
    # 1) strict form (also covers [cmd:list] handling by the caller)
    strict = run_first(text)
    if strict is not None:
        return strict

    # 2) bare form: strip an optional leading mood tag, then match the WHOLE remainder
    stripped = _MOOD_RE.sub('', text).strip()
    for name in sorted(_REGISTRY.keys(), key=len, reverse=True):   # longest name first
        m = re.match(
            rf'^\[?\s*(?:cmd:)?\s*{re.escape(name)}\s*(?:\((.*)\)|\s+(\S.*?))?\s*\]?$',
            stripped, re.IGNORECASE,
        )
        if m:
            arg = m.group(1) if m.group(1) is not None else m.group(2)
            arg = arg.strip() if arg else None
            fn, _ = _REGISTRY[name]
            return (stripped, fn(arg))
    return None