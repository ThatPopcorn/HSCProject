"""
Live-data module for the Tamagotchi bot.

Primary path  — prefetch_context(user_text):
  Detects what live data the user is asking about, fetches it before the
  model call, and returns a compact string to inject into the prompt.
  The model sees the real values and answers naturally — no token tricks.

Fallback path — resolve(text):
  Replaces any [cmd:NAME] / [cmd:NAME(arg)] tokens the model may emit.
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

@_reg("GETTIME", "GETTIME or GETTIME(city)")
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


@_reg("GETDATE", "GETDATE")
def _getdate(arg: Optional[str] = None) -> str:
    today = datetime.date.today()
    return today.strftime(f"%A, %B {today.day} %Y")


@_reg("GETWEATHER", "GETWEATHER(city)")
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


@_reg("GETDAY", "GETDAY — day of the week")
def _getday(arg: Optional[str] = None) -> str:
    return datetime.date.today().strftime("%A")


# ── Public API ──────────────────────────────────────────────────────────────

def command_list() -> str:
    """Return a compact command reference for the system prompt."""
    return "; ".join(desc for _, (_, desc) in sorted(_REGISTRY.items()))


_CMD_RE = re.compile(r'\[cmd:([A-Za-z_]+)(?:\(([^)]*)\))?\]')


def resolve(text: str) -> str:
    """Replace every [cmd:NAME] / [cmd:NAME(arg)] token in text with its result."""
    def _sub(m: re.Match) -> str:
        name  = m.group(1).upper()
        arg   = m.group(2)
        entry = _REGISTRY.get(name)
        if entry is None:
            return f"(unknown command: {name})"
        fn, _ = entry
        return fn(arg)
    return _CMD_RE.sub(_sub, text)


# ── Pre-fetch detection ─────────────────────────────────────────────────────

_WEATHER_RE = re.compile(
    r'weather\s+(?:in|for|at|like\s+in)\s+([A-Za-z][A-Za-z ]{1,24}?)(?=[?,.!]|$)'
    r'|([A-Za-z][A-Za-z ]{1,24}?)\s+weather',
    re.IGNORECASE,
)

_CITY_TIME_RE = re.compile(
    r'(?:time|clock)\s+(?:in|at|for)\s+([A-Za-z][A-Za-z ]{1,24}?)(?=[?,.!]|$)',
    re.IGNORECASE,
)


def prefetch_context(user_text: str) -> str:
    """
    Inspect the user's message, fetch any relevant live data, and return a
    compact string like "time: 3:45 PM; date: Monday, May 12 2026; weather in
    Sydney: Overcast, +19°C" to be injected into the model's context.

    Always includes local time + date; adds weather / remote-city time only
    when the question clearly calls for them.
    """
    parts = [
        f"time: {_gettime()}",
        f"date: {_getdate()}",
    ]

    m = _WEATHER_RE.search(user_text)
    if m:
        city = (m.group(1) or m.group(2) or "").strip().rstrip("?!., ")
        if city:
            parts.append(f"weather in {city}: {_getweather(city)}")

    m = _CITY_TIME_RE.search(user_text)
    if m:
        city = m.group(1).strip().rstrip("?!., ")
        result = _gettime(city)
        if not result.startswith("("):
            parts.append(f"time in {city}: {result}")

    return "; ".join(parts)
