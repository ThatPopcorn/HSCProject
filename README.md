# AI Tamagotchi

A locally-hosted AI companion that runs entirely on a Raspberry Pi 5. Chat with a small digital creature that has moods, memory, and awareness of the current time and weather — no cloud, no subscriptions, no data leaving your device.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- A pulled model (default: `qwen3.5:2b`)

```bash
ollama pull qwen3.5:2b
```

## Setup

```bash
pip install -r requirements.txt
python main.py
```

Then open **http://127.0.0.1:2492** in your browser.

---

## Files

| File | Purpose |
|---|---|
| `main.py` | Backend server — model, memory, WebSocket, FastAPI |
| `ai-interface.html` | Frontend — face animations, chat UI, TTS/STT |
| `inline_commands.py` | Live data injection (time, date, weather) |
| `webserver.py` | Minimal server template for swapping in your own model |
| `requirements.txt` | Python dependencies |

---

## Configuration

Environment variables to override defaults:

| Variable | Default | Description |
|---|---|---|
| `TAMA_MODEL` | `qwen3.5:2b` | Ollama model to use |
| `TAMA_OLLAMA_API` | `http://localhost:11434` | Ollama API endpoint |
| `TAMA_MOCK` | `0` | Set to `1` to run without a model |

```bash
TAMA_MODEL=llama3.2:3b python main.py
```

---

## Features

- **Mood expressions** — the AI responds with a face: `[happy]`, `[sad]`, `[surprised]`, `[neutral]`
- **Conversational memory** — retains the last 8 turns of context
- **Live data** — automatically fetches time, date, and weather when relevant
- **Voice I/O** — speech-to-text and text-to-speech via the browser's Web Speech API (no audio hits the server)
- **Slash commands** — type `/clear` to reset the AI's memory

---

## Privacy

All inference runs locally via Ollama. No conversation data, audio, or personal information is sent to any external service. Weather lookups send only a city name to [wttr.in](https://wttr.in).
