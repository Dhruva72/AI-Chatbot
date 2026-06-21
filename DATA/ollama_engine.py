"""Local Ollama (Llama 3) backend for the NLP chatbot.

Replaces the previous Gemini integration. Exposes:
    - OLLAMA_URL, OLLAMA_MODEL  : config constants
    - OllamaError               : raised on any recoverable failure
    - generate_response()       : main text-generation entry point
    - build_context()           : helper to build intent/sentiment context
"""

from __future__ import annotations

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
_TIMEOUT = 60


class OllamaError(Exception):
    """Raised when the local Ollama server is unreachable, slow, or errors out."""


def generate_response(prompt: str, context: str = "") -> str:
    """
    Send `prompt` (with optional `context`) to the local Ollama model and
    return the generated text. Raises OllamaError on any failure instead of
    returning an error string, so callers can handle it explicitly.
    """
    full_prompt = f"{context}\n\nUser: {prompt}\nAssistant:" if context else prompt

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 300,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise OllamaError(
            "Ollama server is not running. Start it with `ollama serve` "
            f"and make sure the '{OLLAMA_MODEL}' model is installed "
            f"(`ollama pull {OLLAMA_MODEL}`)."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaError(
            "Ollama took too long to respond. Try a shorter message or "
            "restart the Ollama server."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise OllamaError(f"Ollama request failed: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise OllamaError("Ollama returned invalid JSON.") from exc

    if data.get("error"):
        message = str(data["error"])
        if "not found" in message.lower():
            message += f" Install it with `ollama pull {OLLAMA_MODEL}`."
        raise OllamaError(message)

    text = str(data.get("response", "")).strip()
    if not text:
        raise OllamaError("Ollama returned an empty response.")
    return text


def build_context(intent: str, sentiment: dict) -> str:
    """
    Build a system-style context string from intent + sentiment to feed
    Ollama for more tailored, empathetic responses. Optional helper — not
    currently called by ai_chat_service.py, but available if you want to
    pass richer context later.
    """
    mood = sentiment.get("label", "NEUTRAL")
    score = sentiment.get("compound", 0.0)
    return (
        f"You are a helpful assistant. "
        f"The user's intent is '{intent}'. "
        f"Their current mood is {mood} (score: {score:.2f}). "
        f"Respond empathetically and helpfully."
    )
