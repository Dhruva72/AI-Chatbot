"""
Gemini text chat service for fallback chatbot answers.

This keeps general questions inside the chatbot instead of sending users to a
Google search page. It uses the same API key setting as image generation.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

DATA_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DATA_DIR.parent

load_dotenv(PROJECT_DIR / ".env")
load_dotenv(DATA_DIR / ".env")

_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.0-flash")
_API_KEY_NAMES = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "APIM_KEY")


class AIChatServiceError(Exception):
    """Raised for recoverable Gemini chat errors shown to the user."""


class AIChatService:
    """Small wrapper around Gemini text generation."""

    def __init__(self) -> None:
        api_key = _get_api_key()
        if not api_key:
            raise AIChatServiceError(
                "No API key found. Set GOOGLE_API_KEY, GEMINI_API_KEY, or APIM_KEY "
                "in the .env file, then restart the chatbot."
            )
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise AIChatServiceError(
                "google-generativeai is not installed. Run: pip install google-generativeai"
            ) from exc

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            _TEXT_MODEL,
            generation_config={
                "temperature": 0.55,
                "top_p": 0.9,
                "max_output_tokens": 700,
            },
        )

    def answer(self, message: str, user_name: str = "User") -> str:
        prompt = f"""
You are the built-in AI brain for an NLP Chatbot app.
Answer inside the chat. Do not tell the user to search Google.
Be helpful, clear, and concise. Use simple language when possible.
If the user asks for current/live facts that may have changed, answer what you
can and mention that live verification may be needed.

User name: {user_name}
User message: {message}
""".strip()
        try:
            response = self._model.generate_content(prompt)
            text = getattr(response, "text", "").strip()
        except Exception as exc:
            raise AIChatServiceError(f"AI answer failed: {exc}") from exc

        if not text:
            raise AIChatServiceError("AI answer failed: the model returned an empty response.")
        return text


def _get_api_key() -> str:
    for name in _API_KEY_NAMES:
        value = os.getenv(name, "").strip()
        if value and not _is_placeholder(value):
            return value
    return ""


def _is_placeholder(value: str) -> bool:
    return value.upper() in {
        "PASTE_YOUR_REAL_KEY_HERE",
        "YOUR_REAL_KEY_HERE",
        "YOUR_API_KEY_HERE",
    }
