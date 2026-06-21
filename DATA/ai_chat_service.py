"""Ollama-backed text chat service for fallback chatbot answers."""

from __future__ import annotations

from collections.abc import Callable

from ollama_engine import OllamaError, generate_response


class AIChatServiceError(Exception):
    """Raised for recoverable local chat errors shown to the user."""


class AIChatService:
    """Add the chatbot's system context to local Ollama requests."""

    def __init__(
        self,
        response_generator: Callable[..., str] = generate_response,
    ) -> None:
        self._generate = response_generator

    def answer(self, message: str, user_name: str = "User") -> str:
        intent_context = (
            f"You are a helpful, friendly chatbot assistant. "
            f"Answer clearly and directly in a sentence or two. "
            f"User name: {user_name}"
        )

        try:
            return self._generate(message, context=intent_context)
        except OllamaError as exc:
            raise AIChatServiceError(str(exc)) from exc
        except Exception as exc:
            raise AIChatServiceError(f"Local AI answer failed: {exc}") from exc