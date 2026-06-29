"""Ollama-backed text chat service for fallback chatbot answers."""

from __future__ import annotations
from collections.abc import Callable

try:
    from .ollama_engine import OllamaError, generate_response
except ImportError:
    from ollama_engine import OllamaError, generate_response


class AIChatServiceError(Exception):
    """Raised for recoverable local chat errors shown to the user."""


class AIChatService:
    """Add the chatbot system context to local Ollama requests."""

    def __init__(
        self,
        response_generator: Callable[..., str] = generate_response,
    ) -> None:
        self._generate = response_generator

    def answer(self, message: str, user_name: str = "User") -> str:
        intent_context = (
            "You are a helpful, knowledgeable assistant. "
            "Answer the user's question thoroughly and completely. "
            "Be clear and well-structured. "
            "The user's name is " + user_name + "."
        )

        try:
            return self._generate(message, context=intent_context)
        except OllamaError as exc:
            raise AIChatServiceError(str(exc)) from exc
        except Exception as exc:
            raise AIChatServiceError(f"Local AI answer failed: {exc}") from exc
