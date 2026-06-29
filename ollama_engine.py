"""Public Ollama helper used by scripts launched from the project root."""

from DATA.ollama_engine import (  # noqa: F401
    OLLAMA_MODEL,
    OLLAMA_URL,
    OllamaError,
    build_context,
    generate_response,
)

__all__ = ["OLLAMA_MODEL", "OLLAMA_URL", "OllamaError", "build_context", "generate_response"]
