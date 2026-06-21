"""Ollama-backed image analysis for the chatbot."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from ollama_engine import OLLAMA_URL

DATA_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DATA_DIR.parent

load_dotenv(PROJECT_DIR / ".env")
load_dotenv(DATA_DIR / ".env")

GENERATED_IMAGES_DIR = DATA_DIR / "generated_images"
GENERATED_IMAGES_DIR.mkdir(exist_ok=True)

_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision").strip()
_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))


class AIImageServiceError(Exception):
    """Raised for recoverable local image-service errors."""


@dataclass
class GeneratedImage:
    """Compatibility result type for callers that display generated files."""

    path: Path
    message: str


class AIImageService:
    """Analyze uploaded images with an Ollama vision-capable model."""

    def __init__(
        self,
        *,
        session: Any = requests,
        url: str = OLLAMA_URL,
        vision_model: str = _VISION_MODEL,
    ) -> None:
        self._session = session
        self._url = url
        self._vision_model = vision_model

    def generate_image(self, prompt: str) -> GeneratedImage:
        """Explain the capability gap instead of calling a remote provider."""
        raise AIImageServiceError(
            "The local Llama 3 model is text-only and cannot generate image files. "
            "Ollama supports language and vision models, but not text-to-image generation."
        )

    def analyze_image(self, image_path: Path, question: str = "") -> str:
        """Analyze an uploaded image with the configured Ollama vision model."""
        if not image_path.exists():
            raise AIImageServiceError(f"Image not found: {image_path}")

        supported = {".jpg", ".jpeg", ".png", ".webp"}
        if image_path.suffix.lower() not in supported:
            raise AIImageServiceError(
                "Unsupported image type. Choose a PNG, JPG, JPEG, or WEBP file."
            )

        prompt = question.strip() or "Describe this image in detail."
        payload = {
            "model": self._vision_model,
            "prompt": prompt,
            "images": [base64.b64encode(image_path.read_bytes()).decode("ascii")],
            "stream": False,
        }

        try:
            response = self._session.post(self._url, json=payload, timeout=_TIMEOUT)
            response.raise_for_status()
            data = response.json()
        except requests.ConnectionError as exc:
            raise AIImageServiceError(
                "Ollama is not reachable. Start Ollama and try again."
            ) from exc
        except requests.Timeout as exc:
            raise AIImageServiceError("Ollama image analysis timed out.") from exc
        except requests.RequestException as exc:
            detail = _request_error(getattr(exc, "response", None), self._vision_model)
            raise AIImageServiceError(f"Image analysis failed{detail}.") from exc
        except ValueError as exc:
            raise AIImageServiceError("Ollama returned invalid JSON.") from exc

        if data.get("error"):
            raise AIImageServiceError(f"Image analysis failed: {data['error']}")
        text = str(data.get("response", "")).strip()
        if not text:
            raise AIImageServiceError("The vision model returned an empty response.")
        return text


def _request_error(response: requests.Response | None, vision_model: str) -> str:
    if response is None:
        return ""
    try:
        message = str(response.json().get("error", "")).strip()
    except (ValueError, AttributeError):
        message = ""
    if "not found" in message.lower():
        return f": {message}. Install it with `ollama pull {vision_model}`"
    return f": {message}" if message else f" with HTTP {response.status_code}"
