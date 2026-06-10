"""
AI Image Service — wraps Google Gemini for image generation and analysis.
Requires GOOGLE_API_KEY, GEMINI_API_KEY, or APIM_KEY in the environment.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

DATA_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DATA_DIR.parent

load_dotenv(PROJECT_DIR / ".env")
load_dotenv(DATA_DIR / ".env")

GENERATED_IMAGES_DIR = DATA_DIR / "generated_images"
GENERATED_IMAGES_DIR.mkdir(exist_ok=True)

_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash")
_IMAGE_MODEL  = os.getenv("GEMINI_IMAGE_MODEL",  "gemini-2.0-flash-preview-image-generation")
_API_KEY_NAMES = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "APIM_KEY")


class AIImageServiceError(Exception):
    """Raised for recoverable AI service errors shown to the user."""


@dataclass
class GeneratedImage:
    path: Path
    message: str


class AIImageService:
    """Thin wrapper around the Google Generative AI SDK."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        genai_module: Any | None = None,
        image_client: Any | None = None,
    ) -> None:
        api_key = (api_key or _get_api_key()).strip()
        if not api_key:
            raise AIImageServiceError(
                "No API key found.\n"
                "Set GOOGLE_API_KEY, GEMINI_API_KEY, or APIM_KEY in the .env file.\n"
                "Get a free key at https://aistudio.google.com/app/apikey"
            )
        self._api_key = api_key
        self._image_client = image_client

        if genai_module is None:
            try:
                import google.generativeai as genai_module
            except ImportError as exc:
                raise AIImageServiceError(
                    "google-generativeai is not installed.\n"
                    "Run:  pip install google-generativeai"
                ) from exc
        if hasattr(genai_module, "configure"):
            genai_module.configure(api_key=api_key)
        self._genai = genai_module

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------
    def generate_image(self, prompt: str) -> GeneratedImage:
        """Generate an image from a text prompt and save it as PNG."""
        try:
            client = self._image_client
            config = None
            if client is None:
                from google.genai import Client
                from google.genai.types import GenerateContentConfig

                client = Client(api_key=self._api_key)
                config = GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])

            response = client.models.generate_content(
                model=_IMAGE_MODEL,
                contents=prompt,
                config=config,
            )
            message = "Image generated and saved."
            for part in _response_parts(response):
                text = getattr(part, "text", None)
                if text:
                    message = str(text).strip()
                if part.inline_data is not None:
                    image_data = _inline_data_bytes(part.inline_data.data)
                    save_path = GENERATED_IMAGES_DIR / _unique_name("generated", ".png")
                    save_path.write_bytes(image_data)
                    return GeneratedImage(path=save_path, message=message)
            raise AIImageServiceError("The model did not return an image. Try a different prompt.")
        except AIImageServiceError:
            raise
        except Exception as exc:
            raise AIImageServiceError(f"Image generation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Image analysis
    # ------------------------------------------------------------------
    def analyze_image(self, image_path: Path, question: str = "") -> str:
        """Analyze a local image file using Gemini vision."""
        if not image_path.exists():
            raise AIImageServiceError(f"Image not found: {image_path}")

        suffix = image_path.suffix.lower().lstrip(".")
        mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png", "webp": "image/webp",
                    "heic": "image/heic", "heif": "image/heif"}
        mime = mime_map.get(suffix)
        if mime is None:
            raise AIImageServiceError(
                "Unsupported image type. Choose a PNG, JPG, JPEG, WEBP, HEIC, or HEIF file."
            )

        prompt_text = question.strip() or "Describe this image in detail."

        try:
            model = self._genai.GenerativeModel(_VISION_MODEL)
            image_data = image_path.read_bytes()
            response = model.generate_content([
                {"mime_type": mime, "data": image_data},
                prompt_text,
            ])
            return response.text.strip()
        except Exception as exc:
            raise AIImageServiceError(f"Image analysis failed: {exc}") from exc


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _unique_name(prefix: str, suffix: str) -> str:
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{ts}{suffix}"


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


def _response_parts(response: Any) -> list[Any]:
    """Return Gemini response parts across old and new SDK response shapes."""
    direct_parts = getattr(response, "parts", None)
    if direct_parts:
        return list(direct_parts)

    candidates = getattr(response, "candidates", None) or []
    parts: list[Any] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        candidate_parts = getattr(content, "parts", None)
        if candidate_parts:
            parts.extend(candidate_parts)
    return parts


def _inline_data_bytes(data: Any) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, str):
        return base64.b64decode(data)
    raise AIImageServiceError("The model returned image data in an unsupported format.")
