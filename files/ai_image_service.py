"""
AI Image Service — wraps Google Gemini for image generation and analysis.
Requires GOOGLE_API_KEY in a .env file (or the environment).
"""

from __future__ import annotations

import os
import base64
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GENERATED_IMAGES_DIR = Path(__file__).resolve().parent / "generated_images"
GENERATED_IMAGES_DIR.mkdir(exist_ok=True)

_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash")
_IMAGE_MODEL  = os.getenv("GEMINI_IMAGE_MODEL",  "gemini-2.0-flash-preview-image-generation")


class AIImageServiceError(Exception):
    """Raised for recoverable AI service errors shown to the user."""


@dataclass
class GeneratedImage:
    path: Path
    message: str


class AIImageService:
    """Thin wrapper around the Google Generative AI SDK."""

    def __init__(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key or api_key == "PASTE_YOUR_REAL_KEY_HERE":
            raise AIImageServiceError(
                "No Google API key found.\n"
                "Open the .env file and set GOOGLE_API_KEY to your real key.\n"
                "Get a free key at https://aistudio.google.com/app/apikey"
            )
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._genai = genai
        except ImportError as exc:
            raise AIImageServiceError(
                "google-generativeai is not installed.\n"
                "Run:  pip install google-generativeai"
            ) from exc

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------
    def generate_image(self, prompt: str) -> GeneratedImage:
        """Generate an image from a text prompt and save it as PNG."""
        try:
            from google.genai import Client
            from google.genai.types import GenerateContentConfig

            client = Client(api_key=os.getenv("GOOGLE_API_KEY"))
            response = client.models.generate_content(
                model=_IMAGE_MODEL,
                contents=prompt,
                config=GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    image_data = base64.b64decode(
                        base64.b64encode(part.inline_data.data)
                    )
                    save_path = GENERATED_IMAGES_DIR / _unique_name("generated", ".png")
                    save_path.write_bytes(image_data)
                    return GeneratedImage(path=save_path, message=f"Image generated and saved.")
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
        mime = mime_map.get(suffix, "image/png")

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
