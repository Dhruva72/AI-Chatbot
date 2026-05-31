import mimetypes
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent.parent
GENERATED_IMAGES_DIR = Path(__file__).resolve().parent / "generated_images"
SUPPORTED_IMAGE_TYPES = {
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
MAX_INLINE_IMAGE_BYTES = 20 * 1024 * 1024


class AIImageServiceError(RuntimeError):
    """Raised when an AI image operation cannot be completed."""


@dataclass
class GeneratedImage:
    path: Path
    message: str


class AIImageService:
    def __init__(self, client: object | None = None) -> None:
        load_dotenv(PROJECT_DIR / ".env")
        self.vision_model = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
        self.image_model = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
        self.client = client or self._create_client()

    @staticmethod
    def _create_client() -> object:
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key or api_key in {"PASTE_YOUR_REAL_KEY_HERE", "your API key"}:
            raise AIImageServiceError(
                "Set GOOGLE_API_KEY in the project .env file before using AI image tools."
            )

        try:
            from google import genai
        except ImportError as exc:
            raise AIImageServiceError(
                "The google-genai package is missing. Run: pip install -r requirements.txt"
            ) from exc

        return genai.Client(api_key=api_key)

    def generate_image(self, prompt: str) -> GeneratedImage:
        prompt = prompt.strip()
        if not prompt:
            raise AIImageServiceError("Enter a description for the image you want to generate.")

        try:
            response = self.client.models.generate_content(
                model=self.image_model,
                contents=prompt,
            )
        except Exception as exc:
            raise AIImageServiceError(f"Image generation failed: {exc}") from exc

        parts = getattr(response, "parts", None)
        if parts is None and getattr(response, "candidates", None):
            parts = response.candidates[0].content.parts

        image_bytes = None
        response_text = ""
        for part in parts or []:
            if getattr(part, "text", None):
                response_text += part.text.strip()
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                image_bytes = inline_data.data
                break

        if not image_bytes:
            raise AIImageServiceError(
                response_text or "The model completed the request but did not return an image."
            )

        GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = GENERATED_IMAGES_DIR / f"generated_{timestamp}.png"
        output_path.write_bytes(image_bytes)
        return GeneratedImage(
            path=output_path,
            message=response_text or "Your generated image is ready.",
        )

    def analyze_image(self, image_path: str | Path, question: str = "") -> str:
        path = Path(image_path)
        if not path.is_file():
            raise AIImageServiceError("Choose an image file from your device.")
        if path.stat().st_size >= MAX_INLINE_IMAGE_BYTES:
            raise AIImageServiceError("Choose an image smaller than 20 MB.")

        mime_type = SUPPORTED_IMAGE_TYPES.get(path.suffix.lower())
        if not mime_type:
            mime_type = mimetypes.guess_type(path.name)[0]
        if mime_type not in SUPPORTED_IMAGE_TYPES.values():
            raise AIImageServiceError("Use a PNG, JPEG, WEBP, HEIC, or HEIF image.")

        try:
            from google.genai import types
        except ImportError as exc:
            raise AIImageServiceError(
                "The google-genai package is missing. Run: pip install -r requirements.txt"
            ) from exc

        prompt = question.strip() or (
            "Describe this image and predict what it contains. "
            "List the main objects or subject and explain your answer briefly."
        )
        try:
            response = self.client.models.generate_content(
                model=self.vision_model,
                contents=[
                    types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type),
                    prompt,
                ],
            )
        except Exception as exc:
            raise AIImageServiceError(f"Image prediction failed: {exc}") from exc

        answer = getattr(response, "text", "").strip()
        if not answer:
            raise AIImageServiceError("The model completed the request without a prediction.")
        return answer
