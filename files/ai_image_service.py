"""Compatibility import for the maintained local Ollama image service."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from DATA.ai_image_service import (  # noqa: E402,F401
    AIImageService,
    AIImageServiceError,
    GENERATED_IMAGES_DIR,
    GeneratedImage,
)

__all__ = [
    "AIImageService",
    "AIImageServiceError",
    "GENERATED_IMAGES_DIR",
    "GeneratedImage",
]
