import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import ai_image_service
from ai_image_service import AIImageService, AIImageServiceError


class FakeModels:
    def __init__(self) -> None:
        self.calls = []

    def generate_content(self, *args, **kwargs):
        model = kwargs.get("model", ai_image_service._VISION_MODEL)
        contents = kwargs.get("contents", args[0] if args else None)
        self.calls.append((model, contents))
        if model == ai_image_service._IMAGE_MODEL:
            return SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(text="Image created.", inline_data=None),
                                SimpleNamespace(
                                    text=None,
                                    inline_data=SimpleNamespace(data=b"fake png bytes"),
                                ),
                            ]
                        )
                    )
                ]
            )
        return SimpleNamespace(text="The image contains a test subject.")


class FakeGenAI:
    def __init__(self, models) -> None:
        self._models = models

    def configure(self, api_key) -> None:
        self.api_key = api_key

    def GenerativeModel(self, model_name):
        self.model_name = model_name
        return self._models


class AIImageServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.models = FakeModels()
        self.service = AIImageService(
            api_key="test-key",
            genai_module=FakeGenAI(self.models),
            image_client=SimpleNamespace(models=self.models),
        )

    def test_generate_image_saves_model_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(ai_image_service, "GENERATED_IMAGES_DIR", Path(temp_dir)):
                generated = self.service.generate_image("a blue sunrise")

            self.assertEqual(generated.message, "Image created.")
            self.assertEqual(generated.path.read_bytes(), b"fake png bytes")

    def test_analyze_image_sends_device_image_to_vision_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            image_path.write_bytes(b"fake png bytes")

            result = self.service.analyze_image(image_path, "What is shown?")

        model, contents = self.models.calls[-1]
        self.assertEqual(model, ai_image_service._VISION_MODEL)
        self.assertEqual(contents[1], "What is shown?")
        self.assertEqual(result, "The image contains a test subject.")

    def test_analyze_image_rejects_unsupported_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            text_path = Path(temp_dir) / "sample.txt"
            text_path.write_text("not an image", encoding="utf-8")

            with self.assertRaises(AIImageServiceError):
                self.service.analyze_image(text_path)


if __name__ == "__main__":
    unittest.main()
