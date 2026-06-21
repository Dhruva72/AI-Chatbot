import tempfile
import unittest
from pathlib import Path

import requests

from ai_image_service import AIImageService, AIImageServiceError


class FakeResponse:
    def __init__(self, payload, status_code=200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response) -> None:
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


class AIImageServiceTest(unittest.TestCase):
    def test_generate_image_explains_llama3_limitation(self) -> None:
        with self.assertRaisesRegex(AIImageServiceError, "text-only"):
            AIImageService().generate_image("a blue sunrise")

    def test_analyze_image_sends_base64_to_ollama_vision_model(self) -> None:
        session = FakeSession(FakeResponse({"response": "A test subject."}))
        service = AIImageService(
            session=session,
            url="http://ollama.test/api/generate",
            vision_model="vision-test",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            image_path.write_bytes(b"fake png bytes")
            result = service.analyze_image(image_path, "What is shown?")

        url, call = session.calls[-1]
        self.assertEqual(url, "http://ollama.test/api/generate")
        self.assertEqual(call["json"]["model"], "vision-test")
        self.assertEqual(call["json"]["prompt"], "What is shown?")
        self.assertEqual(len(call["json"]["images"]), 1)
        self.assertEqual(result, "A test subject.")

    def test_analyze_image_rejects_unsupported_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            text_path = Path(temp_dir) / "sample.txt"
            text_path.write_text("not an image", encoding="utf-8")

            with self.assertRaises(AIImageServiceError):
                AIImageService().analyze_image(text_path)

    def test_missing_vision_model_has_install_hint(self) -> None:
        session = FakeSession(FakeResponse({"error": "model not found"}, 404))
        service = AIImageService(session=session)

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            image_path.write_bytes(b"fake png bytes")
            with self.assertRaisesRegex(AIImageServiceError, "ollama pull"):
                service.analyze_image(image_path)


if __name__ == "__main__":
    unittest.main()
