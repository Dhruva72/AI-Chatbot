import base64
import binascii
import json
import tempfile
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from ai_image_service import AIImageService, AIImageServiceError, GENERATED_IMAGES_DIR


APP_DIR = Path(__file__).resolve().parent
WEB_DIR = APP_DIR / "web"
INDEX_FILE = WEB_DIR / "index.html"
MAX_REQUEST_BYTES = 28 * 1024 * 1024


class WebChatApp:
    def __init__(self) -> None:
        self._engine = None
        self._engine_lock = threading.Lock()
        self._ai_service = None
        self._ai_lock = threading.Lock()

    def get_engine(self):
        if self._engine is None:
            with self._engine_lock:
                if self._engine is None:
                    from main import ChatbotEngine

                    self._engine = ChatbotEngine()
        return self._engine

    def get_ai_service(self) -> AIImageService:
        if self._ai_service is None:
            with self._ai_lock:
                if self._ai_service is None:
                    self._ai_service = AIImageService()
        return self._ai_service

    def chat(self, payload: dict) -> dict:
        message = str(payload.get("message", "")).strip()
        user_name = str(payload.get("user_name", "User")).strip() or "User"
        reply = self.get_engine().reply(message, user_name=user_name)
        result = {
            "text": reply.text,
            "action": reply.action,
            "url": reply.url,
        }
        if reply.action == "generate_image" and reply.prompt:
            generated = self.get_ai_service().generate_image(reply.prompt)
            result["text"] = generated.message
            result["action"] = "show_image"
            result["image_url"] = f"/generated/{generated.path.name}"
        return result

    def analyze(self, payload: dict) -> dict:
        file_name = Path(str(payload.get("file_name", "upload.png"))).name
        question = str(payload.get("question", "")).strip()
        encoded_image = str(payload.get("content_base64", ""))
        if not encoded_image:
            raise AIImageServiceError("Choose an image file from your device.")

        try:
            image_bytes = base64.b64decode(encoded_image, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise AIImageServiceError("The selected image could not be read.") from exc

        suffix = Path(file_name).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(image_bytes)
        try:
            prediction = self.get_ai_service().analyze_image(temp_path, question)
        finally:
            temp_path.unlink(missing_ok=True)
        return {"text": f"Prediction for {file_name}:\n{prediction}"}


class ChatbotRequestHandler(BaseHTTPRequestHandler):
    app: WebChatApp

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send_file(INDEX_FILE, "text/html; charset=utf-8")
            return
        if path == "/api/health":
            self.send_json({"status": "ok"})
            return
        if path.startswith("/generated/"):
            file_name = Path(unquote(path.removeprefix("/generated/"))).name
            image_path = GENERATED_IMAGES_DIR / file_name
            if image_path.is_file() and image_path.parent == GENERATED_IMAGES_DIR:
                self.send_file(image_path, "image/png")
                return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self.read_json()
            if path == "/api/chat":
                self.send_json(self.app.chat(payload))
                return
            if path == "/api/analyze":
                self.send_json(self.app.analyze(payload))
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except (AIImageServiceError, ValueError) as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_json({"error": f"Chatbot request failed: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def read_json(self) -> dict:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid request size.") from exc
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            raise ValueError("Request is empty or too large.")
        try:
            payload = json.loads(self.rfile.read(content_length))
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


def run_web_app(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    ChatbotRequestHandler.app = WebChatApp()
    server = ThreadingHTTPServer((host, port), ChatbotRequestHandler)
    url = f"http://{host}:{server.server_port}"
    print(f"Chatbot web app is running at {url}")
    print("Press Ctrl+C to stop the server.")
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping chatbot web app.")
    finally:
        server.server_close()

