import sys
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "DATA"
sys.path.insert(0, str(DATA_DIR))

from ai_image_service import AIImageService, AIImageServiceError


def main() -> None:
    prompt = input("Describe the image to generate: ").strip()
    try:
        generated = AIImageService().generate_image(prompt)
    except AIImageServiceError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(generated.message)
    print(f"Saved to: {generated.path}")


if __name__ == "__main__":
    main()
