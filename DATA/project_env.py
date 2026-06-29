from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = PROJECT_DIR / ".conda" / "python.exe"


def ensure_project_python() -> None:
    """Restart direct project entry points with the bundled Python environment."""
    if os.environ.get("NLP_CHATBOT_NO_ENV_REDIRECT"):
        return
    if not PROJECT_PYTHON.exists():
        return

    try:
        current_python = Path(sys.executable).resolve()
        project_python = PROJECT_PYTHON.resolve()
    except OSError:
        return

    if current_python == project_python:
        return

    script_arg = sys.argv[0] if sys.argv else ""
    if not script_arg or script_arg in {"-c", "-m"}:
        return

    try:
        script_path = Path(script_arg).resolve()
    except OSError:
        return

    if not script_path.exists() or not script_path.is_relative_to(PROJECT_DIR):
        return

    env = os.environ.copy()
    env["NLP_CHATBOT_PROJECT_PYTHON"] = str(project_python)
    os.execve(
        str(project_python),
        [str(project_python), str(script_path), *sys.argv[1:]],
        env,
    )
