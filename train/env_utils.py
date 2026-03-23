from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOTENV_PATH = REPO_ROOT / ".env"


def _parse_dotenv_line(line: str) -> tuple[str | None, str | None]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None, None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()
    if "=" not in stripped:
        return None, None

    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None, None

    value = value.strip()
    if value and len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return key, value


def load_dotenv_file(dotenv_path: Path | str = DEFAULT_DOTENV_PATH) -> None:
    path = Path(dotenv_path)
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        key, value = _parse_dotenv_line(raw_line)
        if not key or value == "":
            continue
        os.environ.setdefault(key, value)


def configure_swanlab(default_project: str, repo_root: Path | None = None) -> None:
    root = repo_root or REPO_ROOT
    load_dotenv_file(root / ".env")

    os.environ.setdefault("SWANLAB_SAVE_DIR", str(root / ".swanlab"))
    os.environ.setdefault("SWANLAB_LOG_DIR", str(root / "swanlog"))

    if not os.environ.get("SWANLAB_PROJECT"):
        os.environ["SWANLAB_PROJECT"] = default_project

    swanlab_api_key = os.getenv("SWANLAB_API_KEY", "").strip()
    if swanlab_api_key:
        os.environ["SWANLAB_API_KEY"] = swanlab_api_key
        if not os.environ.get("SWANLAB_MODE"):
            os.environ["SWANLAB_MODE"] = "cloud"
    elif not os.environ.get("SWANLAB_MODE"):
        os.environ["SWANLAB_MODE"] = "local"
