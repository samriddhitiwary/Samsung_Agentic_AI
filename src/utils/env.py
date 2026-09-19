"""Small .env loader for local, machine-specific settings."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from path without overriding existing environment variables."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env_path(name: str, default: Path, *, project_root: Path) -> Path:
    """Read a path env var; resolve relative values from project_root."""
    value = os.environ.get(name)
    if not value:
        return default

    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path


def llama_server_url(default_host: str = "127.0.0.1", default_port: str = "8081") -> str:
    """Build the local llama.cpp server URL from env vars."""
    host = os.environ.get("JCR_LLAMA_SERVER_HOST", default_host)
    port = os.environ.get("JCR_LLAMA_SERVER_PORT", default_port)
    return f"http://{host}:{port}"
