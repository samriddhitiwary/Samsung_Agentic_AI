from __future__ import annotations

import fnmatch
import hashlib
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescriptreact",
    ".jsx": "javascriptreact",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "c/cpp-header",
    ".hpp": "cpp-header",
    ".go": "go",
    ".rs": "rust",
}


DEFAULT_EXCLUDED_DIRS: tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "target",
    "out",
    ".next",
    ".turbo",
    ".gradle",
)

DEFAULT_EXCLUDED_GLOBS: tuple[str, ...] = (
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.dll",
    "*.dylib",
    "*.exe",
    "*.obj",
    "*.o",
    "*.a",
    "*.lib",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.7z",
    "*.rar",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.webp",
    "*.ico",
    "*.pdf",
    "*.npy",
    "*.npz",
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.bin",
    "*.gguf",
    "*.onnx",
)


@dataclass(frozen=True)
class ScannerConfig:
    """Configurable repository scan policy.

    The scanner is file-level today, but the manifest records stable file hashes
    and identifiers so later stages can add chunk-level content-addressed reuse
    without rewriting the Git scan layer.
    """

    supported_extensions: frozenset[str] = field(
        default_factory=lambda: frozenset(LANGUAGE_BY_EXTENSION)
    )
    excluded_dirs: tuple[str, ...] = DEFAULT_EXCLUDED_DIRS
    excluded_globs: tuple[str, ...] = DEFAULT_EXCLUDED_GLOBS
    max_file_size_bytes: int = 2_000_000


class GitScannerError(RuntimeError):
    """Raised when Git cannot scan the requested repository version."""


def scan_git_repository(
    repo_path: str | Path,
    commit_ref: str = "HEAD",
    repo_id: str | None = None,
    config: ScannerConfig | None = None,
) -> dict[str, Any]:
    """Scan supported source files in a local Git repository at one commit.

    No checkout is performed. File content is read from Git objects, which keeps
    the scan deterministic for a given repo + commit.
    """

    started = time.perf_counter()
    repo = Path(repo_path).resolve()
    config = config or ScannerConfig()
    if not repo.exists():
        raise GitScannerError(f"Repository path does not exist: {repo}")

    commit = _git_text(repo, ["rev-parse", commit_ref]).strip()
    resolved_repo_id = repo_id or repo.name
    commit_statuses = _commit_name_status(repo, commit)

    files: dict[str, dict[str, Any]] = {}
    skipped: dict[str, int] = {
        "unsupported_extension": 0,
        "excluded": 0,
        "binary": 0,
        "too_large": 0,
        "non_blob": 0,
    }

    for entry in _ls_tree(repo, commit):
        rel_path = entry["path"]
        if entry["object_type"] != "blob":
            skipped["non_blob"] += 1
            continue

        decision = should_index_path(
            rel_path=rel_path,
            size=entry["size"],
            config=config,
        )
        if not decision["candidate"]:
            skipped[str(decision["reason"])] += 1
            continue

        content = _git_bytes(repo, ["show", f"{commit}:{rel_path}"])
        binary = is_binary_content(content)
        if binary:
            skipped["binary"] += 1
            continue

        ext = Path(rel_path).suffix.lower()
        files[rel_path] = {
            "repo_id": resolved_repo_id,
            "commit": commit,
            "path": rel_path,
            "stable_id": f"{resolved_repo_id}:{commit}:{rel_path}",
            "language": LANGUAGE_BY_EXTENSION[ext],
            "extension": ext,
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "git_blob_sha": entry["object_sha"],
            "last_modified_status": commit_statuses.get(rel_path),
            "binary": False,
            "indexable": True,
        }

    elapsed = time.perf_counter() - started
    return {
        "repo": resolved_repo_id,
        "commit": commit,
        "source": {
            "type": "git",
            "path": None,
            "commit_ref": commit_ref,
        },
        "scanner": {
            "supported_extensions": sorted(config.supported_extensions),
            "excluded_dirs": list(config.excluded_dirs),
            "excluded_globs": list(config.excluded_globs),
            "max_file_size_bytes": config.max_file_size_bytes,
        },
        "stats": {
            "scan_time_seconds": elapsed,
            "files_indexable": len(files),
            "files_scanned": len(files),
            "skipped": skipped,
        },
        "files": dict(sorted(files.items())),
    }


def should_index_path(
    rel_path: str,
    size: int,
    config: ScannerConfig | None = None,
) -> dict[str, str | bool | None]:
    config = config or ScannerConfig()
    normalized = rel_path.replace("\\", "/")
    parts = normalized.split("/")

    if any(part in config.excluded_dirs for part in parts):
        return {"candidate": False, "reason": "excluded"}

    if any(fnmatch.fnmatch(normalized, pattern) for pattern in config.excluded_globs):
        return {"candidate": False, "reason": "excluded"}

    if size > config.max_file_size_bytes:
        return {"candidate": False, "reason": "too_large"}

    ext = Path(normalized).suffix.lower()
    if ext not in config.supported_extensions:
        return {"candidate": False, "reason": "unsupported_extension"}

    return {"candidate": True, "reason": None}


def is_binary_content(content: bytes) -> bool:
    if b"\x00" in content:
        return True
    if not content:
        return False
    sample = content[:4096]
    text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(32, 127)))
    non_text = sample.translate(None, text_chars)
    return len(non_text) / len(sample) > 0.30


def _ls_tree(repo: Path, commit: str) -> list[dict[str, Any]]:
    raw = _git_bytes(repo, ["ls-tree", "-r", "-l", "-z", commit])
    entries: list[dict[str, Any]] = []
    for item in raw.split(b"\x00"):
        if not item:
            continue
        metadata, path_bytes = item.split(b"\t", 1)
        mode, object_type, object_sha, size_text = metadata.decode("utf-8").split()
        size = 0 if size_text == "-" else int(size_text)
        entries.append(
            {
                "mode": mode,
                "object_type": object_type,
                "object_sha": object_sha,
                "size": size,
                "path": path_bytes.decode("utf-8"),
            }
        )
    return entries


def _commit_name_status(repo: Path, commit: str) -> dict[str, str]:
    raw = _git_text(repo, ["diff-tree", "--no-commit-id", "--name-status", "-r", "--root", commit])
    statuses: dict[str, str] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith(("R", "C")) and len(parts) >= 3:
            statuses[parts[2]] = status
        elif len(parts) >= 2:
            statuses[parts[1]] = status
    return statuses


def _git_text(repo: Path, args: list[str]) -> str:
    return _git_bytes(repo, args).decode("utf-8", errors="replace")


def _git_bytes(repo: Path, args: list[str]) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise GitScannerError(f"git {' '.join(args)} failed: {stderr}")
    return result.stdout

