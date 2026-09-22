from __future__ import annotations

import json
import statistics
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.versioning.chunker import CHUNKER_VERSION, ChunkerConfig, chunk_source_file
from src.versioning.manifest import compare_manifests, load_manifest


def build_chunk_manifest(
    *,
    repo_path: str | Path,
    file_manifest_path: str | Path,
    config: ChunkerConfig | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = config or ChunkerConfig()
    file_manifest_path = Path(file_manifest_path)
    file_manifest = load_manifest(file_manifest_path)
    repo = file_manifest["repo"]
    commit = file_manifest["commit"]
    repo_path = Path(repo_path).resolve()

    chunks: dict[str, dict[str, Any]] = {}
    for rel_path, file_meta in sorted(file_manifest.get("files", {}).items()):
        if not file_meta.get("indexable", False):
            continue
        raw = _git_bytes(repo_path, ["show", f"{commit}:{rel_path}"])
        text = raw.decode("utf-8", errors="replace")
        file_chunks = chunk_source_file(
            repo=repo,
            commit=commit,
            path=rel_path,
            language=file_meta["language"],
            text=text,
            config=config,
        )
        for chunk in file_chunks:
            chunks[chunk["version_id"]] = chunk

    elapsed = time.perf_counter() - started
    stats = _chunk_stats(chunks)
    stats["build_time_seconds"] = elapsed
    stats["total_files"] = len(file_manifest.get("files", {}))
    stats["total_chunks"] = len(chunks)

    return {
        "manifest_version": 1,
        "chunker_version": CHUNKER_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "repo": repo,
        "commit": commit,
        "source_manifest": {
            "path": file_manifest_path.as_posix(),
            "commit": commit,
            "file_manifest_version": file_manifest.get("manifest_version"),
        },
        "identity": {
            "content_id": "sha256(language + conservative_normalized_chunk_content)",
            "version_id": "sha256(repo + commit + relative path + chunk location + content_id)",
            "embedding_cache_key": "content_id",
        },
        "config": {
            "fallback_window_lines": config.fallback_window_lines,
            "fallback_overlap_lines": config.fallback_overlap_lines,
            "min_chunk_chars": config.min_chunk_chars,
        },
        "stats": stats,
        "chunks": dict(sorted(chunks.items(), key=lambda item: (item[1]["path"], item[1]["start_line"], item[0]))),
    }


def chunk_manifest_output_path(
    output_root: str | Path,
    repo_id: str,
    commit: str,
) -> Path:
    safe_repo = repo_id.replace("\\", "_").replace("/", "_").replace(":", "_")
    return Path(output_root) / safe_repo / f"{commit}.json"


def save_chunk_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_chunk_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_chunk_manifests(
    old_chunk_manifest: dict[str, Any],
    new_chunk_manifest: dict[str, Any],
    old_file_manifest: dict[str, Any] | None = None,
    new_file_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    old_chunks = old_chunk_manifest.get("chunks", {})
    new_chunks = new_chunk_manifest.get("chunks", {})

    old_by_content = _group_by(old_chunks.values(), "content_id")
    new_by_content = _group_by(new_chunks.values(), "content_id")
    old_by_occurrence = {chunk["occurrence_key"]: chunk for chunk in old_chunks.values()}
    new_by_occurrence = {chunk["occurrence_key"]: chunk for chunk in new_chunks.values()}

    reused: list[dict[str, Any]] = []
    modified_replaced: list[dict[str, Any]] = []
    new_required: list[dict[str, Any]] = []

    for new_chunk in sorted(new_chunks.values(), key=_chunk_sort_key):
        content_id = new_chunk["content_id"]
        old_matches = old_by_content.get(content_id, [])
        if old_matches:
            reused.append(
                {
                    "new_version_id": new_chunk["version_id"],
                    "old_version_ids": [chunk["version_id"] for chunk in old_matches],
                    "content_id": content_id,
                    "path": new_chunk["path"],
                    "symbol": new_chunk["symbol"],
                    "chunk_type": new_chunk["chunk_type"],
                }
            )
            continue

        old_same_occurrence = old_by_occurrence.get(new_chunk["occurrence_key"])
        if old_same_occurrence is not None:
            modified_replaced.append(
                {
                    "old_version_id": old_same_occurrence["version_id"],
                    "new_version_id": new_chunk["version_id"],
                    "old_content_id": old_same_occurrence["content_id"],
                    "new_content_id": content_id,
                    "path": new_chunk["path"],
                    "symbol": new_chunk["symbol"],
                    "chunk_type": new_chunk["chunk_type"],
                }
            )
        else:
            new_required.append(_chunk_ref(new_chunk))

    old_content_ids = set(old_by_content)
    new_content_ids = set(new_by_content)
    deleted = [
        _chunk_ref(chunk)
        for chunk in sorted(old_chunks.values(), key=_chunk_sort_key)
        if chunk["content_id"] not in new_content_ids
        and chunk["occurrence_key"] not in new_by_occurrence
    ]

    new_needing_embeddings_count = len(new_chunks) - len(reused)
    old_count = len(old_chunks)
    new_count = len(new_chunks)
    reuse_percentage = (len(reused) / new_count * 100.0) if new_count else 0.0
    elapsed = time.perf_counter() - started

    report: dict[str, Any] = {
        "old": {
            "repo": old_chunk_manifest.get("repo"),
            "commit": old_chunk_manifest.get("commit"),
            "chunk_count": old_count,
        },
        "new": {
            "repo": new_chunk_manifest.get("repo"),
            "commit": new_chunk_manifest.get("commit"),
            "chunk_count": new_count,
        },
        "counts": {
            "reused_chunks": len(reused),
            "modified_replaced_chunks": len(modified_replaced),
            "new_chunks": len(new_required),
            "new_chunks_requiring_embeddings": new_needing_embeddings_count,
            "deleted_chunks": len(deleted),
        },
        "reuse_percentage": reuse_percentage,
        "chunks": {
            "reused": reused,
            "modified_replaced": modified_replaced,
            "new": new_required,
            "deleted": deleted,
        },
        "comparison_time_seconds": elapsed,
    }

    if old_file_manifest is not None and new_file_manifest is not None:
        report["file_level"] = compare_manifests(old_file_manifest, new_file_manifest)
    return report


def save_chunk_comparison_report(report: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def format_chunk_comparison_summary(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        f"Old commit: {report['old']['commit']} ({report['old']['chunk_count']} chunks)",
        f"New commit: {report['new']['commit']} ({report['new']['chunk_count']} chunks)",
        f"Reused chunks: {counts['reused_chunks']} ({report['reuse_percentage']:.2f}% of new chunks)",
        f"Modified/replaced chunks: {counts['modified_replaced_chunks']}",
        f"New chunks: {counts['new_chunks']}",
        f"New chunks requiring future embeddings: {counts['new_chunks_requiring_embeddings']}",
        f"Deleted chunks: {counts['deleted_chunks']}",
        f"Comparison time: {report['comparison_time_seconds']:.6f}s",
    ]
    file_level = report.get("file_level")
    if file_level:
        file_counts = file_level["counts"]
        lines.extend(
            [
                "File-level reuse:",
                f"  Unchanged files: {file_counts['unchanged']}",
                f"  Modified files: {file_counts['modified']}",
                f"  Added files: {file_counts['added']}",
                f"  Deleted files: {file_counts['deleted']}",
            ]
        )
    return "\n".join(lines)


def _chunk_stats(chunks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sizes = [chunk["size_chars"] for chunk in chunks.values()]
    return {
        "chunks_by_language": dict(Counter(chunk["language"] for chunk in chunks.values())),
        "chunks_by_type": dict(Counter(chunk["chunk_type"] for chunk in chunks.values())),
        "average_chunk_size_chars": statistics.fmean(sizes) if sizes else 0.0,
        "median_chunk_size_chars": statistics.median(sizes) if sizes else 0.0,
    }


def _git_bytes(repo: Path, args: list[str]) -> bytes:
    result = subprocess.run(["git", "-C", str(repo), *args], check=False, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr}")
    return result.stdout


def _group_by(chunks: Any, key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk[key], []).append(chunk)
    return grouped


def _chunk_sort_key(chunk: dict[str, Any]) -> tuple[str, int, str]:
    return chunk["path"], int(chunk["start_line"]), chunk["version_id"]


def _chunk_ref(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "version_id": chunk["version_id"],
        "content_id": chunk["content_id"],
        "path": chunk["path"],
        "symbol": chunk["symbol"],
        "chunk_type": chunk["chunk_type"],
        "start_line": chunk["start_line"],
        "end_line": chunk["end_line"],
    }

