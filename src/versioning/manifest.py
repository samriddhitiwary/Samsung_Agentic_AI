from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def add_manifest_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    """Add reproducibility metadata without machine-specific absolute paths."""

    manifest = dict(manifest)
    manifest.setdefault("manifest_version", 1)
    manifest.setdefault("generated_at", datetime.now(UTC).isoformat())
    manifest.setdefault(
        "identity",
        {
            "file_id_fields": ["repo", "commit", "relative_path"],
            "content_id_fields": ["sha256"],
            "future_chunking_note": (
                "File-level sha256 values are deterministic and can feed later "
                "chunk-level content-addressed reuse."
            ),
        },
    )
    return manifest


def manifest_output_path(
    output_root: str | Path,
    repo_id: str,
    commit: str,
) -> Path:
    safe_repo = repo_id.replace("\\", "_").replace("/", "_").replace(":", "_")
    return Path(output_root) / safe_repo / f"{commit}.json"


def save_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched = add_manifest_metadata(manifest)
    output.write_text(
        json.dumps(enriched, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_manifests(old_manifest: dict[str, Any], new_manifest: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    old_files = old_manifest.get("files", {})
    new_files = new_manifest.get("files", {})

    old_paths = set(old_files)
    new_paths = set(new_files)
    shared_paths = old_paths & new_paths

    unchanged = sorted(
        path
        for path in shared_paths
        if old_files[path].get("sha256") == new_files[path].get("sha256")
    )
    modified = sorted(
        path
        for path in shared_paths
        if old_files[path].get("sha256") != new_files[path].get("sha256")
    )
    added = sorted(new_paths - old_paths)
    deleted = sorted(old_paths - new_paths)

    total_new = len(new_paths)
    total_old = len(old_paths)
    denominator = max(len(old_paths | new_paths), 1)

    elapsed = time.perf_counter() - started
    return {
        "old": {
            "repo": old_manifest.get("repo"),
            "commit": old_manifest.get("commit"),
            "file_count": total_old,
        },
        "new": {
            "repo": new_manifest.get("repo"),
            "commit": new_manifest.get("commit"),
            "file_count": total_new,
        },
        "counts": {
            "unchanged": len(unchanged),
            "modified": len(modified),
            "added": len(added),
            "deleted": len(deleted),
        },
        "percentages": {
            "unchanged": _percent(len(unchanged), denominator),
            "modified": _percent(len(modified), denominator),
            "added": _percent(len(added), denominator),
            "deleted": _percent(len(deleted), denominator),
        },
        "files": {
            "unchanged": unchanged,
            "modified": modified,
            "added": added,
            "deleted": deleted,
        },
        "comparison_time_seconds": elapsed,
    }


def save_comparison_report(report: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def format_comparison_summary(report: dict[str, Any]) -> str:
    counts = report["counts"]
    percentages = report["percentages"]
    lines = [
        f"Old commit: {report['old']['commit']} ({report['old']['file_count']} files)",
        f"New commit: {report['new']['commit']} ({report['new']['file_count']} files)",
        f"Unchanged: {counts['unchanged']} ({percentages['unchanged']:.2f}%)",
        f"Modified: {counts['modified']} ({percentages['modified']:.2f}%)",
        f"Added: {counts['added']} ({percentages['added']:.2f}%)",
        f"Deleted: {counts['deleted']} ({percentages['deleted']:.2f}%)",
        f"Comparison time: {report['comparison_time_seconds']:.6f}s",
    ]
    return "\n".join(lines)


def _percent(count: int, total: int) -> float:
    return (count / total) * 100.0 if total else 0.0
