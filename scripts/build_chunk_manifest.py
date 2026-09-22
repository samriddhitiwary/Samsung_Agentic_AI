from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.versioning.chunk_manifest import (
    build_chunk_manifest,
    chunk_manifest_output_path,
    save_chunk_manifest,
)
from src.versioning.chunker import ChunkerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a content-addressed code chunk manifest.")
    parser.add_argument("--repo", required=True, help="Local Git repository path.")
    parser.add_argument("--manifest", required=True, help="File manifest JSON path from build_repo_manifest.py.")
    parser.add_argument("--output-root", default="data/versioning/chunks", help="Root directory for chunk manifests.")
    parser.add_argument("--fallback-window-lines", type=int, default=80)
    parser.add_argument("--fallback-overlap-lines", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ChunkerConfig(
        fallback_window_lines=args.fallback_window_lines,
        fallback_overlap_lines=args.fallback_overlap_lines,
    )
    manifest = build_chunk_manifest(repo_path=args.repo, file_manifest_path=args.manifest, config=config)
    output = chunk_manifest_output_path(args.output_root, manifest["repo"], manifest["commit"])
    save_chunk_manifest(manifest, output)
    stats = manifest["stats"]
    print(f"Saved chunk manifest: {output}")
    print(f"Repo: {manifest['repo']}")
    print(f"Commit: {manifest['commit']}")
    print(f"Files: {stats['total_files']}")
    print(f"Chunks: {stats['total_chunks']}")
    print(f"Chunks by language: {stats['chunks_by_language']}")
    print(f"Chunks by type: {stats['chunks_by_type']}")
    print(f"Average chunk size chars: {stats['average_chunk_size_chars']:.2f}")
    print(f"Median chunk size chars: {stats['median_chunk_size_chars']:.2f}")
    print(f"Build time: {stats['build_time_seconds']:.6f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

