from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.versioning.git_scanner import ScannerConfig, scan_git_repository
from src.versioning.manifest import manifest_output_path, save_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic Git repository version manifest.")
    parser.add_argument("--repo", required=True, help="Local Git repository path.")
    parser.add_argument("--commit", default="HEAD", help="Commit SHA/ref to scan.")
    parser.add_argument("--repo-id", default=None, help="Stable repository ID/name. Defaults to repo directory name.")
    parser.add_argument(
        "--output-root",
        default="data/versioning/manifests",
        help="Root directory for manifests.",
    )
    parser.add_argument(
        "--max-file-size-bytes",
        type=int,
        default=2_000_000,
        help="Skip files larger than this size.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ScannerConfig(max_file_size_bytes=args.max_file_size_bytes)
    manifest = scan_git_repository(
        repo_path=args.repo,
        commit_ref=args.commit,
        repo_id=args.repo_id,
        config=config,
    )
    output = manifest_output_path(args.output_root, manifest["repo"], manifest["commit"])
    save_manifest(manifest, output)

    stats = manifest["stats"]
    print(f"Saved manifest: {output}")
    print(f"Repo: {manifest['repo']}")
    print(f"Commit: {manifest['commit']}")
    print(f"Files scanned/indexable: {stats['files_scanned']}")
    print(f"Scan time: {stats['scan_time_seconds']:.6f}s")
    print(f"Skipped: {stats['skipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

