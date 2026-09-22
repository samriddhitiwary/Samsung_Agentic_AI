from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.versioning.chunk_manifest import (
    compare_chunk_manifests,
    format_chunk_comparison_summary,
    load_chunk_manifest,
    save_chunk_comparison_report,
)
from src.versioning.manifest import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two content-addressed chunk manifests.")
    parser.add_argument("--old", required=True, help="Old chunk manifest JSON path.")
    parser.add_argument("--new", required=True, help="New chunk manifest JSON path.")
    parser.add_argument("--old-file-manifest", default=None, help="Optional old file manifest JSON path.")
    parser.add_argument("--new-file-manifest", default=None, help="Optional new file manifest JSON path.")
    parser.add_argument("--json-output", default=None, help="Optional machine-readable comparison JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    old_manifest = load_chunk_manifest(args.old)
    new_manifest = load_chunk_manifest(args.new)
    old_file_manifest = load_manifest(args.old_file_manifest) if args.old_file_manifest else None
    new_file_manifest = load_manifest(args.new_file_manifest) if args.new_file_manifest else None
    report = compare_chunk_manifests(
        old_manifest,
        new_manifest,
        old_file_manifest=old_file_manifest,
        new_file_manifest=new_file_manifest,
    )
    print(format_chunk_comparison_summary(report))
    if args.json_output:
        output = save_chunk_comparison_report(report, args.json_output)
        print(f"Saved chunk comparison JSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

