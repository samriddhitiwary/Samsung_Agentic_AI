from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.versioning.manifest import (
    compare_manifests,
    format_comparison_summary,
    load_manifest,
    save_comparison_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two repository version manifests.")
    parser.add_argument("--old", required=True, help="Old manifest JSON path.")
    parser.add_argument("--new", required=True, help="New manifest JSON path.")
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional machine-readable comparison JSON output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    old_manifest = load_manifest(args.old)
    new_manifest = load_manifest(args.new)
    report = compare_manifests(old_manifest, new_manifest)

    print(format_comparison_summary(report))
    if args.json_output:
        output = save_comparison_report(report, args.json_output)
        print(f"Saved comparison JSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

