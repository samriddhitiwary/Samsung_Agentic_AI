from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.versioning.embedding_cache import EmbeddingConfig
from src.versioning.index_updater import build_version_index, save_index_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a persistent version-aware vector index from a chunk manifest.")
    parser.add_argument("--chunk-manifest", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--report-output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_version_index(
        chunk_manifest_path=args.chunk_manifest,
        embedding_cache_dir=args.embedding_cache_dir,
        index_dir=args.index_dir,
        embedding_config=EmbeddingConfig(),
    )
    if args.report_output:
        save_index_report(report, args.report_output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

