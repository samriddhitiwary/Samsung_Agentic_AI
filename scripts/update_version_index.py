from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.versioning.embedding_cache import EmbeddingConfig
from src.versioning.index_updater import save_index_report, update_version_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incrementally update a version-aware vector index.")
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--target-chunk-manifest", required=True)
    parser.add_argument("--embedding-cache-dir", required=True)
    parser.add_argument("--report-output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = update_version_index(
        existing_index_dir=args.index_dir,
        target_chunk_manifest_path=args.target_chunk_manifest,
        embedding_cache_dir=args.embedding_cache_dir,
        embedding_config=EmbeddingConfig(),
    )
    if args.report_output:
        save_index_report(report, args.report_output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

