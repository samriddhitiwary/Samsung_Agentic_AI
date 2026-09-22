from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env import env_path, llama_server_url, load_dotenv
from src.versioning.embedding_cache import EmbeddingConfig
from src.versioning.incremental_embedder import build_incremental_embeddings, save_embedding_report


GGUF_FILE = "jina-code-embeddings-1.5b-Q8_0.gguf"
DEFAULT_MODEL_FILE = PROJECT_ROOT / "data" / "cache" / "model_files" / GGUF_FILE
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "versioning" / "embeddings" / "jina_code_1.5b_q8_1024"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "data" / "versioning" / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/reuse content-addressed chunk embeddings.")
    parser.add_argument("--chunk-manifest", required=True, help="Chunk manifest JSON path.")
    parser.add_argument("--server-url", default=llama_server_url())
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    model_file = env_path("JCR_JINA_CODE_1_5B_GGUF_PATH", DEFAULT_MODEL_FILE, project_root=PROJECT_ROOT)
    if not model_file.exists():
        raise FileNotFoundError(f"Expected local GGUF missing: {model_file}")

    config = EmbeddingConfig()
    report = build_incremental_embeddings(
        chunk_manifest_path=args.chunk_manifest,
        cache_dir=args.cache_dir,
        server_url=args.server_url,
        config=config,
        batch_size=args.batch_size,
    )
    report["model_config"]["model_file"] = str(model_file)
    output = save_embedding_report(report, args.report_root)

    print(f"Saved embedding report: {output}")
    print(f"Repo: {report['repo']}")
    print(f"Commit: {report['commit']}")
    print(f"Total chunks: {report['total_chunks']}")
    print(f"Unique content IDs: {report['unique_content_ids']}")
    print(f"Cache hits: {report['cache_hits']}")
    print(f"Cache misses: {report['cache_misses']}")
    print(f"Reused embeddings: {report['reused_embeddings']}")
    print(f"Newly generated embeddings: {report['newly_generated_embeddings']}")
    print(f"Duplicate occurrences avoided: {report['duplicate_occurrences_avoided']}")
    print(f"Reuse percentage: {report['reuse_percentage']:.2f}%")
    print(f"Embedding work reduction: {report['embedding_work_reduction']:.2%}")
    print(f"Runtime: {json.dumps(report['runtime'], sort_keys=True)}")
    print(f"Model/config fingerprint: {report['model_config']['fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

