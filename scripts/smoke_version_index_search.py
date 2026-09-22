from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.jina_code import QUERY_INSTRUCTION
from src.retrieval.jina_gguf import truncate_with_tokenizer
from src.versioning.embedding_cache import EmbeddingConfig
from src.versioning.incremental_embedder import _embed_http, check_server
from src.versioning.vector_index import VersionedVectorIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manual query smoke tests against a version-aware index.")
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8081")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = EmbeddingConfig(prompt_instruction=QUERY_INSTRUCTION, prompt_version="jina_code_1.5b_query_v1")
    passage_config = EmbeddingConfig()
    check_server(args.server_url)
    index = VersionedVectorIndex.load(args.index_dir, passage_config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    prompts = [
        truncate_with_tokenizer(
            tokenizer=tokenizer,
            text=query,
            instruction=QUERY_INSTRUCTION,
            max_length=config.max_sequence_length,
        )
        for query in args.query
    ]
    embeddings = _embed_http(
        texts=prompts,
        server_url=args.server_url,
        batch_size=4,
        normalize=True,
        desc="embed smoke queries",
    )
    if embeddings.shape[1] != config.embedding_dimension:
        raise ValueError(f"Query embedding dimension mismatch: {embeddings.shape}")
    payload = {
        "index_dir": args.index_dir,
        "commit": args.commit,
        "top_k": args.top_k,
        "queries": [],
    }
    for query, embedding in zip(args.query, embeddings, strict=True):
        results = index.search(query_embedding=embedding.astype(np.float32), commit=args.commit, top_k=args.top_k)
        payload["queries"].append({"query": query, "results": results})
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

