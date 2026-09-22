from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.jina_code import QUERY_INSTRUCTION
from src.retrieval.jina_gguf import truncate_with_tokenizer
from src.versioning.embedding_cache import EmbeddingConfig
from src.versioning.evolution_search import search_across_versions
from src.versioning.incremental_embedder import _embed_http, check_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search code across indexed Git versions.")
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--evolution-dir", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--start-commit", default=None)
    parser.add_argument("--end-commit", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--include-evolution-context", action="store_true")
    parser.add_argument("--server-url", default="http://127.0.0.1:8081")
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    query_config = EmbeddingConfig(prompt_instruction=QUERY_INSTRUCTION, prompt_version="jina_code_1.5b_query_v1")
    check_server(args.server_url)
    tokenizer = AutoTokenizer.from_pretrained(query_config.model_name)
    prompt = truncate_with_tokenizer(
        tokenizer=tokenizer,
        text=args.query,
        instruction=QUERY_INSTRUCTION,
        max_length=query_config.max_sequence_length,
    )
    embedding = _embed_http(
        texts=[prompt],
        server_url=args.server_url,
        batch_size=1,
        normalize=True,
        desc="embed evolution query",
    )[0]
    payload = search_across_versions(
        index_dir=args.index_dir,
        evolution_dir=args.evolution_dir,
        query_embedding=embedding,
        start_commit=args.start_commit,
        end_commit=args.end_commit,
        top_k=args.top_k,
        raw=args.raw,
        include_evolution_context=args.include_evolution_context,
    )
    payload["query"] = args.query
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
