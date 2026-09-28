from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.service import ApiService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm up demo services without preloading final answers.")
    parser.add_argument("--repo-id", default=None, help="Optional already-indexed repo ID whose active index/graph should be loaded.")
    parser.add_argument("--commit", default=None, help="Optional indexed commit/ref.")
    args = parser.parse_args()

    service = ApiService()
    started = time.perf_counter()
    health = service.health()
    if not health["llama_cpp"]["server_healthy"]:
        raise SystemExit(f"Embedding server is unavailable at {health['llama_cpp']['server_url']}")

    # Harmless embedding warm-up. This does not preload final answers or run any
    # benchmark/demo query.
    _, embed_profile = service._embed_query_profile("warm up embedding connection")

    loaded_repo = None
    if args.repo_id:
        record = service._repo_record(args.repo_id)
        commit = service._resolve_commit(Path(record["repo_path"]), args.commit or record.get("active_commit") or "HEAD")
        chunk_path = service._abs(record["chunk_manifest_paths"][commit])
        service._load_chunk_manifest_cached(chunk_path)
        service._load_graph_for_record(record, commit)
        service._load_index(args.repo_id, commit, service._paths(args.repo_id)["index"])
        loaded_repo = {"repo_id": args.repo_id, "commit": commit}

    print(
        json.dumps(
            {
                "status": "ok",
                "note": "Warm-up only; not an evaluation query and no indexes were modified.",
                "embedding": embed_profile,
                "loaded_repo": loaded_repo,
                "runtime_ms": (time.perf_counter() - started) * 1000.0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

