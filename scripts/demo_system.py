from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.service import ApiService


DEFAULT_REPO = "data/versioning/real_repos/itsdangerous"
DEFAULT_START = "2f69e841d2a979c616a55b226e444694f5d9c962"
DEFAULT_END = "31f46a3469dbfb2ecf83dd0c4297c1efc508fcca"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal local demo for the Samsung code retrieval system.")
    parser.add_argument("--repo-path", default=DEFAULT_REPO)
    parser.add_argument("--repo-id", default="itsdangerous_api_demo")
    parser.add_argument("--start-commit", default=DEFAULT_START)
    parser.add_argument("--end-commit", default=DEFAULT_END)
    parser.add_argument("--server-url", default="http://127.0.0.1:8081")
    parser.add_argument(
        "--seed-cache",
        action="store_true",
        help="Copy the existing itsdangerous benchmark embedding cache to make the demo fast.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = ApiService(server_url=args.server_url)
    if args.seed_cache:
        seed_cache(args.repo_id)

    print("1) Register/index repository")
    registered = service.register_repo(repo_path=args.repo_path, repo_id=args.repo_id, commit=args.start_commit)
    print(json.dumps(registered, indent=2, sort_keys=True))

    print("\n2) Search current version")
    first_search = service.search(
        repo_id=args.repo_id,
        query="FIPS SHA1 digest method",
        commit=registered["commit"],
        top_k=3,
    )
    print(json.dumps(first_search, indent=2, sort_keys=True))

    print("\n3) Incrementally update repository")
    updated = service.update_repo(repo_id=args.repo_id, commit=args.end_commit)
    print(json.dumps(updated, indent=2, sort_keys=True))

    print("\n4) Search updated version")
    second_search = service.search(
        repo_id=args.repo_id,
        query="FIPS SHA1 digest method",
        commit=updated["new_commit"],
        top_k=3,
    )
    print(json.dumps(second_search, indent=2, sort_keys=True))

    print("\n5) Evolutionary search across indexed history")
    evo = service.evolution_search(
        repo_id=args.repo_id,
        query="serializer signing",
        start_commit=None,
        end_commit=None,
        top_k=3,
        include_evolution_context=True,
    )
    print(json.dumps(evo, indent=2, sort_keys=True))

    print("\n6) Focused evolutionary state query")
    focused = service.evolution_search(
        repo_id=args.repo_id,
        query="HMAC algorithm using lazy SHA1 for FIPS builds",
        start_commit=None,
        end_commit=None,
        top_k=3,
        include_evolution_context=True,
    )
    print(json.dumps(focused, indent=2, sort_keys=True))

    print("\n7) Symbol evolution chain")
    chain = service.symbol_evolution(
        repo_id=args.repo_id,
        symbol="HMACAlgorithm",
        path="src/itsdangerous/signer.py",
    )
    print(json.dumps(chain, indent=2, sort_keys=True))
    return 0


def seed_cache(repo_id: str) -> None:
    source = PROJECT_ROOT / "data/versioning/real_benchmark_artifacts/itsdangerous/embedding_cache_incremental"
    target = PROJECT_ROOT / "data/api/repos" / repo_id / "embedding_cache"
    if not source.exists():
        print(f"Seed cache not found, continuing without it: {source}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    shutil.copytree(source, target)
    print(f"Seeded demo embedding cache: {target}")


if __name__ == "__main__":
    raise SystemExit(main())
