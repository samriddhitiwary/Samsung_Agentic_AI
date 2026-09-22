from __future__ import annotations

import json
import shutil
import sys
import subprocess
from pathlib import Path

from fastapi import HTTPException
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.app import app
from src.api.service import ApiService


REPO_ID = "api_p1_demo"
REPO_PATH = "data/versioning/test_repos/p1_chunk_demo"
COMMIT_G = "ab29b7899134c654647bcdead1f3506e92314eb9"
COMMIT_H = "20cc8cd06a9f5707c23a40eb84fa197d0a563c36"


def main() -> int:
    seed_cache()
    client = TestClient(app)
    checks: list[tuple[str, bool, str]] = []

    response = client.get("/health")
    checks.append(("health endpoint", response.status_code == 200, response.text))

    response = client.post(
        "/repos/register",
        json={"repo_path": REPO_PATH, "repo_id": REPO_ID, "commit": COMMIT_G},
    )
    checks.append(("repo registration", response.status_code == 200 and response.json()["chunks"] > 0, response.text))

    response = client.post(
        "/search",
        json={"repo_id": REPO_ID, "query": "function removed from newer version", "commit": COMMIT_G, "top_k": 5},
    )
    deleted_absent = response.status_code == 200 and all(
        item["symbol"] != "removed_later" for item in response.json()["results"]
    )
    checks.append(("deleted-code isolation before reintroduction", deleted_absent, response.text))

    response = client.post(f"/repos/{REPO_ID}/update", json={"commit": COMMIT_H})
    checks.append(
        (
            "incremental update",
            response.status_code == 200
            and response.json()["embeddings_generated"] == 0
            and response.json()["vectors_added"] == 1,
            response.text,
        )
    )

    response = client.post(
        "/search",
        json={"repo_id": REPO_ID, "query": "function removed from newer version", "commit": COMMIT_H, "top_k": 5},
    )
    reintroduced_found = response.status_code == 200 and any(
        item["symbol"] == "removed_later" for item in response.json()["results"]
    )
    checks.append(("version-specific search after update", reintroduced_found, response.text))

    response = client.post(
        "/search/evolution",
        json={"repo_id": REPO_ID, "query": "token authentication", "top_k": 3},
    )
    checks.append(("evolutionary search", response.status_code == 200 and len(response.json()["results"]) > 0, response.text))

    response = client.get(f"/repos/{REPO_ID}/symbols/evolution", params={"symbol": "removed_later", "path": "src/auth.py"})
    symbol_ok = (
        response.status_code == 200
        and response.json()["states"][0]["status"] == "absent"
        and response.json()["states"][-1]["status"] == "present"
        and any(transition["transition"] in {"added", "reintroduced"} for transition in response.json()["transitions"])
    )
    checks.append(("symbol evolution", symbol_ok, response.text))

    response = client.get("/metrics/summary")
    metrics_ok = response.status_code == 200 and response.json()["p0"]["jina_code_1_5b_q8_full_retrieval"]["NDCG@10"] >= 0.86
    checks.append(("metrics summary", metrics_ok, response.text))

    second_client = TestClient(app)
    response = second_client.post(
        "/search",
        json={"repo_id": REPO_ID, "query": "token authentication", "commit": COMMIT_H, "top_k": 3},
    )
    checks.append(("persistence after restart", response.status_code == 200 and len(response.json()["results"]) > 0, response.text))

    response = client.post("/repos/register", json={"repo_path": "missing/repo", "repo_id": "bad", "commit": "HEAD"})
    checks.append(("invalid repo error handling", response.status_code == 400, response.text))

    response = client.post("/search", json={"repo_id": "not_registered", "query": "x", "commit": COMMIT_H, "top_k": 1})
    checks.append(("unregistered repo error handling", response.status_code == 404, response.text))

    unavailable_ok = False
    try:
        ApiService(server_url="http://127.0.0.1:9").register_repo(
            repo_path=REPO_PATH,
            repo_id="unavailable_server_probe",
            commit=COMMIT_G,
        )
    except HTTPException as exc:
        unavailable_ok = exc.status_code == 503
    checks.append(("unavailable model server error handling", unavailable_ok, "expected HTTP 503"))

    unsupported_repo = make_unsupported_repo()
    response = client.post(
        "/repos/register",
        json={"repo_path": str(unsupported_repo), "repo_id": "unsupported_only", "commit": "HEAD"},
    )
    checks.append(("unsupported source files error handling", response.status_code == 400, response.text))

    state_path = PROJECT_ROOT / "data/api/repos" / REPO_ID / "index" / "state.json"
    backup_path = state_path.with_suffix(".json.bak_api_verify")
    corrupt_ok = False
    if state_path.exists():
        shutil.copy2(state_path, backup_path)
        state_path.write_text("{not valid json", encoding="utf-8")
        response = client.post(
            "/search",
            json={"repo_id": REPO_ID, "query": "token authentication", "commit": COMMIT_H, "top_k": 1},
        )
        corrupt_ok = response.status_code == 500
        shutil.move(str(backup_path), str(state_path))
    checks.append(("corrupt persisted index error handling", corrupt_ok, "expected HTTP 500"))

    all_passed = all(ok for _, ok, _ in checks)
    payload = {
        "all_passed": all_passed,
        "checks": [{"name": name, "passed": ok, "detail": trim(detail)} for name, ok, detail in checks],
    }
    output = PROJECT_ROOT / "data/api/verification_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all_passed else 1


def seed_cache() -> None:
    source = PROJECT_ROOT / "data/versioning/embeddings/p1_task3_demo_cache_repro"
    target = PROJECT_ROOT / "data/api/repos" / REPO_ID / "embedding_cache"
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)


def make_unsupported_repo() -> Path:
    repo = PROJECT_ROOT / "data/api/tmp_unsupported_repo"
    if repo.exists():
        return repo
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "README.txt").write_text("no supported source files here\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "api@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "API Verify"], cwd=repo, check=True)
    subprocess.run(["git", "add", "README.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "unsupported files only"], cwd=repo, check=True, capture_output=True, text=True)
    return repo


def trim(value: str, limit: int = 500) -> str:
    compact = " ".join(value.split())
    return compact[:limit] + ("..." if len(compact) > limit else "")


if __name__ == "__main__":
    raise SystemExit(main())
