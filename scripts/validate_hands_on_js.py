from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.service import PROJECT_ROOT, ApiService


REPORT_PATH = PROJECT_ROOT / "data/api/js_hands_on_validation.json"
REPO_PATH = PROJECT_ROOT / "data/api/js_smoke_repo"


QUERIES = [
    {
        "query": "How is a payload signed?",
        "type": "semantic",
        "expected_files": ["src/signer.js"],
        "expected_symbols": ["createSignature"],
    },
    {
        "query": "Where is openBluetoothSettings used?",
        "type": "usage",
        "expected_files": ["src/settings.js", "src/workflow.js"],
        "expected_symbols": ["openBluetoothSettings", "setupDevice"],
    },
    {
        "query": "Which functions call validate before save?",
        "type": "structural",
        "expected_files": ["src/workflow.js"],
        "expected_symbols": ["setupDevice"],
    },
    {
        "query": "Where is SHA1 signer used before serialization?",
        "type": "mixed",
        "expected_files": ["src/signer.js"],
        "expected_symbols": ["legacySign"],
    },
    {
        "query": "Where is the Bluetooth settings deeplink defined?",
        "type": "usage",
        "expected_files": ["src/settings.js"],
        "expected_symbols": ["BLUETOOTH_SETTINGS_DEEPLINK"],
    },
]


def main() -> None:
    started = time.perf_counter()
    repo, commits = make_repo()
    service = ApiService()
    if not service.health()["llama_cpp"]["server_healthy"]:
        raise SystemExit("Embedding server is offline; start llama.cpp before running hands-on validation.")

    index_started = time.perf_counter()
    register = service.register_repo(repo_path=str(repo), repo_id="js_hands_on_smoke", commit=commits[-1])
    indexing_seconds = time.perf_counter() - index_started

    rows = []
    latencies = []
    for item in QUERIES:
        q_started = time.perf_counter()
        payload = service.query(repo_id="js_hands_on_smoke", query=item["query"], commit=commits[-1], top_k=10)
        elapsed_ms = (time.perf_counter() - q_started) * 1000.0
        latencies.append(elapsed_ms)
        results = payload["results"]
        rows.append(
            {
                **item,
                "classified_as": payload["query_type"],
                "latency_ms": elapsed_ms,
                "top_result": _result_ref(results[0]) if results else None,
                "retrieved_top_k": [_result_ref(r) for r in results],
                "precision_at_1": _precision_at_k(results, item, 1),
                "precision_at_5": _precision_at_k(results, item, 5),
                "recall_at_5": _recall_at_k(results, item, 5),
                "recall_at_10": _recall_at_k(results, item, 10),
                "mrr": _mrr(results, item),
            }
        )

    report = {
        "label": "Internal Hands-On Retrieval Validation",
        "generated_at": datetime.now(UTC).isoformat(),
        "repo": {
            "path": "data/api/js_smoke_repo",
            "commits": commits,
            "note": "Temporary engineering smoke-test repository; not the Samsung official sample repository.",
        },
        "indexing": {
            "seconds": indexing_seconds,
            "source_files": register.get("source_files"),
            "loc": register.get("loc"),
            "symbols": register.get("symbols"),
            "chunks": register.get("chunks"),
            "call_edges": register.get("call_edges"),
            "reference_edges": register.get("reference_edges"),
            "embeddings_reused": register.get("reused_embeddings"),
            "embeddings_generated": register.get("newly_generated_embeddings"),
            "total_indexing_time_seconds": register.get("indexing_runtime_seconds"),
        },
        "metrics": {
            "precision_at_1": _avg(rows, "precision_at_1"),
            "precision_at_5": _avg(rows, "precision_at_5"),
            "recall_at_5": _avg(rows, "recall_at_5"),
            "recall_at_10": _avg(rows, "recall_at_10"),
            "mrr": _avg(rows, "mrr"),
            "median_latency_ms": median(latencies),
            "p95_latency_ms": _percentile_nearest_rank(latencies, 0.95),
        },
        "queries": rows,
        "runtime_seconds": time.perf_counter() - started,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def make_repo() -> tuple[Path, list[str]]:
    if REPO_PATH.exists():
        shutil.rmtree(REPO_PATH, onerror=_make_writable)
    (REPO_PATH / "src").mkdir(parents=True)
    _run(["git", "init"], REPO_PATH)
    _run(["git", "config", "user.email", "validation@example.local"], REPO_PATH)
    _run(["git", "config", "user.name", "Validation"], REPO_PATH)

    (REPO_PATH / "src/settings.js").write_text(
        """export const BLUETOOTH_SETTINGS_DEEPLINK = "app://settings/bluetooth";

export function openBluetoothSettings(navigator) {
  return navigator.open(BLUETOOTH_SETTINGS_DEEPLINK);
}

export class BluetoothSettings {
  constructor(navigator) {
    this.navigator = navigator;
  }

  open() {
    return openBluetoothSettings(this.navigator);
  }
}
""",
        encoding="utf-8",
    )
    (REPO_PATH / "src/workflow.js").write_text(
        """import { openBluetoothSettings } from "./settings.js";

function validate(device) {
  return Boolean(device && device.id);
}

function save(device) {
  return localStorage.setItem("device", JSON.stringify(device));
}

export function setupDevice(device, navigator) {
  validate(device);
  save(device);
  return openBluetoothSettings(navigator);
}
""",
        encoding="utf-8",
    )
    _run(["git", "add", "."], REPO_PATH)
    _run(["git", "commit", "-m", "initial bluetooth settings workflow"], REPO_PATH)
    commit_a = _run(["git", "rev-parse", "HEAD"], REPO_PATH)

    (REPO_PATH / "src/signer.js").write_text(
        """function serialize(payload) {
  return JSON.stringify(payload);
}

function sha1(input) {
  return `sha1:${input}`;
}

export function createSignature(payload, secret) {
  const encoded = serialize(payload);
  return sha1(`${encoded}.${secret}`);
}

export function legacySign(payload, secret) {
  const digest = sha1(secret);
  const body = serialize(payload);
  return createSignature({ digest, body }, secret);
}
""",
        encoding="utf-8",
    )
    _run(["git", "add", "."], REPO_PATH)
    _run(["git", "commit", "-m", "add signer workflow"], REPO_PATH)
    commit_b = _run(["git", "rev-parse", "HEAD"], REPO_PATH)
    return REPO_PATH, [commit_a, commit_b]


def _run(args: list[str], cwd: Path) -> str:
    result = subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _make_writable(func: Any, path: str, _: Any) -> None:
    os.chmod(path, 0o700)
    func(path)


def _result_ref(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": result.get("rank"),
        "path": result.get("path"),
        "symbol": result.get("symbol"),
        "location": f"{result.get('path')}:{result.get('start_line')}-{result.get('end_line')}",
        "score": result.get("score"),
        "evidence_type": result.get("evidence_type"),
    }


def _is_relevant(result: dict[str, Any], expected: dict[str, Any]) -> bool:
    return result.get("path") in expected["expected_files"] or result.get("symbol") in expected["expected_symbols"]


def _precision_at_k(results: list[dict[str, Any]], expected: dict[str, Any], k: int) -> float:
    subset = results[:k]
    return sum(1 for result in subset if _is_relevant(result, expected)) / max(1, len(subset))


def _recall_at_k(results: list[dict[str, Any]], expected: dict[str, Any], k: int) -> float:
    expected_files = set(expected["expected_files"])
    found = {result.get("path") for result in results[:k] if result.get("path") in expected_files}
    return len(found) / max(1, len(expected_files))


def _mrr(results: list[dict[str, Any]], expected: dict[str, Any]) -> float:
    for idx, result in enumerate(results, start=1):
        if _is_relevant(result, expected):
            return 1.0 / idx
    return 0.0


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(1, len(rows))


def _percentile_nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


if __name__ == "__main__":
    main()
