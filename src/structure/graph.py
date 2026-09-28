from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.versioning.chunk_manifest import load_chunk_manifest


GRAPH_VERSION = "structural_graph_v1"


def build_structural_graph(*, chunk_manifest_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    manifest = load_chunk_manifest(chunk_manifest_path)
    symbols: dict[str, list[dict[str, Any]]] = defaultdict(list)
    references: dict[str, list[dict[str, Any]]] = defaultdict(list)
    calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    imports: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}

    for chunk in manifest.get("chunks", {}).values():
        chunks[chunk["version_id"]] = {
            "version_id": chunk["version_id"],
            "content_id": chunk["content_id"],
            "path": chunk["path"],
            "symbol": chunk.get("symbol"),
            "symbol_type": chunk.get("symbol_type", chunk.get("chunk_type")),
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "language": chunk["language"],
            "snippet": chunk.get("text", ""),
        }
        symbol = chunk.get("symbol")
        if symbol:
            symbols[symbol].append(_chunk_location(chunk, evidence_type="definition"))
        for item in chunk.get("imports", []):
            imports.append({**_chunk_location(chunk, evidence_type="import"), "import": item})
        for identifier in chunk.get("referenced_identifiers", []):
            references[identifier].append(_chunk_location(chunk, evidence_type="reference", referenced_symbol=identifier))
        for call in chunk.get("calls", []):
            name = call.get("name") or call.get("raw")
            if not name:
                continue
            calls[name].append({**_chunk_location(chunk, evidence_type="call", referenced_symbol=name), "call": call})
            short = name.split(".")[-1]
            if short != name:
                calls[short].append({**_chunk_location(chunk, evidence_type="call", referenced_symbol=short), "call": call})

    graph = {
        "graph_version": GRAPH_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "repo": manifest["repo"],
        "commit": manifest["commit"],
        "source_chunk_manifest": str(chunk_manifest_path),
        "symbols": dict(sorted(symbols.items())),
        "references": dict(sorted(references.items())),
        "calls": dict(sorted(calls.items())),
        "imports": imports,
        "chunks": chunks,
        "stats": {
            "symbols": sum(len(v) for v in symbols.values()),
            "reference_edges": sum(len(v) for v in references.values()),
            "call_edges": sum(len(v) for v in calls.values()),
            "import_edges": len(imports),
            "chunks": len(chunks),
            "build_time_seconds": time.perf_counter() - started,
        },
    }
    output_path = structural_graph_output_path(output_dir, manifest["repo"], manifest["commit"])
    save_structural_graph(graph, output_path)
    return graph


def structural_graph_output_path(output_dir: str | Path, repo_id: str, commit: str) -> Path:
    safe_repo = repo_id.replace("\\", "_").replace("/", "_").replace(":", "_")
    return Path(output_dir) / safe_repo / f"{commit}.json"


def save_structural_graph(graph: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_structural_graph(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def search_references(graph: dict[str, Any], terms: list[str], *, limit: int = 50) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for term in terms:
        for bucket in ("calls", "references", "symbols"):
            for item in graph.get(bucket, {}).get(term, []):
                key = (item["path"], item.get("symbol") or "", int(item["start_line"]))
                if key in seen:
                    continue
                seen.add(key)
                results.append({**item, "matched_term": term, "evidence_type": "reference" if bucket != "symbols" else "definition"})
                if len(results) >= limit:
                    return results
    return results


def search_call_order(graph: dict[str, Any], first: str, second: str, *, limit: int = 50) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for chunk in graph.get("chunks", {}).values():
        calls = _calls_in_chunk(graph, chunk["version_id"])
        first_calls = [c for c in calls if _call_matches(c, first)]
        second_calls = [c for c in calls if _call_matches(c, second)]
        if not first_calls or not second_calls:
            continue
        ordered = [
            (a, b)
            for a in first_calls
            for b in second_calls
            if (int(a["call"].get("line", 0)), int(a["call"].get("column", 0))) < (int(b["call"].get("line", 0)), int(b["call"].get("column", 0)))
        ]
        if not ordered:
            continue
        first_call, second_call = ordered[0]
        matches.append(
            {
                **chunk,
                "evidence_type": "structural",
                "ordered_calls": [first_call["call"], second_call["call"]],
                "referenced_symbol": f"{first} before {second}",
                "first_occurrence": first_call["call"],
                "second_occurrence": second_call["call"],
                "limitation": "static source-order / structural ordering",
            }
        )
        if len(matches) >= limit:
            break
    return matches


def _calls_in_chunk(graph: dict[str, Any], version_id: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for items in graph.get("calls", {}).values():
        for item in items:
            if item.get("version_id") == version_id:
                found.append(item)
    found.sort(key=lambda item: (int(item.get("call", {}).get("line", 0)), int(item.get("call", {}).get("column", 0)), item.get("referenced_symbol") or ""))
    return found


def _call_matches(item: dict[str, Any], term: str) -> bool:
    term_l = term.lower()
    call = item.get("call", {})
    candidates = [item.get("referenced_symbol", ""), call.get("name", ""), call.get("raw", "")]
    return any(c and (c.lower() == term_l or c.lower().endswith("." + term_l)) for c in candidates)


def _chunk_location(chunk: dict[str, Any], *, evidence_type: str, referenced_symbol: str | None = None) -> dict[str, Any]:
    return {
        "version_id": chunk["version_id"],
        "content_id": chunk["content_id"],
        "path": chunk["path"],
        "symbol": chunk.get("symbol"),
        "symbol_type": chunk.get("symbol_type", chunk.get("chunk_type")),
        "start_line": chunk["start_line"],
        "end_line": chunk["end_line"],
        "snippet": chunk.get("text", ""),
        "evidence_type": evidence_type,
        "referenced_symbol": referenced_symbol,
    }

