from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from src.versioning.chunker import make_content_id, make_version_id, normalize_chunk_text


JS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx"}
JS_LANGUAGES = {"javascript", "javascriptreact", "typescript", "typescriptreact"}
PARSER_VERSION = "tree_sitter_js_structure_v1"


class JavaScriptParserUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceEntity:
    repo: str
    commit: str
    path: str
    language: str
    symbol: str
    symbol_type: str
    start_line: int
    end_line: int
    parent_symbol: str | None
    snippet: str
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    referenced_identifiers: list[str] = field(default_factory=list)
    content_id: str = ""
    version_id: str = ""
    content_sha256: str = ""

    def to_chunk(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "commit": self.commit,
            "path": self.path,
            "language": self.language,
            "chunk_type": self.symbol_type,
            "symbol_type": self.symbol_type,
            "symbol": self.symbol,
            "parent_symbol": self.parent_symbol,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content_sha256": self.content_sha256,
            "content_id": self.content_id,
            "version_id": self.version_id,
            "occurrence_key": f"{self.path}:{self.symbol_type}:{self.symbol}",
            "size_chars": len(self.snippet),
            "size_lines": max(1, self.end_line - self.start_line + 1),
            "text": self.snippet,
            "imports": self.imports,
            "exports": self.exports,
            "calls": self.calls,
            "referenced_identifiers": self.referenced_identifiers,
            "parser": PARSER_VERSION,
        }


def parse_javascript_entities(
    *,
    repo: str,
    commit: str,
    path: str,
    language: str,
    text: str,
) -> list[dict[str, Any]]:
    parser = _get_parser(language)
    source = text.replace("\r\n", "\n").replace("\r", "\n")
    data = source.encode("utf-8", errors="replace")
    tree = parser.parse(data)
    root = tree.root_node
    if root.has_error:
        # Tree-sitter often recovers usefully, but for indexing we keep only
        # explicit syntax nodes. We do not fabricate regex-derived entities.
        pass

    lines = source.split("\n")
    imports = _top_level_imports(root)
    exports = _top_level_exports(root)
    raw_entities = _collect_entity_nodes(root, data)
    raw_entities.sort(key=lambda item: (item["node"].start_point[0], item["node"].end_point[0], item["symbol"]))

    entities: list[SourceEntity] = []
    for item in raw_entities:
        node = item["node"]
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        snippet = "\n".join(lines[start_line - 1 : end_line])
        normalized = normalize_chunk_text(snippet)
        if not normalized:
            continue
        symbol = item["symbol"]
        symbol_type = item["symbol_type"]
        parent = _find_parent_symbol(item, raw_entities)
        calls = _collect_calls(node, data)
        identifiers = sorted(_collect_identifiers(node, data))
        content_id = make_content_id(language, normalized)
        version_id = make_version_id(
            repo=repo,
            commit=commit,
            path=path,
            chunk_type=symbol_type,
            symbol=symbol,
            start_line=start_line,
            end_line=end_line,
            content_id=content_id,
        )
        entities.append(
            SourceEntity(
                repo=repo,
                commit=commit,
                path=path,
                language=language,
                symbol=symbol,
                symbol_type=symbol_type,
                start_line=start_line,
                end_line=end_line,
                parent_symbol=parent,
                snippet=normalized,
                imports=imports,
                exports=exports,
                calls=calls,
                referenced_identifiers=identifiers,
                content_id=content_id,
                version_id=version_id,
                content_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            )
        )
    return [entity.to_chunk() for entity in entities]


def _get_parser(language: str):
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError as exc:  # pragma: no cover - exercised in deployment if dependency missing
        raise JavaScriptParserUnavailable(
            "tree-sitter-language-pack is required for JavaScript structural indexing. "
            "Install project requirements before indexing JS repositories."
        ) from exc
    parser_name = "typescript" if language in {"typescript", "typescriptreact"} else "javascript"
    return get_parser(parser_name)


def _collect_entity_nodes(root: Any, data: bytes) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        node_type = node.type
        if node_type == "function_declaration":
            name = _field_text(node, "name", data)
            if name:
                entities.append({"node": node, "symbol": name, "symbol_type": "function"})
        elif node_type == "class_declaration":
            name = _field_text(node, "name", data)
            if name:
                entities.append({"node": node, "symbol": name, "symbol_type": "class"})
        elif node_type == "method_definition":
            name = _field_text(node, "name", data)
            if name:
                kind = "constructor" if name == "constructor" else "method"
                entities.append({"node": node, "symbol": name, "symbol_type": kind})
        elif node_type == "variable_declarator":
            name = _field_text(node, "name", data)
            value = node.child_by_field_name("value")
            if name and value is not None and value.type in {"arrow_function", "function", "function_expression"}:
                entities.append({"node": node, "symbol": name, "symbol_type": "arrow_function" if value.type == "arrow_function" else "function_expression"})
        elif node_type in {"pair", "public_field_definition"}:
            key = node.child_by_field_name("key") or node.child_by_field_name("property")
            value = node.child_by_field_name("value")
            if key is not None and value is not None and value.type in {"arrow_function", "function", "function_expression"}:
                entities.append({"node": node, "symbol": _node_text(key, data), "symbol_type": "object_method"})
        for child in node.children:
            visit(child)

    visit(root)
    return entities


def _collect_calls(node: Any, data: bytes) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def visit(current: Any) -> None:
        if current.type == "call_expression":
            fn = current.child_by_field_name("function")
            if fn is not None:
                calls.append(
                    {
                        "name": _call_name(fn, data),
                        "raw": _node_text(fn, data),
                        "line": current.start_point[0] + 1,
                        "column": current.start_point[1] + 1,
                    }
                )
        for child in current.children:
            visit(child)

    visit(node)
    return calls


def _collect_identifiers(node: Any, data: bytes) -> set[str]:
    identifiers: set[str] = set()

    def visit(current: Any) -> None:
        if current.type in {"identifier", "property_identifier", "shorthand_property_identifier"}:
            text = _node_text(current, data)
            if text:
                identifiers.add(text)
        for child in current.children:
            visit(child)

    visit(node)
    return identifiers


def _top_level_imports(root: Any) -> list[str]:
    return [child.text.decode("utf-8", errors="replace").strip() for child in root.children if child.type == "import_statement"]


def _top_level_exports(root: Any) -> list[str]:
    return [child.text.decode("utf-8", errors="replace").strip() for child in root.children if child.type == "export_statement"]


def _find_parent_symbol(item: dict[str, Any], all_items: list[dict[str, Any]]) -> str | None:
    node = item["node"]
    candidates = [
        other
        for other in all_items
        if other is not item
        and other["node"].start_byte <= node.start_byte
        and other["node"].end_byte >= node.end_byte
        and (other["node"].end_byte - other["node"].start_byte) > (node.end_byte - node.start_byte)
    ]
    if not candidates:
        return None
    parent = min(candidates, key=lambda other: other["node"].end_byte - other["node"].start_byte)
    return parent["symbol"]


def _field_text(node: Any, field: str, data: bytes) -> str | None:
    child = node.child_by_field_name(field)
    return _node_text(child, data) if child is not None else None


def _node_text(node: Any, data: bytes) -> str:
    return data[node.start_byte : node.end_byte].decode("utf-8", errors="replace").strip()


def _call_name(node: Any, data: bytes) -> str:
    if node.type == "member_expression":
        prop = node.child_by_field_name("property")
        if prop is not None:
            obj = node.child_by_field_name("object")
            prefix = _node_text(obj, data) if obj is not None else ""
            prop_text = _node_text(prop, data)
            return f"{prefix}.{prop_text}" if prefix else prop_text
    return _node_text(node, data)
