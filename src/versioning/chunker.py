from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from typing import Any


CHUNKER_VERSION = "p1_chunker_v1"


@dataclass(frozen=True)
class ChunkerConfig:
    fallback_window_lines: int = 80
    fallback_overlap_lines: int = 12
    min_chunk_chars: int = 1


def normalize_chunk_text(text: str) -> str:
    """Conservative normalization for content identity.

    This intentionally does not rename variables, remove comments/strings, or
    reformat code. It only normalizes line endings, strips trailing whitespace,
    and removes outer blank lines.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


def make_content_id(language: str, normalized_content: str) -> str:
    payload = f"{language}\n{normalized_content}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_version_id(
    repo: str,
    commit: str,
    path: str,
    chunk_type: str,
    symbol: str | None,
    start_line: int,
    end_line: int,
    content_id: str,
) -> str:
    symbol_part = symbol or "<anonymous>"
    payload = f"{repo}:{commit}:{path}:{chunk_type}:{symbol_part}:{start_line}-{end_line}:{content_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chunk_source_file(
    *,
    repo: str,
    commit: str,
    path: str,
    language: str,
    text: str,
    config: ChunkerConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or ChunkerConfig()
    if language == "python":
        chunks = _chunk_python(repo=repo, commit=commit, path=path, language=language, text=text, config=config)
        if chunks:
            return chunks
    return _chunk_fallback(repo=repo, commit=commit, path=path, language=language, text=text, config=config)


def _chunk_python(
    *,
    repo: str,
    commit: str,
    path: str,
    language: str,
    text: str,
    config: ChunkerConfig,
) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    chunks: list[dict[str, Any]] = []

    imports = _leading_import_block(tree)
    if imports is not None:
        start_line, end_line = imports
        _append_chunk(
            chunks,
            repo=repo,
            commit=commit,
            path=path,
            language=language,
            chunk_type="imports",
            symbol="<module_imports>",
            start_line=start_line,
            end_line=end_line,
            lines=lines,
            config=config,
        )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _append_chunk(
                chunks,
                repo=repo,
                commit=commit,
                path=path,
                language=language,
                chunk_type="async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
                symbol=node.name,
                start_line=node.lineno,
                end_line=_end_lineno(node),
                lines=lines,
                config=config,
            )
        elif isinstance(node, ast.ClassDef):
            _append_class_context_chunk(
                chunks=chunks,
                repo=repo,
                commit=commit,
                path=path,
                language=language,
                node=node,
                lines=lines,
                config=config,
            )
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _append_chunk(
                        chunks,
                        repo=repo,
                        commit=commit,
                        path=path,
                        language=language,
                        chunk_type="async_method" if isinstance(child, ast.AsyncFunctionDef) else "method",
                        symbol=f"{node.name}.{child.name}",
                        start_line=child.lineno,
                        end_line=_end_lineno(child),
                        lines=lines,
                        config=config,
                        parent_symbol=node.name,
                    )

    if not chunks:
        return []
    return chunks


def _append_class_context_chunk(
    *,
    chunks: list[dict[str, Any]],
    repo: str,
    commit: str,
    path: str,
    language: str,
    node: ast.ClassDef,
    lines: list[str],
    config: ChunkerConfig,
) -> None:
    """Create a compact class chunk without duplicating full method bodies."""

    selected_line_numbers: set[int] = {node.lineno}
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            selected_line_numbers.add(child.lineno)
        else:
            for line_no in range(child.lineno, _end_lineno(child) + 1):
                selected_line_numbers.add(line_no)

    selected = [lines[line_no - 1] for line_no in sorted(selected_line_numbers) if 1 <= line_no <= len(lines)]
    class_text = "\n".join(selected)
    _append_explicit_text_chunk(
        chunks,
        repo=repo,
        commit=commit,
        path=path,
        language=language,
        chunk_type="class",
        symbol=node.name,
        start_line=node.lineno,
        end_line=_end_lineno(node),
        raw_text=class_text,
        config=config,
    )


def _chunk_fallback(
    *,
    repo: str,
    commit: str,
    path: str,
    language: str,
    text: str,
    config: ChunkerConfig,
) -> list[dict[str, Any]]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines:
        return []

    step = max(1, config.fallback_window_lines - config.fallback_overlap_lines)
    chunks: list[dict[str, Any]] = []
    start_idx = 0
    index = 0
    while start_idx < len(lines):
        end_idx = min(len(lines), start_idx + config.fallback_window_lines)
        symbol = f"window_{index:04d}"
        _append_chunk(
            chunks,
            repo=repo,
            commit=commit,
            path=path,
            language=language,
            chunk_type="fallback",
            symbol=symbol,
            start_line=start_idx + 1,
            end_line=end_idx,
            lines=lines,
            config=config,
        )
        if end_idx >= len(lines):
            break
        start_idx += step
        index += 1
    return chunks


def _append_chunk(
    chunks: list[dict[str, Any]],
    *,
    repo: str,
    commit: str,
    path: str,
    language: str,
    chunk_type: str,
    symbol: str,
    start_line: int,
    end_line: int,
    lines: list[str],
    config: ChunkerConfig,
    parent_symbol: str | None = None,
) -> None:
    raw_text = "\n".join(lines[start_line - 1 : end_line])
    _append_explicit_text_chunk(
        chunks,
        repo=repo,
        commit=commit,
        path=path,
        language=language,
        chunk_type=chunk_type,
        symbol=symbol,
        start_line=start_line,
        end_line=end_line,
        raw_text=raw_text,
        config=config,
        parent_symbol=parent_symbol,
    )


def _append_explicit_text_chunk(
    chunks: list[dict[str, Any]],
    *,
    repo: str,
    commit: str,
    path: str,
    language: str,
    chunk_type: str,
    symbol: str,
    start_line: int,
    end_line: int,
    raw_text: str,
    config: ChunkerConfig,
    parent_symbol: str | None = None,
) -> None:
    normalized = normalize_chunk_text(raw_text)
    if len(normalized) < config.min_chunk_chars:
        return
    content_id = make_content_id(language, normalized)
    content_sha256 = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    version_id = make_version_id(
        repo=repo,
        commit=commit,
        path=path,
        chunk_type=chunk_type,
        symbol=symbol,
        start_line=start_line,
        end_line=end_line,
        content_id=content_id,
    )
    chunks.append(
        {
            "repo": repo,
            "commit": commit,
            "path": path,
            "language": language,
            "chunk_type": chunk_type,
            "symbol": symbol,
            "parent_symbol": parent_symbol,
            "start_line": start_line,
            "end_line": end_line,
            "content_sha256": content_sha256,
            "content_id": content_id,
            "version_id": version_id,
            "occurrence_key": f"{path}:{chunk_type}:{symbol}",
            "size_chars": len(normalized),
            "size_lines": max(1, end_line - start_line + 1),
            "text": normalized,
        }
    )


def _leading_import_block(tree: ast.Module) -> tuple[int, int] | None:
    import_nodes: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_nodes.append(node)
            continue
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            continue
        break
    if not import_nodes:
        return None
    return import_nodes[0].lineno, _end_lineno(import_nodes[-1])


def _end_lineno(node: ast.AST) -> int:
    return int(getattr(node, "end_lineno", getattr(node, "lineno", 1)))

