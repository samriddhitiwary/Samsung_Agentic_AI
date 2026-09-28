from __future__ import annotations

import re
from dataclasses import dataclass


IDENTIFIER_RE = re.compile(r"\b[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?\b")
STRUCTURAL_ORDER_RE = re.compile(
    r"\b(?:call|calls|invoke|invokes|run|runs|happen|happens)\s+([A-Za-z_$][\w$]*)\s+(?:before|then|prior to)\s+([A-Za-z_$][\w$]*)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryPlan:
    query_type: str
    terms: list[str]
    order_pair: tuple[str, str] | None
    semantic_top_k: int
    structural_top_k: int


def classify_query(query: str, *, top_k: int = 10) -> QueryPlan:
    text = query.strip()
    lower = text.lower()
    order_pair = _extract_order_pair(text)
    identifiers = _identifier_terms(text)

    if order_pair:
        query_type = "structural" if len(identifiers) <= 4 else "mixed"
    elif any(marker in lower for marker in (" used", " usages", " referenced", " references", " call ", " calls ", " callers", " where is")) and identifiers:
        query_type = "usage"
    elif identifiers and any(marker in lower for marker in (" before ", " after ", " deeplink", "endpoint", "event handler")):
        query_type = "mixed"
    else:
        query_type = "semantic"

    return QueryPlan(
        query_type=query_type,
        terms=identifiers,
        order_pair=order_pair,
        semantic_top_k=max(top_k * 3, 30),
        structural_top_k=max(top_k * 5, 25),
    )


def _extract_order_pair(query: str) -> tuple[str, str] | None:
    match = STRUCTURAL_ORDER_RE.search(query)
    if match:
        return match.group(1), match.group(2)
    before_match = re.search(r"\b([A-Za-z_$][\w$]*)\s+before\s+([A-Za-z_$][\w$]*)\b", query, re.IGNORECASE)
    if before_match:
        return before_match.group(1), before_match.group(2)
    return None


def _identifier_terms(query: str) -> list[str]:
    stop = {
        "where", "which", "what", "when", "how", "does", "used", "usage", "referenced", "references",
        "files", "functions", "function", "class", "classes", "call", "calls", "before", "after",
        "the", "this", "that", "with", "from", "code", "snippet", "source", "handled", "created",
    }
    terms: list[str] = []
    for token in IDENTIFIER_RE.findall(query):
        if token.lower() in stop:
            continue
        if token not in terms:
            terms.append(token)
        short = token.split(".")[-1]
        if short != token and short not in terms:
            terms.append(short)
    return terms[:10]
