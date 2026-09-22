from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from tqdm import tqdm
from transformers import AutoTokenizer

from src.retrieval.jina_code import (
    DOCUMENT_INSTRUCTION,
    NORMALIZE_EMBEDDINGS,
    POOLING_STRATEGY,
    QUERY_INSTRUCTION,
)
from src.retrieval.jina_gguf import embed_http_cached, model_slug, prompt_hash, truncate_with_tokenizer


MODEL_NAME = "jinaai/jina-code-embeddings-1.5b"
GGUF_FILE = "jina-code-embeddings-1.5b-Q8_0.gguf"
QUANTIZATION = "Q8_0"
MAX_SEQUENCE_LENGTH = 1024
EMBEDDING_DIMENSION = 1536


class SamsungAppsRetrievalEncoder(AbsEncoder):
    """MTEB AbsEncoder wrapper for the frozen Samsung P0 AppsRetrieval run.

    This class intentionally preserves the already-verified P0 embedding
    behavior: Jina Code 1.5B Q8 GGUF, max length 1024, last-token pooling from
    llama.cpp, normalized vectors, official NL→Code query instruction, and
    official passage instruction.
    """

    mteb_model_meta = ModelMeta(
        loader=None,
        name="SamsungPRISM/jina-code-1.5b-q8-gguf-p0",
        revision="jina-code-embeddings-1.5b-Q8_0.gguf",
        release_date=None,
        languages=["eng-Latn"],
        n_parameters=1_500_000_000,
        memory_usage_mb=None,
        max_tokens=MAX_SEQUENCE_LENGTH,
        embed_dim=EMBEDDING_DIMENSION,
        license=None,
        open_weights=True,
        public_training_code=None,
        public_training_data=None,
        framework=["GGUF"],
        similarity_fn_name=ScoringFunction.COSINE,
        use_instructions=True,
        training_datasets=None,
    )

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        server_url: str = "http://127.0.0.1:8081",
        batch_size: int = 4,
        require_cache: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.full_cache_dir = self.cache_dir / "full"
        self.server_url = server_url
        self.batch_size = batch_size
        self.require_cache = require_cache
        self.tokenizer: Any | None = None
        self.cache_events: list[dict[str, Any]] = []

    def encode(
        self,
        inputs: Any,
        *,
        task_metadata: Any,
        hf_split: str,
        hf_subset: str,
        prompt_type: Any | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        role = self._role(prompt_type)
        ids: list[str] = []
        texts: list[str] = []
        for batch in inputs:
            batch_ids, batch_texts = self._extract_batch(batch, role)
            ids.extend(batch_ids)
            texts.extend(batch_texts)

        instruction = QUERY_INSTRUCTION if role == "query" else DOCUMENT_INSTRUCTION
        prompt_role = "query_nl2code" if role == "query" else "corpus_passage"
        cached = self._load_matching_full_cache(ids=ids, role=role)
        if cached is not None:
            embeddings, cache_path = cached
            self._record_event(role, ids, True, cache_path)
            return embeddings

        if self.require_cache:
            raise FileNotFoundError(
                f"No compatible frozen {role} embedding cache found under {self.full_cache_dir}"
            )

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        prompts = [
            truncate_with_tokenizer(
                tokenizer=self.tokenizer,
                text=text,
                instruction=instruction,
                max_length=MAX_SEQUENCE_LENGTH,
            )
            for text in tqdm(texts, desc=f"prepare {role} prompts")
        ]
        cache_stem = (
            f"{model_slug(GGUF_FILE)}_full1024_{prompt_role}_last_maxseq-{MAX_SEQUENCE_LENGTH}_"
            f"{prompt_hash(ids, prompts)}"
        )
        embeddings, cache_hit, cache_path = embed_http_cached(
            ids=ids,
            texts=prompts,
            server_url=self.server_url,
            batch_size=int(kwargs.get("batch_size", self.batch_size)),
            desc=f"embed {role} for MTEB",
            cache_dir=self.full_cache_dir,
            cache_stem=cache_stem,
        )
        self._record_event(role, ids, cache_hit, cache_path)
        self._validate_embeddings(role, embeddings)
        return embeddings

    def _load_matching_full_cache(self, *, ids: list[str], role: str) -> tuple[np.ndarray, Path] | None:
        if not self.full_cache_dir.exists():
            return None
        candidates = sorted(self.full_cache_dir.glob("*_query_nl2code_*.meta.json" if role == "query" else "*_corpus_passage_*.meta.json"))
        for meta_path in candidates:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata.get("ids") != ids:
                continue
            shape = metadata.get("shape")
            if shape != [len(ids), EMBEDDING_DIMENSION]:
                continue
            if metadata.get("max_sequence_length") not in (None, MAX_SEQUENCE_LENGTH):
                continue
            if metadata.get("normalization") is not NORMALIZE_EMBEDDINGS:
                continue
            path = meta_path.with_suffix("").with_suffix(".npy")
            if not path.exists():
                continue
            embeddings = np.load(path).astype(np.float32)
            self._validate_embeddings(role, embeddings)
            return embeddings, path
        return None

    def _validate_embeddings(self, role: str, embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(f"{role} embeddings have incompatible shape {embeddings.shape}")
        if not np.isfinite(embeddings).all():
            raise ValueError(f"{role} embeddings contain non-finite values")

    def _record_event(self, role: str, ids: list[str], cache_hit: bool, cache_path: Path) -> None:
        self.cache_events.append(
            {
                "role": role,
                "count": len(ids),
                "cache_hit": cache_hit,
                "cache_path": str(cache_path),
            }
        )

    @staticmethod
    def _role(prompt_type: Any | None) -> str:
        value = getattr(prompt_type, "value", str(prompt_type or "")).lower()
        return "query" if "query" in value else "document"

    @staticmethod
    def _extract_batch(batch: Any, role: str) -> tuple[list[str], list[str]]:
        if not isinstance(batch, dict):
            texts = [str(item) for item in batch]
            return [str(index) for index in range(len(texts))], texts

        ids = _to_list(batch.get("id"))
        if role == "query" and "query" in batch:
            texts = _to_list(batch["query"])
        else:
            texts = _to_list(batch.get("text"))
        if not ids:
            ids = [str(index) for index in range(len(texts))]
        return [str(item) for item in ids], [str(text) for text in texts]


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def frozen_encoder_config() -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "gguf_file": GGUF_FILE,
        "quantization": QUANTIZATION,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "query_instruction": QUERY_INSTRUCTION,
        "passage_instruction": DOCUMENT_INSTRUCTION,
        "pooling": POOLING_STRATEGY,
        "normalization": NORMALIZE_EMBEDDINGS,
        "similarity": "cosine/dot-product on normalized embeddings",
    }
