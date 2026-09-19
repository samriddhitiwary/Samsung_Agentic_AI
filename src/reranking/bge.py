"""BGE cross-encoder reranking helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "BAAI/bge-reranker-v2-m3"
ONNX_MODEL_NAME = "onnx-community/bge-reranker-v2-m3-ONNX"
ONNX_MODEL_FILE = "onnx/model_int8.onnx"


@dataclass(frozen=True)
class BGEReranker:
    model_name: str
    max_length: int
    batch_size: int
    device: str
    tokenizer: Any
    model: Any


@dataclass(frozen=True)
class BGEOnnxReranker:
    model_name: str
    onnx_file: str
    max_length: int
    batch_size: int
    tokenizer: Any
    session: Any
    input_names: set[str]


def load_bge_reranker(
    *,
    model_name: str = MODEL_NAME,
    max_length: int = 512,
    batch_size: int = 4,
    device: str = "cpu",
) -> BGEReranker:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return BGEReranker(
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        tokenizer=tokenizer,
        model=model,
    )


def load_bge_onnx_reranker(
    *,
    model_name: str = ONNX_MODEL_NAME,
    onnx_file: str = ONNX_MODEL_FILE,
    max_length: int = 512,
    batch_size: int = 4,
) -> BGEOnnxReranker:
    import onnxruntime as ort

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_path = hf_hub_download(repo_id=model_name, filename=onnx_file)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_names = {item.name for item in session.get_inputs()}
    return BGEOnnxReranker(
        model_name=model_name,
        onnx_file=onnx_file,
        max_length=max_length,
        batch_size=batch_size,
        tokenizer=tokenizer,
        session=session,
        input_names=input_names,
    )


def score_pairs(
    reranker: BGEReranker,
    pairs: list[tuple[str, str]],
    *,
    desc: str = "BGE score pairs",
) -> list[float]:
    scores: list[float] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(pairs), reranker.batch_size), desc=desc):
            batch_pairs = pairs[start : start + reranker.batch_size]
            inputs = reranker.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=reranker.max_length,
            )
            inputs = {key: value.to(reranker.device) for key, value in inputs.items()}
            logits = reranker.model(**inputs, return_dict=True).logits.view(-1).float()
            scores.extend(float(score) for score in logits.cpu().tolist())
    return scores


def score_pairs_onnx(
    reranker: BGEOnnxReranker,
    pairs: list[tuple[str, str]],
    *,
    desc: str = "BGE ONNX score pairs",
) -> list[float]:
    scores: list[float] = []
    for start in tqdm(range(0, len(pairs), reranker.batch_size), desc=desc):
        batch_pairs = pairs[start : start + reranker.batch_size]
        encoded = reranker.tokenizer(
            batch_pairs,
            padding=True,
            truncation=True,
            return_tensors="np",
            max_length=reranker.max_length,
        )
        inputs = {
            key: value.astype(np.int64, copy=False)
            for key, value in encoded.items()
            if key in reranker.input_names
        }
        outputs = reranker.session.run(None, inputs)
        logits = np.asarray(outputs[0]).reshape(-1).astype(np.float32)
        scores.extend(float(score) for score in logits.tolist())
    return scores


def timed_score_pairs(
    reranker: BGEReranker,
    pairs: list[tuple[str, str]],
    *,
    desc: str = "BGE score pairs",
) -> tuple[list[float], float]:
    start = perf_counter()
    scores = score_pairs(reranker, pairs, desc=desc)
    return scores, perf_counter() - start


def timed_score_pairs_onnx(
    reranker: BGEOnnxReranker,
    pairs: list[tuple[str, str]],
    *,
    desc: str = "BGE ONNX score pairs",
) -> tuple[list[float], float]:
    start = perf_counter()
    scores = score_pairs_onnx(reranker, pairs, desc=desc)
    return scores, perf_counter() - start


def rerank_top_k(
    *,
    query_text: str,
    candidate_doc_ids: list[str],
    doc_text_by_id: dict[str, str],
    scores: Iterable[float],
) -> list[tuple[str, float]]:
    scored = [
        (doc_id, float(score))
        for doc_id, score in zip(candidate_doc_ids, scores, strict=True)
        if doc_id in doc_text_by_id
    ]
    return sorted(scored, key=lambda item: (item[1], item[0]), reverse=True)
