# Samsung PRISM code retrieval

This project targets the strongest legitimate retrieval accuracy on the official CoIR `AppsRetrieval` benchmark for Samsung PRISM Gen AI Hackathon 3.0 Theme 1. The primary screening metrics are **NDCG@10** and **MRR@10**.

## Reproducible benchmark definition

- Python: 3.11 (environment created with 3.11.9)
- MTEB: 2.21.0
- Hugging Face `datasets`: 5.0.1
- NumPy: 2.4.6
- tqdm: 4.70.1
- bm25s: 0.2.13
- Sentence Transformers: 5.1.2
- Transformers: 4.57.1
- protobuf: 6.33.2
- PyTorch (MTEB dependency): 2.14.0+cpu
- Official task name/type: `AppsRetrieval` / `Retrieval`
- Hugging Face dataset: `CoIR-Retrieval/apps`
- Dataset revision pinned by MTEB: `f22508f96b7a36c2415181ed8bb76f76e04ae2d5`
- MTEB evaluation split: `test`
- Evaluation languages/modalities: `eng-Latn`, `python-Code`
- Main score: `ndcg_at_10`

The official MTEB view contains 8,765 corpus documents (5,000 marked `train`, 3,765 marked `test`), 3,765 test queries, and 3,765 binary qrel pairs - one positive document per test query. The sole MTEB evaluation split is `test`; the train-partition documents are corpus candidates, not evaluation queries.

MTEB 2.21.0 declares PyTorch and Sentence Transformers as dependencies even when only loading a task. No model weights, index library, or retrieval implementation is installed by this project.

## Windows PowerShell setup

From the `samsung-code-retrieval` directory:

```powershell
# Select any installed Python 3.11 interpreter. This machine uses pyenv-win:
pyenv local 3.11.9

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell execution policy prevents activation, invoke the environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\inspect_benchmark.py
```

Run the benchmark inspection after activation with:

```powershell
python scripts\inspect_benchmark.py
```

The first run downloads the exact official dataset revision into the Hugging Face cache. It loads and samples metadata only; it does not run retrieval or evaluation.

## Evaluation interface verified in MTEB 2.21.0

An MTEB `SearchProtocol.search` implementation receives the official query dataset and `top_k=1000`, and returns:

```python
{
    "q5001": {"d5001": 12.34, "d1234": 10.25},
    # one inner ranking dictionary per evaluation query
}
```

- Query keys must be exact string IDs from `queries["id"]` (`q...`).
- Inner keys must be exact string IDs from `corpus["id"]` (`d...`).
- Values are numeric relevance scores; larger scores rank earlier.
- Results should cover every evaluation query and contain up to the requested 1,000 documents per query. Only the first *k* ranks affect a metric at cutoff *k*.
- If predictions are saved through MTEB, `AppsRetrieval_predictions.json` wraps that ranking as `{"mteb_model_meta": {...}, "default": {"test": <ranking>}}`.

The official cutoff set is `1, 3, 5, 10, 20, 100, 1000`. MTEB exposes NDCG, MAP, recall, precision, MRR, hit rate, and normalized-AUC variants at these cutoffs. Therefore both NDCG@10 and MRR@10 are directly supported in this installed version.

NDCG@10 is computed by `pytrec_eval` as `ndcg_cut_10` and averaged across queries. MRR@10 is computed by MTEB: for each query, the first document with qrel greater than zero in the top 10 contributes `1 / rank`, or zero if none is present, then values are averaged. Score ties in MRR are resolved by descending document ID to match `pytrec_eval`. AppsRetrieval's test data has one binary-positive qrel per query, so these definitions reduce respectively to discounted gain and reciprocal rank of that one positive document when it appears in the top 10.

## Baseline milestone

Task 2 adds a reusable MTEB-compatible evaluator plus two top-100 retrieval baselines. Retrieval code receives only queries and corpus; qrels are used only after predictions are produced.

Run both baselines with:

```powershell
python scripts\run_baselines.py
```

The dense default is `sentence-transformers/all-MiniLM-L6-v2` with normalized cosine similarity and `max_seq_length=128` by default to keep the full benchmark practical on this CPU-only machine:

```powershell
python scripts\run_baselines.py --skip-bm25 --dense-model sentence-transformers/all-MiniLM-L6-v2 --dense-corpus-batch-size 64 --dense-query-batch-size 128 --dense-max-seq-length 128
```

The BM25 run uses `bm25s` with Lucene-style BM25 defaults (`k1=1.5`, `b=0.75`) and `code_split` tokenization, which preserves code tokens and also splits snake_case/camelCase identifier parts.

Initial code-specialized dense candidates were inspected first. `jinaai/jina-embeddings-v2-base-code` is a better fit conceptually for text-to-code retrieval, but it required Jina remote model code incompatible with the originally installed Transformers 5.x and remained impractical on this CPU-only machine even after pinning Transformers 4.x and truncating sequence length. `flax-sentence-embeddings/st-codesearch-distilroberta-base` and `microsoft/codebert-base` were also attempted, but local Hugging Face model load/download did not complete reliably during this milestone. The MiniLM dense result is therefore an operational dense pipeline baseline, not the final model choice for maximizing score.

Current saved results:

| Method | NDCG@10 | MRR@10 | Recall@10 | HitRate@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 `code_split` | 0.00826 | 0.00686 | 0.01301 | 0.01301 |
| Dense `sentence-transformers/all-MiniLM-L6-v2`, max_seq=128 | 0.05158 | 0.04291 | 0.08021 | 0.08021 |

Additional cutoffs:

| Method | NDCG@1 | NDCG@3 | NDCG@5 | MRR@1 | MRR@3 | MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 `code_split` | 0.00531 | 0.00591 | 0.00680 | 0.00531 | 0.00575 | 0.00626 |
| Dense MiniLM | 0.03028 | 0.03920 | 0.04436 | 0.03028 | 0.03705 | 0.03992 |

Artifacts:

- `results/bm25_code_split/metrics.json`
- `results/bm25_code_split/top100.json`
- `results/dense_sentence-transformers_all-MiniLM-L6-v2/metrics.json`
- `results/dense_sentence-transformers_all-MiniLM-L6-v2/top100.json`
- `data/cache/sentence-transformers_all-MiniLM-L6-v2_maxseq-128_*.npz`

Evaluator sanity checks passed for both baselines:

- exactly one positive qrel per query
- MRR@10 equals average reciprocal rank of the positive document within top 10
- HitRate@10 equals the fraction of queries whose positive document appears in the top 10

## Run API

Start the existing local Jina Code 1.5B GGUF embedding server first:

```powershell
& 'C:\Users\samri\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe' `
  --embedding `
  --model 'data\cache\model_files\jina-code-embeddings-1.5b-Q8_0.gguf' `
  --host 127.0.0.1 `
  --port 8081 `
  --ctx-size 2048 `
  --ubatch-size 512 `
  --pooling last `
  --parallel 1 `
  --device none `
  --gpu-layers 0 `
  --no-op-offload
```

Run the local FastAPI app:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/health
```

## Register repository

```powershell
$body = @{
  repo_path = "data/versioning/real_repos/itsdangerous"
  repo_id = "itsdangerous_demo"
  commit = "2f69e841d2a979c616a55b226e444694f5d9c962"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/repos/register -ContentType "application/json" -Body $body
```

## Search commit

```powershell
$body = @{
  repo_id = "itsdangerous_demo"
  query = "FIPS SHA1 digest method"
  commit = "2f69e841d2a979c616a55b226e444694f5d9c962"
  top_k = 5
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/search -ContentType "application/json" -Body $body
```

## Incrementally update

```powershell
$body = @{
  commit = "31f46a3469dbfb2ecf83dd0c4297c1efc508fcca"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/repos/itsdangerous_demo/update -ContentType "application/json" -Body $body
```

## Search across history

```powershell
$body = @{
  repo_id = "itsdangerous_demo"
  query = "serializer signing"
  top_k = 5
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/search/evolution -ContentType "application/json" -Body $body
```

## View metrics

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/metrics/summary
```

## Demo CLI

The CLI demonstrates registration, version-specific search, incremental update, evolutionary search, and symbol evolution:

```powershell
.\.venv\Scripts\python.exe scripts\demo_system.py --seed-cache
```

API verification without a browser:

```powershell
.\.venv\Scripts\python.exe scripts\verify_api.py
```
