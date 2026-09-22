# Architecture

This project has two verified layers:

- P0: high-accuracy code retrieval for the official CoIR AppsRetrieval benchmark.
- P1: version-aware retrieval for Git repositories, with incremental reuse across commits.

The API layer in `src/api/` is intentionally thin. It exposes the existing P0/P1 implementation for demos and local evaluation; it does not redesign retrieval, chunking, embedding, scoring, or benchmark evaluation.

## P1 pipeline

```text
Git repository
      ↓
File manifest
      ↓
Structural chunks
      ↓
content_id
      ↓
Embedding cache
      ↓
Version-aware vector index
      ↓
Version-specific search
      ↓
Evolutionary retrieval
```

## Identifiers

`content_id` is the deterministic identity of a code chunk's content under the current chunking rules. If the same code appears in multiple commits, files, or paths, it has the same `content_id`.

`version_id` is the deterministic identity of one occurrence of a chunk in a specific repository, commit, path, and source location. The same `content_id` can have many `version_id` occurrences.

Embeddings are stored once per compatible `content_id` and embedding configuration. Version occurrences reference reusable vectors through their `content_id`, so unchanged code does not need to be embedded again.

## Version index

The version-aware vector index stores unique vectors by content ID and active commit occurrences separately. Updating from one commit to another:

1. scans the target commit,
2. builds deterministic file and chunk manifests,
3. compares file/chunk changes by content hash,
4. reuses existing content embeddings,
5. inserts only new content vectors,
6. updates active commit occurrence metadata,
7. tombstones removed active occurrences without deleting historical metadata.

## Evolution metadata

Evolution metadata is built from the chronological chunk manifests. It records chains keyed by path, chunk type, and symbol, with deterministic transitions:

- `added`
- `unchanged`
- `modified`
- `deleted`
- `reintroduced`

Line-level diff counts are deterministic and computed without LLM summaries.

## API layer

`src/api/app.py` exposes:

- repository registration,
- incremental update,
- version-specific search,
- evolutionary search,
- symbol evolution,
- frozen benchmark metric summary,
- health/status.

The API stores a small local registry in `data/api/registry.json`. Heavy data remains in local artifact directories under `data/api/repos/<repo_id>/`.
