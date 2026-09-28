from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import EvolutionSearchRequest, QueryRequest, RegisterRepoRequest, SearchRequest, UpdateRepoRequest
from src.api.service import ApiService


app = FastAPI(
    title="Samsung Code Retrieval Demo API",
    version="0.1.0",
    description="Thin API over verified P0 retrieval metrics and P1 version-aware code retrieval.",
)
service = ApiService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return service.health()


@app.post("/repos/register")
def register_repo(request: RegisterRepoRequest) -> dict:
    return service.register_repo(repo_path=request.repo_path, repo_id=request.repo_id, commit=request.commit)


@app.post("/repos/{repo_id}/update")
def update_repo(repo_id: str, request: UpdateRepoRequest) -> dict:
    return service.update_repo(repo_id=repo_id, commit=request.commit)


@app.post("/search")
def search(request: SearchRequest) -> dict:
    return service.search(repo_id=request.repo_id, query=request.query, commit=request.commit, top_k=request.top_k)


@app.post("/query")
def query(request: QueryRequest) -> dict:
    return service.query(repo_id=request.repo_id, query=request.query, commit=request.commit, top_k=request.top_k)


@app.post("/search/evolution")
def search_evolution(request: EvolutionSearchRequest) -> dict:
    return service.evolution_search(
        repo_id=request.repo_id,
        query=request.query,
        start_commit=request.start_commit,
        end_commit=request.end_commit,
        top_k=request.top_k,
        include_evolution_context=request.include_evolution_context,
    )


@app.get("/repos/{repo_id}/symbols/evolution")
def symbol_evolution(repo_id: str, symbol: str, path: str | None = None) -> dict:
    return service.symbol_evolution(repo_id=repo_id, symbol=symbol, path=path)


@app.get("/metrics/summary")
def metrics_summary() -> dict:
    return service.metrics_summary()
