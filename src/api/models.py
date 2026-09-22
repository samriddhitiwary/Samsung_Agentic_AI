from __future__ import annotations

from pydantic import BaseModel, Field


class RegisterRepoRequest(BaseModel):
    repo_path: str = Field(..., description="Local Git repository path.")
    repo_id: str | None = Field(default=None, description="Stable repository ID. Defaults to repo directory name.")
    commit: str = Field(default="HEAD", description="Git commit/ref to index.")


class UpdateRepoRequest(BaseModel):
    commit: str = Field(..., description="Target Git commit/ref.")


class SearchRequest(BaseModel):
    repo_id: str
    query: str
    commit: str
    top_k: int = Field(default=10, ge=1, le=100)


class EvolutionSearchRequest(BaseModel):
    repo_id: str
    query: str
    start_commit: str | None = None
    end_commit: str | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    include_evolution_context: bool = Field(
        default=False,
        description="Attach deterministic state-level evolution metadata to grouped results.",
    )
