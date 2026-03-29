from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    id: str
    name: str
    stored_path: str
    pages: int
    chunks: int
    uploaded_at: str


class Citation(BaseModel):
    source_id: str
    doc_id: str
    doc_name: str
    page: int
    section_title: str
    quote: str
    similarity_score: float


class RetrievedChunk(BaseModel):
    source_id: str
    doc_id: str
    doc_name: str
    page: int
    section_title: str
    text: str
    similarity_score: float


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int | None = Field(default=None, ge=1, le=12)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    enough_evidence: bool
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    generation_model: str
    embedding_model: str
    indexed_documents: int


class EvaluationCitation(BaseModel):
    doc_name: str
    page: int


class EvaluationItem(BaseModel):
    question: str
    expected_answer_contains: list[str] = Field(default_factory=list)
    expected_citations: list[EvaluationCitation] = Field(default_factory=list)
    allow_not_enough_evidence: bool = False


class EvaluationCaseResult(BaseModel):
    question: str
    answer: str
    latency_ms: float
    keyword_score: float
    citation_recall: float
    citation_precision: float
    supported: bool | None
    unsupported_claims: list[str] = Field(default_factory=list)
    citations: list[Citation]


class EvaluationSummary(BaseModel):
    dataset_name: str
    support_check_enabled: bool
    cases: list[EvaluationCaseResult]
    total_questions: int
    average_keyword_score: float
    average_citation_recall: float
    average_citation_precision: float
    unsupported_claim_rate: float | None
    average_latency_ms: float
    report_path: str


class OllamaJudgeResult(BaseModel):
    supported: bool
    unsupported_claims: list[str] = Field(default_factory=list)


class OllamaAnswerResult(BaseModel):
    answer: str
    enough_evidence: bool
    citations: list[str] = Field(default_factory=list)
    notes: dict[str, Any] = Field(default_factory=dict)
