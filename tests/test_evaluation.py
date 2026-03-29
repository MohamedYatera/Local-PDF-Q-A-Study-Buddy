from __future__ import annotations

from pathlib import Path

from app.evaluation import run_evaluation
from app.schemas import Citation, EvaluationCitation, EvaluationItem, QueryResponse, RetrievedChunk


class DummyRAGService:
    def __init__(self) -> None:
        self.judge_calls = 0

    def answer_question(self, question: str, top_k: int | None = None) -> QueryResponse:
        chunk = RetrievedChunk(
            source_id="S1",
            doc_id="doc-1",
            doc_name="dsa-lecture.pdf",
            page=3,
            section_title="Binary Search",
            text="Binary search runs in O(log n).",
            similarity_score=0.8,
        )
        citation = Citation(
            source_id="S1",
            doc_id="doc-1",
            doc_name="dsa-lecture.pdf",
            page=3,
            section_title="Binary Search",
            quote="Binary search runs in O(log n).",
            similarity_score=0.8,
        )
        return QueryResponse(
            answer="Binary search runs in O(log n). [S1]",
            citations=[citation],
            retrieved_chunks=[chunk],
            enough_evidence=True,
            latency_ms=100.0,
        )

    def judge_support(self, answer: str, retrieved_chunks: list[RetrievedChunk]):
        self.judge_calls += 1
        raise AssertionError("judge_support should not run in fast mode")


def test_run_evaluation_fast_mode_skips_support_check(tmp_path: Path):
    settings = type("Settings", (), {"reports_dir": tmp_path})
    rag_service = DummyRAGService()
    items = [
        EvaluationItem(
            question="What is binary search complexity?",
            expected_answer_contains=["O(log n)"],
            expected_citations=[EvaluationCitation(doc_name="dsa-lecture.pdf", page=3)],
        )
    ]

    summary = run_evaluation(
        "test.json",
        items,
        rag_service,
        settings,
        support_check_enabled=False,
    )

    assert summary.support_check_enabled is False
    assert summary.unsupported_claim_rate is None
    assert summary.cases[0].supported is None
    assert rag_service.judge_calls == 0

