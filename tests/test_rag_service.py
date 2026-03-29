from __future__ import annotations

from app.rag_service import RAGService
from app.schemas import OllamaAnswerResult, RetrievedChunk


class DummySettings:
    default_top_k = 6
    min_relevant_chunks = 1
    min_similarity_score = 0.3


class DummyStore:
    def __init__(self, retrieved):
        self._retrieved = retrieved

    def query(self, question: str, top_k: int):
        return self._retrieved


class DummyOllama:
    def __init__(self, answer_result: OllamaAnswerResult):
        self.answer_result = answer_result

    def generate_json(self, system_prompt: str, user_prompt: str):
        return self.answer_result.model_dump()


def build_chunk(source_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        source_id=source_id,
        doc_id="doc-1",
        doc_name="lecture.pdf",
        page=4,
        section_title="Key Idea",
        text="Important grounded text from the lecture slides.",
        similarity_score=score,
    )


def test_uses_declared_citations_when_answer_text_has_no_inline_tags():
    retrieved = [build_chunk("S1", 0.61)]
    service = RAGService(
        DummySettings(),
        DummyStore(retrieved),
        DummyOllama(
            OllamaAnswerResult(
                answer="The concept is defined directly in the lecture.",
                enough_evidence=True,
                citations=["S1"],
                notes={},
            )
        ),
    )

    response = service.answer_question("What is the concept?")

    assert response.enough_evidence is True
    assert response.citations[0].source_id == "S1"
    assert "[S1]" in response.answer


def test_single_strong_chunk_counts_as_enough_evidence():
    retrieved = [build_chunk("S1", 0.52)]
    service = RAGService(
        DummySettings(),
        DummyStore(retrieved),
        DummyOllama(
            OllamaAnswerResult(
                answer="It is supported by one strong chunk. [S1]",
                enough_evidence=True,
                citations=["S1"],
                notes={},
            )
        ),
    )

    response = service.answer_question("Explain the point.")

    assert response.enough_evidence is True
    assert response.answer.startswith("It is supported")

