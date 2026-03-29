from __future__ import annotations

import re
import time

from app.config import Settings
from app.ollama_client import OllamaService, OllamaServiceError
from app.schemas import Citation, OllamaAnswerResult, OllamaJudgeResult, QueryResponse, RetrievedChunk
from app.vector_store import DocumentStore

SOURCE_PATTERN = re.compile(r"\[(S\d+)\]")


class RAGService:
    def __init__(self, settings: Settings, store: DocumentStore, ollama: OllamaService) -> None:
        self.settings = settings
        self.store = store
        self.ollama = ollama

    def answer_question(self, question: str, top_k: int | None = None) -> QueryResponse:
        started = time.perf_counter()
        top_k = top_k or self.settings.default_top_k
        retrieved = self.store.query(question, top_k=top_k)

        if not self._has_enough_evidence(retrieved):
            return QueryResponse(
                answer="Not enough evidence in the documents.",
                citations=[],
                retrieved_chunks=retrieved,
                enough_evidence=False,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        try:
            model_output = self._generate_answer(question, retrieved)
        except OllamaServiceError:
            return QueryResponse(
                answer="Not enough evidence in the documents.",
                citations=[],
                retrieved_chunks=retrieved,
                enough_evidence=False,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        if not model_output.enough_evidence:
            answer = "Not enough evidence in the documents."
            citations: list[Citation] = []
        else:
            answer = model_output.answer.strip() or "Not enough evidence in the documents."
            citations = self._resolve_citations(answer, retrieved, model_output.citations)
            if not citations:
                answer = "Not enough evidence in the documents."
            elif not SOURCE_PATTERN.search(answer):
                answer = f"{answer}\n\nSources: {' '.join(f'[{citation.source_id}]' for citation in citations)}"

        return QueryResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved,
            enough_evidence=answer != "Not enough evidence in the documents.",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def judge_support(self, answer: str, retrieved_chunks: list[RetrievedChunk]) -> OllamaJudgeResult:
        if answer.strip() == "Not enough evidence in the documents.":
            return OllamaJudgeResult(supported=True, unsupported_claims=[])

        context = self._build_context(retrieved_chunks)
        system_prompt = (
            "You are a strict RAG evaluator. Determine whether every factual claim in the answer is "
            "supported by the provided context. Return JSON with keys supported and unsupported_claims."
        )
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Answer:\n{answer}\n\n"
            'Return JSON in the form {"supported": true, "unsupported_claims": []}. '
            "List unsupported claims verbatim when possible."
        )

        payload = self.ollama.generate_json(system_prompt, user_prompt)
        return OllamaJudgeResult.model_validate(payload)

    def _has_enough_evidence(self, retrieved: list[RetrievedChunk]) -> bool:
        if not retrieved:
            return False

        relevant = [chunk for chunk in retrieved if chunk.similarity_score >= self.settings.min_similarity_score]
        top_score = max(chunk.similarity_score for chunk in retrieved)
        return len(relevant) >= self.settings.min_relevant_chunks or top_score >= self.settings.min_similarity_score

    def _generate_answer(self, question: str, retrieved: list[RetrievedChunk]) -> OllamaAnswerResult:
        context = self._build_context(retrieved)
        system_prompt = (
            "You are a study assistant that must answer only from the provided context. "
            "Do not add outside knowledge. Every factual sentence must include one or more inline source tags "
            "like [S1]. If the context is insufficient, answer exactly 'Not enough evidence in the documents.' "
            "and set enough_evidence to false."
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "Return strict JSON with this shape:\n"
            '{'
            '"answer": "short grounded answer with [S1] citations", '
            '"enough_evidence": true, '
            '"citations": ["S1", "S2"], '
            '"notes": {"confidence": "low|medium|high"}'
            '}'
        )
        payload = self.ollama.generate_json(system_prompt, user_prompt)
        return OllamaAnswerResult.model_validate(payload)

    def _build_context(self, retrieved: list[RetrievedChunk]) -> str:
        blocks: list[str] = []
        for chunk in retrieved:
            blocks.append(
                "\n".join(
                    [
                        f"[{chunk.source_id}] Document: {chunk.doc_name}",
                        f"Page: {chunk.page}",
                        f"Section: {chunk.section_title}",
                        f"Similarity: {chunk.similarity_score}",
                        f"Text: {chunk.text}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def _resolve_citations(
        self,
        answer: str,
        retrieved: list[RetrievedChunk],
        declared_citations: list[str] | None = None,
    ) -> list[Citation]:
        seen: set[str] = set()
        chunks_by_source = {chunk.source_id: chunk for chunk in retrieved}
        citations: list[Citation] = []
        requested_source_ids = SOURCE_PATTERN.findall(answer)
        if declared_citations:
            requested_source_ids.extend(declared_citations)

        for source_id in requested_source_ids:
            if source_id in seen or source_id not in chunks_by_source:
                continue
            seen.add(source_id)
            chunk = chunks_by_source[source_id]
            citations.append(
                Citation(
                    source_id=source_id,
                    doc_id=chunk.doc_id,
                    doc_name=chunk.doc_name,
                    page=chunk.page,
                    section_title=chunk.section_title,
                    quote=chunk.text[:280].strip(),
                    similarity_score=chunk.similarity_score,
                )
            )

        return citations
