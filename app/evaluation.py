from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from app.config import Settings
from app.rag_service import RAGService
from app.schemas import EvaluationCaseResult, EvaluationItem, EvaluationSummary


def load_evaluation_items(raw_bytes: bytes) -> list[EvaluationItem]:
    payload = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Evaluation dataset must be a JSON array.")
    return [EvaluationItem.model_validate(item) for item in payload]


def run_evaluation(
    dataset_name: str,
    items: list[EvaluationItem],
    rag_service: RAGService,
    settings: Settings,
    support_check_enabled: bool = True,
) -> EvaluationSummary:
    cases: list[EvaluationCaseResult] = []

    for item in items:
        response = rag_service.answer_question(item.question)
        keyword_score = _keyword_score(response.answer, item.expected_answer_contains)
        citation_recall, citation_precision = _citation_scores(response.citations, item.expected_citations)
        if support_check_enabled:
            support = rag_service.judge_support(response.answer, response.retrieved_chunks)
            supported = support.supported
            unsupported_claims = support.unsupported_claims
        else:
            supported = None
            unsupported_claims = []

        if item.allow_not_enough_evidence and response.answer == "Not enough evidence in the documents.":
            keyword_score = 1.0

        cases.append(
            EvaluationCaseResult(
                question=item.question,
                answer=response.answer,
                latency_ms=response.latency_ms,
                keyword_score=keyword_score,
                citation_recall=citation_recall,
                citation_precision=citation_precision,
                supported=supported,
                unsupported_claims=unsupported_claims,
                citations=response.citations,
            )
        )

    summary = EvaluationSummary(
        dataset_name=dataset_name,
        support_check_enabled=support_check_enabled,
        cases=cases,
        total_questions=len(cases),
        average_keyword_score=_average(case.keyword_score for case in cases),
        average_citation_recall=_average(case.citation_recall for case in cases),
        average_citation_precision=_average(case.citation_precision for case in cases),
        unsupported_claim_rate=(
            _average(0 if case.supported else 1 for case in cases if case.supported is not None)
            if support_check_enabled
            else None
        ),
        average_latency_ms=_average(case.latency_ms for case in cases),
        report_path="",
    )

    report_path = write_report(summary, settings)
    return summary.model_copy(update={"report_path": str(report_path)})


def write_report(summary: EvaluationSummary, settings: Settings) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = settings.reports_dir / f"evaluation-{timestamp}.md"

    lines = [
        f"# Evaluation Report: {summary.dataset_name}",
        "",
        f"- Total questions: {summary.total_questions}",
        f"- Average keyword score: {summary.average_keyword_score:.2f}",
        f"- Average citation recall: {summary.average_citation_recall:.2f}",
        f"- Average citation precision: {summary.average_citation_precision:.2f}",
        (
            f"- Unsupported claim rate: {summary.unsupported_claim_rate:.2f}"
            if summary.support_check_enabled and summary.unsupported_claim_rate is not None
            else "- Unsupported claim rate: skipped in fast mode"
        ),
        f"- Average latency (ms): {summary.average_latency_ms:.2f}",
        f"- Support check enabled: {summary.support_check_enabled}",
        "",
        "## Case Details",
        "",
    ]

    for case in summary.cases:
        lines.extend(
            [
                f"### {case.question}",
                "",
                f"- Answer: {case.answer}",
                f"- Keyword score: {case.keyword_score:.2f}",
                f"- Citation recall: {case.citation_recall:.2f}",
                f"- Citation precision: {case.citation_precision:.2f}",
                f"- Supported: {case.supported if case.supported is not None else 'skipped in fast mode'}",
                f"- Latency (ms): {case.latency_ms:.2f}",
            ]
        )
        if case.unsupported_claims:
            lines.append(f"- Unsupported claims: {', '.join(case.unsupported_claims)}")
        if case.citations:
            lines.append(
                "- Citations: "
                + ", ".join(f"{citation.doc_name} p.{citation.page} ({citation.source_id})" for citation in case.citations)
            )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _keyword_score(answer: str, expected_terms: list[str]) -> float:
    if not expected_terms:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for term in expected_terms if term.lower() in answer_lower)
    return hits / len(expected_terms)


def _citation_scores(predicted, expected) -> tuple[float, float]:
    if not expected:
        return (1.0, 1.0 if not predicted else 0.0)

    expected_pairs = {(item.doc_name.lower(), item.page) for item in expected}
    predicted_pairs = {(item.doc_name.lower(), item.page) for item in predicted}

    overlap = expected_pairs & predicted_pairs
    recall = len(overlap) / len(expected_pairs)
    precision = len(overlap) / len(predicted_pairs) if predicted_pairs else 0.0
    return recall, precision


def _average(values) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)
