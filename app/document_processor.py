from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: dict[str, str | int]


@dataclass
class ProcessedDocument:
    doc_id: str
    doc_name: str
    pages: int
    chunks: list[ChunkRecord]


def build_doc_id(file_bytes: bytes) -> str:
    return hashlib.sha1(file_bytes).hexdigest()[:16]


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_section_title(page_text: str, page_number: int) -> str:
    for raw_line in page_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if len(line) > 100:
            continue
        if line.isupper() or line.istitle() or len(line.split()) <= 8:
            return line
        break
    return f"Page {page_number}"


def split_into_chunks(text: str, chunk_size_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_size_words:
        return [" ".join(words)]

    chunks: list[str] = []
    start = 0
    stride = max(1, chunk_size_words - overlap_words)

    while start < len(words):
        end = min(len(words), start + chunk_size_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start += stride

    return chunks


def process_pdf(
    pdf_path: Path,
    chunk_size_words: int,
    overlap_words: int,
    display_name: str | None = None,
) -> ProcessedDocument:
    file_bytes = pdf_path.read_bytes()
    doc_id = build_doc_id(file_bytes)
    doc_name = display_name or pdf_path.name
    chunks: list[ChunkRecord] = []

    with fitz.open(pdf_path) as pdf:
        for page_index in range(pdf.page_count):
            page_number = page_index + 1
            page_text = clean_text(pdf.load_page(page_index).get_text("text"))
            if not page_text:
                continue

            section_title = infer_section_title(page_text, page_number)
            page_chunks = split_into_chunks(page_text, chunk_size_words, overlap_words)

            for chunk_index, chunk_text in enumerate(page_chunks, start=1):
                chunk_id = f"{doc_id}-p{page_number}-c{chunk_index}"
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        metadata={
                            "doc_id": doc_id,
                            "doc_name": doc_name,
                            "page": page_number,
                            "section_title": section_title,
                            "chunk_index": chunk_index,
                        },
                    )
                )

        return ProcessedDocument(
            doc_id=doc_id,
            doc_name=doc_name,
            pages=pdf.page_count,
            chunks=chunks,
        )
