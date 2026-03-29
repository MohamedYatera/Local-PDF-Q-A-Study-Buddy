from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from app.config import Settings
from app.document_processor import ProcessedDocument, process_pdf
from app.ollama_client import OllamaService
from app.schemas import DocumentRecord, RetrievedChunk


class DocumentStore:
    def __init__(self, settings: Settings, ollama: OllamaService) -> None:
        self.settings = settings
        self.ollama = ollama
        self.client = chromadb.PersistentClient(
            path=str(settings.vector_db_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                chroma_product_telemetry_impl="app.chroma_telemetry.NoOpTelemetryClient",
                chroma_telemetry_impl="app.chroma_telemetry.NoOpTelemetryClient",
            ),
        )
        self.collection: Collection = self.client.get_or_create_collection(name="study_buddy_chunks")

    def list_documents(self) -> list[DocumentRecord]:
        if not self.settings.manifest_path.exists():
            return []

        payload = json.loads(self.settings.manifest_path.read_text(encoding="utf-8"))
        return [DocumentRecord.model_validate(item) for item in payload]

    def _save_documents(self, documents: list[DocumentRecord]) -> None:
        serialized = [document.model_dump() for document in documents]
        self.settings.manifest_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

    def _replace_document_record(self, record: DocumentRecord) -> None:
        documents = [doc for doc in self.list_documents() if doc.id != record.id]
        documents.append(record)
        documents.sort(key=lambda item: item.uploaded_at, reverse=True)
        self._save_documents(documents)

    def delete_document(self, doc_id: str) -> DocumentRecord | None:
        documents = self.list_documents()
        target = next((doc for doc in documents if doc.id == doc_id), None)
        if target is None:
            return None

        self.collection.delete(where={"doc_id": doc_id})

        stored_path = Path(target.stored_path)
        if stored_path.exists():
            stored_path.unlink()

        self._save_documents([doc for doc in documents if doc.id != doc_id])
        return target

    def ingest_pdf(self, saved_pdf_path: Path, original_name: str) -> DocumentRecord:
        processed = process_pdf(
            saved_pdf_path,
            chunk_size_words=self.settings.chunk_size_words,
            overlap_words=self.settings.chunk_overlap_words,
            display_name=original_name,
        )

        self.collection.delete(where={"doc_id": processed.doc_id})

        chunk_ids = [chunk.chunk_id for chunk in processed.chunks]
        chunk_texts = [chunk.text for chunk in processed.chunks]
        metadatas = [chunk.metadata for chunk in processed.chunks]
        embeddings = self.ollama.embed(chunk_texts)

        if chunk_ids:
            self.collection.add(
                ids=chunk_ids,
                documents=chunk_texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )

        record = DocumentRecord(
            id=processed.doc_id,
            name=processed.doc_name,
            stored_path=str(saved_pdf_path),
            pages=processed.pages,
            chunks=len(processed.chunks),
            uploaded_at=datetime.now(timezone.utc).isoformat(),
        )
        self._replace_document_record(record)
        return record

    def query(self, question: str, top_k: int) -> list[RetrievedChunk]:
        query_embeddings = self.ollama.embed([question])
        result = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for index, (text, metadata, distance) in enumerate(zip(documents, metadatas, distances), start=1):
            similarity_score = 1 / (1 + float(distance))
            retrieved.append(
                RetrievedChunk(
                    source_id=f"S{index}",
                    doc_id=str(metadata["doc_id"]),
                    doc_name=str(metadata["doc_name"]),
                    page=int(metadata["page"]),
                    section_title=str(metadata["section_title"]),
                    text=str(text),
                    similarity_score=round(similarity_score, 4),
                )
            )

        return retrieved
