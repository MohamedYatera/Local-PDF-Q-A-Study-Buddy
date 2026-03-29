from __future__ import annotations

from pathlib import Path

from app.schemas import DocumentRecord
from app.vector_store import DocumentStore


class DummyCollection:
    def __init__(self) -> None:
        self.deleted_where = None

    def delete(self, where):
        self.deleted_where = where


def test_delete_document_removes_manifest_entry_and_file(tmp_path: Path):
    store = DocumentStore.__new__(DocumentStore)
    store.collection = DummyCollection()

    pdf_path = tmp_path / "uploaded.pdf"
    pdf_path.write_text("content", encoding="utf-8")

    manifest_path = tmp_path / "documents.json"
    settings = type("Settings", (), {"manifest_path": manifest_path})
    store.settings = settings

    doc = DocumentRecord(
        id="doc-1",
        name="lecture.pdf",
        stored_path=str(pdf_path),
        pages=10,
        chunks=4,
        uploaded_at="2026-03-16T00:00:00Z",
    )
    store._save_documents([doc])

    deleted = store.delete_document("doc-1")

    assert deleted is not None
    assert deleted.id == "doc-1"
    assert store.collection.deleted_where == {"doc_id": "doc-1"}
    assert not pdf_path.exists()
    assert store.list_documents() == []

