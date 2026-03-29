from __future__ import annotations

import shutil
from datetime import datetime

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import ensure_directories, get_settings
from app.evaluation import load_evaluation_items, run_evaluation
from app.ollama_client import OllamaService, OllamaServiceError
from app.rag_service import RAGService
from app.schemas import HealthResponse, QueryRequest, QueryResponse
from app.vector_store import DocumentStore

settings = get_settings()
ensure_directories(settings)

ollama_service = OllamaService(settings)
document_store = DocumentStore(settings, ollama_service)
rag_service = RAGService(settings, document_store, ollama_service)

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(settings.static_dir / "index.html")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        ollama_reachable=ollama_service.health_check(),
        generation_model=settings.generation_model,
        embedding_model=settings.embedding_model,
        indexed_documents=len(document_store.list_documents()),
    )


@app.get("/api/documents")
def list_documents():
    return document_store.list_documents()


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    deleted = document_store.delete_document(doc_id)
    if deleted is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": True, "document": deleted}


@app.post("/api/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    ingested = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF.")

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        destination = settings.uploads_dir / f"{timestamp}-{file.filename}"
        with destination.open("wb") as output_stream:
            shutil.copyfileobj(file.file, output_stream)

        try:
            record = document_store.ingest_pdf(destination, original_name=file.filename)
        except OllamaServiceError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        ingested.append(record)

    return ingested


@app.post("/api/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    if not document_store.list_documents():
        raise HTTPException(status_code=400, detail="Upload at least one PDF before asking questions.")

    try:
        return rag_service.answer_question(request.question, top_k=request.top_k)
    except OllamaServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/evaluations/run")
async def evaluate_dataset(dataset: UploadFile = File(...), fast_mode: bool = Form(True)):
    if not document_store.list_documents():
        raise HTTPException(status_code=400, detail="Upload at least one PDF before running an evaluation.")

    if not dataset.filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Evaluation dataset must be a JSON file.")

    raw_bytes = await dataset.read()
    try:
        items = load_evaluation_items(raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    summary = run_evaluation(
        dataset.filename,
        items,
        rag_service,
        settings,
        support_check_enabled=not fast_mode,
    )
    return summary


@app.get("/api/reports/{report_name}")
def get_report(report_name: str):
    report_path = settings.reports_dir / report_name
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(report_path)


@app.on_event("shutdown")
def shutdown_event() -> None:
    ollama_service.close()
