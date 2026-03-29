from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    app_name: str
    ollama_base_url: str
    generation_model: str
    embedding_model: str
    ollama_embed_timeout_seconds: float
    ollama_generation_timeout_seconds: float
    chunk_size_words: int
    chunk_overlap_words: int
    default_top_k: int
    evaluation_top_k: int
    min_relevant_chunks: int
    min_similarity_score: float
    uploads_dir: Path
    vector_db_dir: Path
    reports_dir: Path
    manifest_path: Path
    static_dir: Path


def get_settings() -> Settings:
    storage_dir = ROOT_DIR / "storage"
    return Settings(
        app_name="Local PDF Q&A Study Buddy",
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        generation_model=os.getenv("OLLAMA_GENERATION_MODEL", "llama3.2"),
        embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        ollama_embed_timeout_seconds=float(os.getenv("OLLAMA_EMBED_TIMEOUT_SECONDS", "120")),
        ollama_generation_timeout_seconds=float(os.getenv("OLLAMA_GENERATION_TIMEOUT_SECONDS", "300")),
        chunk_size_words=int(os.getenv("CHUNK_SIZE_WORDS", "350")),
        chunk_overlap_words=int(os.getenv("CHUNK_OVERLAP_WORDS", "70")),
        default_top_k=int(os.getenv("DEFAULT_TOP_K", "6")),
        evaluation_top_k=int(os.getenv("EVALUATION_TOP_K", "3")),
        min_relevant_chunks=int(os.getenv("MIN_RELEVANT_CHUNKS", "1")),
        min_similarity_score=float(os.getenv("MIN_SIMILARITY_SCORE", "0.3")),
        uploads_dir=storage_dir / "uploads",
        vector_db_dir=storage_dir / "chroma",
        reports_dir=storage_dir / "reports",
        manifest_path=storage_dir / "documents.json",
        static_dir=ROOT_DIR / "app" / "static",
    )


def ensure_directories(settings: Settings) -> None:
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.vector_db_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    settings.static_dir.mkdir(parents=True, exist_ok=True)
