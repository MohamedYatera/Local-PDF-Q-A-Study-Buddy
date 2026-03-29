# Local PDF Q&A Study Buddy

Local-first study assistant for course PDFs. It indexes uploaded lecture notes, retrieves relevant chunks with embeddings, answers only from retrieved context, and returns inline source tags plus page-level citation cards.

## What it includes

- FastAPI backend for PDF ingestion, vector search, grounded answer generation, and evaluation
- Browser UI for uploading PDFs, asking questions, inspecting retrieved chunks, and running a JSON evaluation set
- Local Ollama integration for both embeddings and answer generation
- Evaluation harness that measures keyword coverage, citation overlap, unsupported-claim rate, and latency
- Markdown report output saved under `storage/reports/`

## Local architecture

1. Upload one or more PDFs.
2. Extract text page by page with PyMuPDF.
3. Split text into overlapping chunks with page and section metadata.
4. Create embeddings with Ollama and store vectors in persistent Chroma DB.
5. Retrieve top-k chunks for a question.
6. Prompt the local LLM to answer only from retrieved context and cite source IDs like `[S1]`.
7. Map those source IDs back to document names and page numbers in the API/UI.

## Requirements

- Python 3.10+
- Ollama installed locally
- These Ollama models pulled before running the app:

```powershell
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Configuration

Environment variables live in `.env`.

- `OLLAMA_BASE_URL`: default `http://127.0.0.1:11434`
- `OLLAMA_GENERATION_MODEL`: default `llama3.1:8b`
- `OLLAMA_EMBEDDING_MODEL`: default `nomic-embed-text`
- `OLLAMA_EMBED_TIMEOUT_SECONDS`: default `120`
- `OLLAMA_GENERATION_TIMEOUT_SECONDS`: default `300`
- `CHUNK_SIZE_WORDS`: default `350`
- `CHUNK_OVERLAP_WORDS`: default `70`
- `DEFAULT_TOP_K`: default `6`
- `MIN_RELEVANT_CHUNKS`: default `2`
- `MIN_SIMILARITY_SCORE`: default `0.45`

## Evaluation dataset format

Use a JSON array like [`sample_eval/evaluation-template.json`](/c:/Users/yater/Documents/GitHub/Local%20PDF%20Q&A%20Study%20Buddy/sample_eval/evaluation-template.json).

Each item supports:

```json
{
  "question": "What is the instructor's definition of reinforcement learning?",
  "expected_answer_contains": ["reinforcement learning", "reward"],
  "expected_citations": [
    { "doc_name": "lecture-06.pdf", "page": 9 }
  ],
  "allow_not_enough_evidence": false
}
```

Notes:

- `expected_answer_contains` is a simple keyword-based correctness check.
- `expected_citations` is used for citation precision/recall.
- `allow_not_enough_evidence` lets you mark questions that should fail safely.

## API summary

- `GET /api/health`
- `GET /api/documents`
- `POST /api/documents/upload`
- `POST /api/query`
- `POST /api/evaluations/run`

Fast evaluation mode:

- The UI defaults to a faster evaluation mode that skips the extra LLM support-check pass.
- Fast mode still measures keyword coverage, citation overlap, and latency.
- Full mode is slower, but it also estimates unsupported-claim rate.

## Testing

```powershell
pytest
```

## Known limits

- Citation faithfulness is improved by prompting and an LLM-as-judge pass, but it is still heuristic.
- Retrieval quality depends on the embedding model and the PDF text extraction quality.
- The app is local-first, so anyone cloning the repo still needs Ollama and the required models installed on their own machine.
- First answers on Windows can be slow when a local model is loading into memory. If queries time out, increase `OLLAMA_GENERATION_TIMEOUT_SECONDS` or switch to a smaller model.
