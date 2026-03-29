const healthElements = {
  ollamaStatus: document.getElementById("ollama-status"),
  generationModel: document.getElementById("generation-model"),
  embeddingModel: document.getElementById("embedding-model"),
  indexedDocs: document.getElementById("indexed-docs"),
};

const documentsList = document.getElementById("documents-list");
const uploadForm = document.getElementById("upload-form");
const uploadMessage = document.getElementById("upload-message");
const pdfInput = document.getElementById("pdf-input");
const pdfLabelText = document.getElementById("pdf-label-text");
const refreshDocsButton = document.getElementById("refresh-docs");

const queryForm = document.getElementById("query-form");
const queryMessage = document.getElementById("query-message");
const questionInput = document.getElementById("question-input");
const topkInput = document.getElementById("topk-input");
const answerShell = document.getElementById("answer-shell");
const answerText = document.getElementById("answer-text");
const citationsList = document.getElementById("citations-list");
const retrievedList = document.getElementById("retrieved-list");
const evidenceBadge = document.getElementById("evidence-badge");
const latencyText = document.getElementById("latency-text");

const evaluationForm = document.getElementById("evaluation-form");
const evaluationInput = document.getElementById("evaluation-input");
const evaluationLabelText = document.getElementById("evaluation-label-text");
const evaluationMessage = document.getElementById("evaluation-message");
const evaluationSummary = document.getElementById("evaluation-summary");
const evaluationRunButton = document.getElementById("evaluation-run-button");
const evaluationFastMode = document.getElementById("evaluation-fast-mode");

async function apiFetch(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "Request failed.");
  }
  return payload;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderAnswer(answer) {
  const escaped = escapeHtml(answer);
  const linked = escaped.replace(/\[(S\d+)\]/g, '<a href="#source-$1">$1</a>');
  return linked.replace(/\n/g, "<br />");
}

function renderHealth(data) {
  healthElements.ollamaStatus.textContent = data.ollama_reachable ? "Reachable" : "Offline";
  healthElements.ollamaStatus.style.color = data.ollama_reachable ? "var(--success)" : "var(--danger)";
  healthElements.generationModel.textContent = data.generation_model;
  healthElements.embeddingModel.textContent = data.embedding_model;
  healthElements.indexedDocs.textContent = String(data.indexed_documents);
}

function renderDocuments(documents) {
  if (!documents.length) {
    documentsList.className = "document-list empty-state";
    documentsList.textContent = "No documents indexed yet.";
    return;
  }

  documentsList.className = "document-list";
  documentsList.innerHTML = documents
    .map(
      (doc) => `
        <article class="document-card">
          <div class="document-card-header">
            <strong>${escapeHtml(doc.name)}</strong>
            <button class="button button-danger button-small delete-document-button" type="button" data-doc-id="${doc.id}" data-doc-name="${escapeHtml(doc.name)}">Delete</button>
          </div>
          <div class="meta-line">${doc.pages} pages | ${doc.chunks} chunks</div>
          <div class="meta-line">Indexed ${new Date(doc.uploaded_at).toLocaleString()}</div>
        </article>
      `
    )
    .join("");

  document.querySelectorAll(".delete-document-button").forEach((button) => {
    button.addEventListener("click", async () => {
      const docId = button.dataset.docId;
      const docName = button.dataset.docName;
      if (!window.confirm(`Delete ${docName} from the index?`)) {
        return;
      }

      uploadMessage.textContent = `Deleting ${docName}...`;
      try {
        await apiFetch(`/api/documents/${docId}`, { method: "DELETE" });
        uploadMessage.textContent = `${docName} was removed from the index.`;
        await Promise.all([loadHealth(), loadDocuments()]);
      } catch (error) {
        uploadMessage.textContent = error.message;
      }
    });
  });
}

function renderQueryResult(result) {
  answerShell.classList.remove("hidden");
  answerText.innerHTML = renderAnswer(result.answer);
  latencyText.textContent = `${result.latency_ms.toFixed(0)} ms`;
  evidenceBadge.textContent = result.enough_evidence ? "Grounded answer" : "Insufficient evidence";
  evidenceBadge.className = `badge ${result.enough_evidence ? "badge-success" : "badge-danger"}`;

  if (!result.citations.length) {
    citationsList.className = "citation-list empty-state";
    citationsList.textContent = "No citations returned.";
  } else {
    citationsList.className = "citation-list";
    citationsList.innerHTML = result.citations
      .map(
        (citation) => `
          <article class="citation-card" id="source-${citation.source_id}">
            <strong>${citation.source_id} · ${escapeHtml(citation.doc_name)} p.${citation.page}</strong>
            <div class="meta-line">${escapeHtml(citation.section_title)} | score ${citation.similarity_score.toFixed(2)}</div>
            <p>${escapeHtml(citation.quote)}</p>
          </article>
        `
      )
      .join("");
  }

  if (!result.retrieved_chunks.length) {
    retrievedList.className = "retrieved-list empty-state";
    retrievedList.textContent = "No retrieval results.";
  } else {
    retrievedList.className = "retrieved-list";
    retrievedList.innerHTML = result.retrieved_chunks
      .map(
        (chunk) => `
          <article class="retrieved-card">
            <strong>${chunk.source_id} · ${escapeHtml(chunk.doc_name)} p.${chunk.page}</strong>
            <div class="meta-line">${escapeHtml(chunk.section_title)} | similarity ${chunk.similarity_score.toFixed(2)}</div>
            <p>${escapeHtml(chunk.text.slice(0, 420))}${chunk.text.length > 420 ? "..." : ""}</p>
          </article>
        `
      )
      .join("");
  }
}

function renderEvaluation(summary) {
  const unsupportedRateText =
    summary.unsupported_claim_rate === null ? "Skipped" : summary.unsupported_claim_rate.toFixed(2);
  const modeText = summary.support_check_enabled ? "Full evaluation" : "Fast evaluation";

  evaluationSummary.classList.remove("hidden");
  evaluationSummary.innerHTML = `
    <div class="evaluation-metrics">
      <article class="metric-card"><strong>Questions</strong><span>${summary.total_questions}</span></article>
      <article class="metric-card"><strong>Mode</strong><span>${modeText}</span></article>
      <article class="metric-card"><strong>Keyword</strong><span>${summary.average_keyword_score.toFixed(2)}</span></article>
      <article class="metric-card"><strong>Citation Recall</strong><span>${summary.average_citation_recall.toFixed(2)}</span></article>
      <article class="metric-card"><strong>Unsupported Rate</strong><span>${unsupportedRateText}</span></article>
      <article class="metric-card"><strong>Latency</strong><span>${summary.average_latency_ms.toFixed(0)} ms</span></article>
    </div>
    <article class="document-card">
      <strong>Saved report</strong>
      <div class="meta-line">${escapeHtml(summary.report_path)}</div>
    </article>
    <div class="case-list">
      ${summary.cases
        .map(
          (item) => `
            <article class="case-card">
              <strong>${escapeHtml(item.question)}</strong>
              <p>${escapeHtml(item.answer)}</p>
              <p>Keyword ${item.keyword_score.toFixed(2)} | Recall ${item.citation_recall.toFixed(2)} | Precision ${item.citation_precision.toFixed(2)} | Supported ${item.supported === null ? "Skipped" : item.supported}</p>
            </article>
          `
        )
        .join("")}
    </div>
  `;
}

async function loadHealth() {
  const health = await apiFetch("/api/health");
  renderHealth(health);
}

async function loadDocuments() {
  const documents = await apiFetch("/api/documents");
  renderDocuments(documents);
  return documents;
}

function updatePdfLabel() {
  const count = pdfInput.files.length;
  if (!pdfLabelText) {
    return;
  }
  if (!count) {
    pdfLabelText.textContent = "Select one or more PDF files";
    return;
  }
  pdfLabelText.textContent = count === 1 ? pdfInput.files[0].name : `${count} PDF files selected`;
}

function updateEvaluationLabel() {
  if (!evaluationLabelText) {
    return;
  }
  evaluationLabelText.textContent = evaluationInput.files.length
    ? evaluationInput.files[0].name
    : "Select evaluation JSON";
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!pdfInput.files.length) {
    uploadMessage.textContent = "Choose at least one PDF first.";
    return;
  }

  uploadMessage.textContent = "Indexing documents...";
  const formData = new FormData();
  Array.from(pdfInput.files).forEach((file) => formData.append("files", file));

  try {
    await apiFetch("/api/documents/upload", {
      method: "POST",
      body: formData,
    });
    uploadMessage.textContent = "Documents indexed successfully.";
    pdfInput.value = "";
    updatePdfLabel();
    await Promise.all([loadHealth(), loadDocuments()]);
  } catch (error) {
    uploadMessage.textContent = error.message;
  }
});

pdfInput.addEventListener("change", () => {
  updatePdfLabel();
});

refreshDocsButton.addEventListener("click", async () => {
  await Promise.all([loadHealth(), loadDocuments()]);
});

queryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!questionInput.value.trim()) {
    queryMessage.textContent = "Enter a question first.";
    return;
  }

  queryMessage.textContent = "Running retrieval and generation...";
  try {
    const result = await apiFetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: questionInput.value.trim(),
        top_k: Number(topkInput.value),
      }),
    });
    renderQueryResult(result);
    queryMessage.textContent = result.enough_evidence
      ? "Answer generated from the indexed documents."
      : "The system could not find enough support in the indexed documents.";
  } catch (error) {
    queryMessage.textContent = error.message;
  }
});

evaluationForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    if (!evaluationInput.files.length) {
      evaluationMessage.textContent = "Choose an evaluation JSON file first.";
      return;
    }

    evaluationMessage.textContent = "Checking system state...";
    const health = await apiFetch("/api/health");
    if (health.indexed_documents < 1) {
      evaluationMessage.textContent = "Upload and index at least one PDF before running an evaluation.";
      return;
    }

    evaluationSummary.classList.add("hidden");
    evaluationSummary.innerHTML = "";
    if (evaluationRunButton) {
      evaluationRunButton.disabled = true;
      evaluationRunButton.textContent = "Running...";
    }
    evaluationMessage.textContent = `Running evaluation for ${evaluationInput.files[0].name}...`;
    const formData = new FormData();
    formData.append("dataset", evaluationInput.files[0]);
    formData.append("fast_mode", evaluationFastMode && evaluationFastMode.checked ? "true" : "false");

    const summary = await apiFetch("/api/evaluations/run", {
      method: "POST",
      body: formData,
    });
    renderEvaluation(summary);
    evaluationMessage.textContent = "Evaluation completed.";
  } catch (error) {
    evaluationMessage.textContent = error?.message || "Evaluation failed.";
  } finally {
    if (evaluationRunButton) {
      evaluationRunButton.disabled = false;
      evaluationRunButton.textContent = "Run Evaluation";
    }
  }
});

evaluationInput.addEventListener("change", () => {
  updateEvaluationLabel();
  evaluationMessage.textContent = evaluationInput.files.length
    ? `Selected ${evaluationInput.files[0].name}.`
    : "";
});

Promise.all([loadHealth(), loadDocuments()]).catch((error) => {
  queryMessage.textContent = error.message;
});
