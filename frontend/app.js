const backendUrlInput = document.querySelector("#backend-url");
const filesInput = document.querySelector("#files");
const runBtn = document.querySelector("#run-btn");
const statusEl = document.querySelector("#status");
const previewGridEl = document.querySelector("#preview-grid");
const decisionPanelEl = document.querySelector("#decision-panel");
const primaryPreviewEl = document.querySelector("#primary-preview");
const fileCountEl = document.querySelector("#file-count");
const topConfidenceEl = document.querySelector("#top-confidence");
const filesProcessedEl = document.querySelector("#files-processed");
const resultsEl = document.querySelector("#results");
let selectedPreviewName = null;
let selectedFiles = [];

function defaultBackendUrl() {
  if (window.location.protocol === "file:") {
    return "http://127.0.0.1:8000/predict";
  }
  const isLocalhost =
    window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost";
  if (isLocalhost && window.location.port !== "8000") {
    return "http://127.0.0.1:8000/predict";
  }
  return `${window.location.origin}/predict`;
}

backendUrlInput.value = defaultBackendUrl();

function setStatus(message, kind = "") {
  statusEl.textContent = message;
  statusEl.className = `status ${kind}`.trim();
}

function renderPrimaryPreview(file) {
  if (!file) {
    primaryPreviewEl.className = "primary-preview is-empty";
    primaryPreviewEl.innerHTML = `
      <div class="empty-illustration">
        <span>No scan selected</span>
      </div>
    `;
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  primaryPreviewEl.className = "primary-preview";
  primaryPreviewEl.innerHTML = `<img src="${objectUrl}" alt="${file.name}" />`;
}

function renderPreviews(files) {
  selectedFiles = files;
  fileCountEl.textContent = `${files.length} selected`;
  if (!files.length) {
    previewGridEl.innerHTML = '<div class="results-empty">No files selected yet.</div>';
    selectedPreviewName = null;
    renderPrimaryPreview(null);
    return;
  }

  if (!selectedPreviewName || !files.some((file) => file.name === selectedPreviewName)) {
    selectedPreviewName = files[0].name;
  }

  renderPrimaryPreview(files.find((file) => file.name === selectedPreviewName) || files[0]);

  previewGridEl.innerHTML = files
    .map((file) => {
      const objectUrl = URL.createObjectURL(file);
      return `
        <button class="preview-tile ${file.name === selectedPreviewName ? "is-selected" : ""}" type="button" data-filename="${file.name}">
          <div class="preview-frame">
            <img src="${objectUrl}" alt="${file.name}" />
          </div>
          <div class="preview-meta">
            <div class="preview-name">${file.name}</div>
            <div class="preview-sub">Ready for review</div>
          </div>
        </button>
      `;
    })
    .join("");

  previewGridEl.querySelectorAll(".preview-tile").forEach((tile) => {
    tile.addEventListener("click", () => {
      selectedPreviewName = tile.dataset.filename;
      renderPreviews(selectedFiles);
    });
  });
}

function renderDecision(results) {
  if (!results.length) {
    topConfidenceEl.textContent = "--";
    filesProcessedEl.textContent = "0";
    decisionPanelEl.className = "decision-panel decision-panel-empty";
    decisionPanelEl.innerHTML = `
      <div class="decision-main">
        <span class="decision-label">Awaiting inference</span>
        <strong class="decision-value">No result yet</strong>
      </div>
      <p class="decision-copy">Run the model to see the highest-risk finding across the uploaded images.</p>
    `;
    return;
  }

  const topResult = [...results].sort((a, b) => b.probability - a.probability)[0];
  const isPositive = Number(topResult.prediction) === 1;
  topConfidenceEl.textContent = `${(Number(topResult.probability) * 100).toFixed(1)}%`;
  filesProcessedEl.textContent = String(results.length);
  decisionPanelEl.className = `decision-panel ${
    isPositive ? "decision-panel-positive" : "decision-panel-negative"
  }`;
  decisionPanelEl.innerHTML = `
    <div class="decision-main">
      <span class="decision-label">Highest-risk finding</span>
      <strong class="decision-value">${isPositive ? "Cardiomegaly likely" : "No cardiomegaly detected"}</strong>
    </div>
    <p class="decision-copy">
      ${topResult.filename} returned the highest probability at ${Number(topResult.probability).toFixed(4)}.
    </p>
  `;
}

function renderResults(results) {
  if (!results.length) {
    resultsEl.innerHTML = '<div class="results-empty">No predictions returned.</div>';
    renderDecision([]);
    return;
  }

  const cards = results
    .map(
      (row) => `
        <article class="result-card">
          <div class="result-top">
            <div class="result-name">${row.filename}</div>
            <span class="result-tag ${row.prediction === 1 ? "positive" : "negative"}">
              ${row.prediction === 1 ? "Cardiomegaly likely" : "No cardiomegaly detected"}
            </span>
          </div>
          <div class="result-prob">
            <span>Confidence ${Number(row.probability).toFixed(4)}</span>
            <div class="prob-bar">
              <div class="prob-bar-fill" style="width: ${Math.max(3, Number(row.probability) * 100)}%"></div>
            </div>
          </div>
        </article>
      `
    )
    .join("");

  resultsEl.innerHTML = `<div class="results-list">${cards}</div>`;

  renderDecision(results);
}

filesInput.addEventListener("change", () => {
  renderPreviews([...filesInput.files]);
});

runBtn.addEventListener("click", async () => {
  const files = [...filesInput.files];
  if (!files.length) {
    setStatus("Choose at least one X-ray image first.", "error");
    return;
  }

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  setStatus("Running inference...", "");
  runBtn.disabled = true;

  try {
    const response = await fetch(backendUrlInput.value, {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Prediction request failed.");
    }

    renderResults(payload.results || []);
    setStatus("Inference completed.", "success");
  } catch (error) {
    renderResults([]);
    setStatus(error.message, "error");
  } finally {
    runBtn.disabled = false;
  }
});
