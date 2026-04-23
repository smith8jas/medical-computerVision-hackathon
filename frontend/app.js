const backendUrlInput = document.querySelector("#backend-url");
const filesInput = document.querySelector("#files");
const uploadZoneEl = document.querySelector("#upload-zone");
const runBtn = document.querySelector("#run-btn");
const healthBtn = document.querySelector("#health-btn");
const statusEl = document.querySelector("#status");
const previewGridEl = document.querySelector("#preview-grid");
const decisionPanelEl = document.querySelector("#decision-panel");
const primaryPreviewEl = document.querySelector("#primary-preview");
const fileCountEl = document.querySelector("#file-count");
const topConfidenceEl = document.querySelector("#top-confidence");
const filesProcessedEl = document.querySelector("#files-processed");
const apiStatusEl = document.querySelector("#api-status");
const resultsEl = document.querySelector("#results");
const reportBtn = document.querySelector("#report-btn");
const doctorNameInput = document.querySelector("#doctor-name");
const patientNameInput = document.querySelector("#patient-name");
const patientIdInput = document.querySelector("#patient-id");
const clinicalIndicationInput = document.querySelector("#clinical-indication");
const doctorNotesInput = document.querySelector("#doctor-notes");
let selectedPreviewName = null;
let selectedFiles = [];
let lastResults = [];

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

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => {
    const entities = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;",
    };
    return entities[char];
  });
}

function healthUrlFromPredictUrl(predictUrl) {
  const url = new URL(predictUrl, window.location.href);
  if (/\/predict\/?$/.test(url.pathname)) {
    url.pathname = url.pathname.replace(/\/predict\/?$/, "/health");
  } else {
    url.pathname = "/health";
  }
  return url.toString();
}

function setStatus(message, kind = "") {
  statusEl.textContent = message;
  statusEl.className = `status ${kind}`.trim();
}

function setApiStatus(message, kind = "") {
  apiStatusEl.textContent = message;
  apiStatusEl.className = kind ? `api-status ${kind}` : "api-status";
}

function formatServiceStatus(value) {
  return String(value || "unknown")
    .replaceAll("_", " ")
    .replace(/^./, (char) => char.toUpperCase());
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
  primaryPreviewEl.innerHTML = `<img src="${objectUrl}" alt="${escapeHtml(file.name)}" />`;
}

function selectedFile() {
  if (!selectedFiles.length) {
    return null;
  }
  return selectedFiles.find((file) => file.name === selectedPreviewName) || selectedFiles[0];
}

function selectedResult() {
  if (!lastResults.length) {
    return null;
  }
  return lastResults.find((result) => result.filename === selectedPreviewName) || lastResults[0];
}

function updateReportAvailability() {
  reportBtn.disabled = !(selectedFile() && selectedResult());
}

function renderPreviews(files) {
  selectedFiles = files;
  fileCountEl.textContent = `${files.length} selected`;
  if (!files.length) {
    previewGridEl.innerHTML = '<div class="results-empty">No files selected yet.</div>';
    selectedPreviewName = null;
    renderPrimaryPreview(null);
    updateReportAvailability();
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
        <button class="preview-tile ${file.name === selectedPreviewName ? "is-selected" : ""}" type="button" data-filename="${escapeHtml(file.name)}">
          <div class="preview-frame">
            <img src="${objectUrl}" alt="${escapeHtml(file.name)}" />
          </div>
          <div class="preview-meta">
            <div class="preview-name">${escapeHtml(file.name)}</div>
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
      renderResults(lastResults);
    });
  });
  updateReportAvailability();
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
      ${escapeHtml(topResult.filename)} returned the highest probability at ${Number(topResult.probability).toFixed(4)}.
      ${topResult.summary ? `<br><br>${escapeHtml(topResult.summary)}` : ""}
    </p>
  `;
}

function renderResults(results) {
  if (!results.length) {
    resultsEl.innerHTML = '<div class="results-empty">No predictions returned.</div>';
    lastResults = [];
    renderDecision([]);
    updateReportAvailability();
    return;
  }

  lastResults = results;

  const cards = results
    .map(
      (row) => `
        <button class="result-card ${row.filename === selectedPreviewName ? "is-selected" : ""}" type="button" data-filename="${escapeHtml(row.filename)}">
          <div class="result-top">
            <div class="result-name">${escapeHtml(row.filename)}</div>
            <span class="result-tag ${Number(row.prediction) === 1 ? "positive" : "negative"}">
              ${Number(row.prediction) === 1 ? "Cardiomegaly likely" : "No cardiomegaly detected"}
            </span>
          </div>
          <div class="result-prob">
            <span>Confidence ${Number(row.probability).toFixed(4)}</span>
            <div class="prob-bar">
              <div class="prob-bar-fill" style="width: ${Math.max(3, Number(row.probability) * 100)}%"></div>
            </div>
          </div>
          ${
            row.summary
              ? `<p class="result-summary">${escapeHtml(row.summary)}</p>`
              : `<p class="result-summary muted">LLM summary unavailable: ${escapeHtml(row.summary_source || "not configured")}</p>`
          }
        </button>
      `
    )
    .join("");

  resultsEl.innerHTML = `<div class="results-list">${cards}</div>`;
  resultsEl.querySelectorAll(".result-card").forEach((card) => {
    card.addEventListener("click", () => {
      selectedPreviewName = card.dataset.filename;
      renderPreviews(selectedFiles);
      renderResults(results);
    });
  });

  renderDecision(results);
  updateReportAvailability();
}

function reportText(value, fallback = "Not provided") {
  const clean = String(value || "").trim();
  return clean ? escapeHtml(clean) : fallback;
}

function imageUrlForReport(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Could not read selected image for report."));
    reader.readAsDataURL(file);
  });
}

function buildReportHtml({ imageUrl, file, result }) {
  const now = new Date();
  const probability = Number(result.probability);
  const isPositive = Number(result.prediction) === 1;
  const diagnostic = isPositive ? "Cardiomegaly likely" : "No cardiomegaly detected";
  const findings =
    result.summary ||
    `The AI classifier returned a ${probability.toFixed(4)} probability for cardiomegaly.`;
  const impression = isPositive
    ? "AI-assisted result is positive for likely cardiomegaly. Clinician review is required."
    : "AI-assisted result does not detect cardiomegaly above the model threshold. Clinician review is required.";

  return `<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Cardiomegaly AI Report - ${escapeHtml(file.name)}</title>
  <style>
    @page { margin: 18mm; }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: #162033;
      font-family: Georgia, "Times New Roman", serif;
      line-height: 1.45;
    }
    .report {
      max-width: 780px;
      margin: 0 auto;
    }
    .header {
      display: flex;
      justify-content: space-between;
      gap: 24px;
      border-bottom: 2px solid #162033;
      padding-bottom: 14px;
      margin-bottom: 18px;
    }
    .brand h1 {
      margin: 0;
      font-size: 22px;
      letter-spacing: 0.02em;
    }
    .brand p,
    .meta p,
    .section p {
      margin: 4px 0;
    }
    .meta {
      text-align: right;
      font-size: 12px;
      color: #4d5b72;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px 24px;
      margin-bottom: 18px;
    }
    .field-label {
      display: block;
      color: #66738a;
      font-size: 11px;
      font-family: Arial, sans-serif;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .field-value {
      display: block;
      font-size: 14px;
      font-weight: 700;
    }
    .section {
      border-top: 1px solid #d9e1ec;
      padding-top: 12px;
      margin-top: 12px;
    }
    .section h2 {
      margin: 0 0 8px;
      font-family: Arial, sans-serif;
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .image-wrap {
      margin-top: 10px;
      border: 1px solid #cfd8e6;
      background: #0f172a;
      padding: 8px;
      text-align: center;
    }
    .image-wrap img {
      max-width: 100%;
      max-height: 430px;
      object-fit: contain;
    }
    .diagnostic {
      display: inline-block;
      padding: 8px 10px;
      border-radius: 6px;
      background: ${isPositive ? "#fff0e4" : "#e6f7ee"};
      color: ${isPositive ? "#8a3d0d" : "#0f6b3e"};
      font-family: Arial, sans-serif;
      font-weight: 700;
    }
    .disclaimer {
      margin-top: 18px;
      padding: 10px;
      background: #f4f6fb;
      color: #5c6b82;
      font-size: 11px;
      font-family: Arial, sans-serif;
    }
    .sign {
      margin-top: 34px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }
    .line {
      border-top: 1px solid #162033;
      padding-top: 6px;
      font-size: 12px;
    }
    @media print {
      .no-print { display: none; }
    }
  </style>
</head>
<body>
  <main class="report">
    <div class="header">
      <div class="brand">
        <h1>Chest X-ray AI-Assisted Report</h1>
        <p>Cardiomegaly screening support</p>
      </div>
      <div class="meta">
        <p><strong>Report Date:</strong> ${escapeHtml(now.toLocaleString())}</p>
        <p><strong>Exam:</strong> Chest X-ray</p>
        <p><strong>Modality:</strong> Radiograph</p>
      </div>
    </div>

    <section class="grid">
      <div><span class="field-label">Patient Name</span><span class="field-value">${reportText(patientNameInput.value)}</span></div>
      <div><span class="field-label">Patient ID / MRN</span><span class="field-value">${reportText(patientIdInput.value, "Not provided")}</span></div>
      <div><span class="field-label">Doctor</span><span class="field-value">${reportText(doctorNameInput.value)}</span></div>
      <div><span class="field-label">Image File</span><span class="field-value">${escapeHtml(file.name)}</span></div>
    </section>

    <section class="section">
      <h2>Clinical Indication</h2>
      <p>${reportText(clinicalIndicationInput.value, "Evaluate cardiomegaly on chest X-ray.")}</p>
    </section>

    <section class="section">
      <h2>Comparison</h2>
      <p>No prior comparison study supplied in this application.</p>
    </section>

    <section class="section">
      <h2>Technique</h2>
      <p>Single uploaded chest radiograph processed by a cardiomegaly binary classification model.</p>
    </section>

    <section class="section">
      <h2>Image</h2>
      <div class="image-wrap"><img src="${imageUrl}" alt="${escapeHtml(file.name)}" /></div>
    </section>

    <section class="section">
      <h2>AI Findings</h2>
      <p>${escapeHtml(findings)}</p>
      <p><strong>Model probability:</strong> ${probability.toFixed(4)} (${(probability * 100).toFixed(1)}%)</p>
    </section>

    <section class="section">
      <h2>Impression / Diagnostic</h2>
      <p><span class="diagnostic">${diagnostic}</span></p>
      <p>${escapeHtml(impression)}</p>
    </section>

    <section class="section">
      <h2>Doctor Notes</h2>
      <p>${reportText(doctorNotesInput.value, "No additional notes provided.")}</p>
    </section>

    <div class="sign">
      <div class="line">Doctor Signature</div>
      <div class="line">Date / Time</div>
    </div>

    <p class="disclaimer">
      This document is generated by decision-support software from a machine-learning classifier output.
      It is not a standalone diagnosis and must be reviewed by a qualified clinician with the original image,
      patient history, physical examination, and any relevant prior studies.
    </p>
  </main>
  <script>
    window.addEventListener("load", () => {
      window.focus();
      window.print();
    });
  </script>
</body>
</html>`;
}

async function generateReport() {
  const file = selectedFile();
  const result = selectedResult();

  if (!file || !result) {
    setStatus("Run inference and select an image before generating a report.", "error");
    return;
  }

  const imageUrl = await imageUrlForReport(file);
  const reportWindow = window.open("", "_blank");
  if (!reportWindow) {
    setStatus("Popup blocked. Allow popups to generate the PDF report.", "error");
    return;
  }
  reportWindow.document.open();
  reportWindow.document.write(buildReportHtml({ imageUrl, file, result }));
  reportWindow.document.close();
  setStatus("Report opened. Choose Save as PDF in the print dialog.", "success");
}

function setInputFiles(files) {
  const transfer = new DataTransfer();
  files.forEach((file) => transfer.items.add(file));
  filesInput.files = transfer.files;
  renderPreviews(files);
}

filesInput.addEventListener("change", () => {
  renderPreviews([...filesInput.files]);
});

["dragenter", "dragover"].forEach((eventName) => {
  uploadZoneEl.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZoneEl.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  uploadZoneEl.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZoneEl.classList.remove("is-dragging");
  });
});

uploadZoneEl.addEventListener("drop", (event) => {
  const files = [...event.dataTransfer.files].filter((file) => file.type.startsWith("image/"));
  if (!files.length) {
    setStatus("Drop PNG, JPG, or JPEG X-ray images.", "error");
    return;
  }
  setInputFiles(files);
  setStatus(`${files.length} file${files.length === 1 ? "" : "s"} ready for inference.`, "");
});

healthBtn.addEventListener("click", async () => {
  setApiStatus("Checking...");
  healthBtn.disabled = true;

  try {
    const response = await fetch(healthUrlFromPredictUrl(backendUrlInput.value));
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(`API returned ${response.status}`);
    }
    setApiStatus(`API online / LLM ${formatServiceStatus(payload.llm_status)}`, "success");
    setStatus("Backend API is reachable.", "success");
  } catch (error) {
    setApiStatus("Offline", "error");
    setStatus(`Backend check failed: ${error.message}`, "error");
  } finally {
    healthBtn.disabled = false;
  }
});

reportBtn.addEventListener("click", () => {
  generateReport().catch((error) => {
    setStatus(error.message, "error");
  });
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
    setApiStatus(`Model ${formatServiceStatus(payload.model_status)} / LLM ${formatServiceStatus(payload.llm_status)}`, "success");
    setStatus("Inference completed.", "success");
  } catch (error) {
    renderResults([]);
    setApiStatus("Error", "error");
    setStatus(error.message, "error");
  } finally {
    runBtn.disabled = false;
  }
});
