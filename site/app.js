// Page logic for the static demo client: state, requests, rendering, logging.
// Design: docs/llds/07_ui.md. Specs: docs/specs/07_ui.md (UI-UI-*).
//
// Two independent paths produce a result: selecting a bundled sample renders
// its precomputed result (no API call), and Run Prediction posts the uploaded
// VCF to the live API. Every sample selection or Run is an "action"; the
// latest action wins — starting one aborts the previous action's fetch and any
// late response from a superseded action is dropped.

import { checkFileMeta, checkFileText } from "./vcf.js";

const MESSAGES = {
  generic: "The prediction request failed — please try again.",
  unavailable: "The prediction service is starting up or unavailable — please try again in a moment.",
  notConfigured: "The prediction service is not configured.",
  busy: "The demo is busy right now — please try again in a few seconds.",
};

// @spec UI-UI-015
const ERROR_CODE_MESSAGES = {
  MODEL_UNAVAILABLE: MESSAGES.unavailable,
  INVALID_VCF:
    "The uploaded VCF could not be processed (multi-sample file, malformed format, or no usable variants).",
  INVALID_INPUT: "The request was rejected as malformed — please reload and try again.",
  INFERENCE_FAILED: "The prediction service encountered an error — please try again or contact support.",
  INTERNAL_ERROR: "The prediction service encountered an error — please try again or contact support.",
};

// Outlasts the Lambda's 29 s timeout and API Gateway's 30 s cutoff, so a
// server-side answer normally arrives first; this covers a stalled network.
const CLIENT_TIMEOUT_MS = 35_000;

const ENDPOINT = (typeof window.PHENO_API_ENDPOINT === "string" ? window.PHENO_API_ENDPOINT : "")
  .trim()
  .replace(/\/+$/, "");

const $ = (id) => document.getElementById(id);
const ui = {
  sampleSection: $("sample-section"),
  sampleOptions: $("sample-options"),
  file: $("vcf-file"),
  uploadError: $("upload-error"),
  run: $("run-button"),
  loading: $("loading"),
  message: $("result-message"),
  result: $("result"),
  source: $("result-source"),
  phenotype: $("result-phenotype"),
  confidence: $("result-confidence"),
  confidenceBar: $("result-confidence-bar"),
  chart: $("marker-chart"),
  tableBody: document.querySelector("#marker-table tbody"),
};

// ── Logging ───────────────────────────────────────────────────────────────────

/**
 * The single exit point for client log events: one JSON object per event.
 * @spec UI-UI-023
 * @param {"INFO"|"WARN"|"ERROR"} level
 * @param {string} event Stable event name.
 * @param {object} [fields] Event-specific fields.
 */
export function log(level, event, fields = {}) {
  const line = JSON.stringify({ level, event, ...fields });
  if (level === "ERROR") console.error(line);
  else if (level === "WARN") console.warn(line);
  else console.info(line);
}

// ── Actions: latest action wins ───────────────────────────────────────────────

let current = null;

// @spec UI-UI-034, UI-UI-052
function beginAction() {
  if (current) current.controller.abort();
  current = { controller: new AbortController(), live: false };
  syncControls();
  return current;
}

const isCurrent = (action) => action === current;

function finish(action) {
  if (!isCurrent(action)) return;
  action.live = false;
  syncControls();
}

// @spec UI-UI-009, UI-UI-051
function syncControls() {
  const busy = Boolean(current && current.live);
  ui.loading.hidden = !busy;
  ui.run.disabled = ui.file.files.length === 0 || busy;
}

// ── Results area ──────────────────────────────────────────────────────────────

// @spec UI-UI-059
function showError(message) {
  ui.message.textContent = message;
  ui.result.hidden = true;
}

// Decimal rounding: 0.825 must read as 83%, but 0.825 * 100 is 82.49999… in
// binary floating point, so trim the representation error before rounding.
function percent(score) {
  return Math.round(Number.parseFloat((score * 100).toPrecision(12)));
}

// @spec UI-UI-011, UI-UI-012, UI-UI-032
function renderResult(result, sourceText) {
  ui.message.textContent = "";
  ui.source.textContent = sourceText;
  ui.phenotype.textContent = result.predicted_phenotype;
  const pct = percent(result.confidence_score);
  ui.confidence.textContent = `${pct}%`;
  ui.confidenceBar.setAttribute("value", String(pct));

  const markers = [...(result.top_markers || [])];
  ui.tableBody.replaceChildren(
    ...[...markers].sort((a, b) => a.rank - b.rank).map((m) => {
      const tr = document.createElement("tr");
      for (const value of [m.rank, m.variant_id, m.chrom, m.pos, m.shap_contribution.toFixed(4)]) {
        const td = document.createElement("td");
        td.textContent = String(value);
        tr.append(td);
      }
      return tr;
    }),
  );
  renderChart(markers);
  ui.result.hidden = false;
}

const SVG_NS = "http://www.w3.org/2000/svg";
const CHART = { width: 640, label: 150, row: 22, gap: 6, top: 4 };

function svg(tag, attrs) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

/**
 * Diverging horizontal bars from a shared zero axis, largest |SHAP| on top.
 * @spec UI-UI-014, UI-UI-037
 */
function renderChart(markers) {
  const sorted = [...markers].sort(
    (a, b) => Math.abs(b.shap_contribution) - Math.abs(a.shap_contribution),
  );
  const half = (CHART.width - CHART.label) / 2;
  const zero = CHART.label + half;
  const maxAbs = Math.max(0, ...sorted.map((m) => Math.abs(m.shap_contribution))) || 1;
  const height = CHART.top * 2 + sorted.length * CHART.row;

  ui.chart.setAttribute("viewBox", `0 0 ${CHART.width} ${height}`);
  ui.chart.setAttribute("data-half-width", String(half));
  ui.chart.replaceChildren();

  sorted.forEach((m, i) => {
    const v = m.shap_contribution;
    const w = (Math.abs(v) / maxAbs) * half;
    const y = CHART.top + i * CHART.row;
    const g = svg("g", { class: `bar ${v < 0 ? "neg" : "pos"}`, "data-variant": m.variant_id });
    const label = svg("text", {
      x: CHART.label - 8, y: y + CHART.row / 2, "text-anchor": "end", "dominant-baseline": "middle",
    });
    label.textContent = m.variant_id;
    const title = svg("title", {});
    title.textContent = `${m.variant_id}: ${v.toFixed(4)}`;
    const rect = svg("rect", {
      x: v < 0 ? zero - w : zero, y: y + CHART.gap / 2, width: w, height: CHART.row - CHART.gap,
    });
    rect.append(title);
    g.append(label, rect);
    ui.chart.append(g);
  });
  ui.chart.append(svg("line", { id: "zero-axis", x1: zero, x2: zero, y1: 0, y2: height }));
}

// ── Live prediction ───────────────────────────────────────────────────────────

// @spec UI-UI-015, UI-UI-025, UI-UI-035, UI-UI-036, UI-UI-010
function handleHttpError(status, bodyText) {
  let body = null;
  try {
    body = JSON.parse(bodyText);
  } catch {
    // not JSON: an API Gateway or proxy page rather than the error contract
  }
  const contract = body !== null && typeof body === "object"
    && typeof body.error === "string" && "detail" in body;
  log("ERROR", "predict_http_error", {
    status,
    error_code: contract ? body.error : null,
    detail: contract ? body.detail : null,
  });

  let message = MESSAGES.generic;
  if (status === 429) message = MESSAGES.busy;
  else if (contract) message = ERROR_CODE_MESSAGES[body.error] ?? MESSAGES.generic;
  else if (status === 503 || status === 504) message = MESSAGES.unavailable;
  showError(message);
}

// @spec UI-UI-008, UI-UI-010, UI-UI-024, UI-UI-026, UI-UI-046, UI-UI-053
async function livePredict(action, { text, filename, sizeBytes, input }) {
  action.live = true;
  syncControls();
  log("INFO", "predict_request_sent", { input, filename, size_bytes: sizeBytes });

  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    action.controller.abort();
  }, CLIENT_TIMEOUT_MS);
  try {
    const response = await fetch(`${ENDPOINT}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ vcf: text }),
      signal: action.controller.signal,
    });
    const bodyText = await response.text();
    if (!isCurrent(action)) return;
    if (!response.ok) {
      handleHttpError(response.status, bodyText);
      return;
    }
    const result = JSON.parse(bodyText);
    renderResult(result, "Live prediction");
    log("INFO", "live_result_rendered", {
      phenotype: result.predicted_phenotype,
      confidence: result.confidence_score,
      markers: (result.top_markers || []).length,
    });
  } catch (err) {
    if (!isCurrent(action)) return; // superseded: its abort is expected, not an error
    log("ERROR", "predict_failed", {
      error: timedOut ? `no response within ${CLIENT_TIMEOUT_MS / 1000} s` : String(err?.message ?? err),
      stack: err?.stack ?? null,
    });
    showError(MESSAGES.generic);
  } finally {
    clearTimeout(timer);
    finish(action);
  }
}

// ── Run Prediction (upload path) ──────────────────────────────────────────────

// @spec UI-UI-003, UI-UI-004, UI-UI-005, UI-UI-030, UI-UI-038, UI-UI-044
async function onRun() {
  const file = ui.file.files[0];
  if (!file) return;
  ui.uploadError.textContent = "";

  // Metadata first, so an oversized file is rejected without being read.
  let errors = checkFileMeta(file.name, file.size);
  let text = null;
  if (errors.length === 0) {
    text = await file.text();
    errors = checkFileText(text);
  }
  if (errors.length > 0) {
    ui.uploadError.textContent = errors.join(" ");
    log("WARN", "upload_invalid", { filename: file.name, errors });
    return;
  }

  const action = beginAction();
  if (!ENDPOINT) {
    showError(MESSAGES.notConfigured);
    finish(action);
    return;
  }
  await livePredict(action, { text, filename: file.name, sizeBytes: file.size, input: "upload" });
}

// ── Samples ───────────────────────────────────────────────────────────────────

// @spec UI-UI-031, UI-UI-033, UI-UI-054, UI-UI-022, UI-UI-045, UI-UI-047, UI-UI-039
async function onSampleSelected(entry) {
  if (!entry) return; // "None" is not an action: it supersedes nothing.
  const action = beginAction();
  const { signal } = action.controller;

  let result;
  try {
    const response = await fetch(`samples/${entry.result}`, { signal });
    if (!response.ok) throw new Error(`HTTP ${response.status} for ${entry.result}`);
    result = JSON.parse(await response.text());
  } catch (err) {
    if (!isCurrent(action)) return;
    log("WARN", "precomputed_result_unavailable", {
      sample: entry.label, error: String(err?.message ?? err),
    });
    await fallBackToLive(action, entry);
    return;
  }
  if (!isCurrent(action)) return;
  renderResult(result, `Precomputed result for ${entry.label}`);
  log("INFO", "precomputed_result_rendered", {
    sample: entry.label, phenotype: result.predicted_phenotype, confidence: result.confidence_score,
  });
  finish(action);
}

async function fallBackToLive(action, entry) {
  if (!ENDPOINT) {
    showError(MESSAGES.notConfigured);
    finish(action);
    return;
  }
  let text;
  try {
    const response = await fetch(`samples/${entry.vcf}`, { signal: action.controller.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status} for ${entry.vcf}`);
    text = await response.text();
  } catch (err) {
    if (!isCurrent(action)) return;
    log("ERROR", "predict_failed", { error: String(err?.message ?? err), stack: err?.stack ?? null });
    showError(MESSAGES.generic);
    finish(action);
    return;
  }
  if (!isCurrent(action)) return;
  // Bundled samples are pre-validated fixtures: no browser-side checks.
  await livePredict(action, {
    text, filename: entry.vcf, sizeBytes: new Blob([text]).size, input: "sample",
  });
}

function radio(value, labelText, checked) {
  const label = document.createElement("label");
  const input = document.createElement("input");
  input.type = "radio";
  input.name = "sample";
  input.value = value;
  input.checked = checked;
  label.append(input, document.createTextNode(labelText));
  return { label, input };
}

// @spec UI-UI-016, UI-UI-017, UI-UI-018, UI-UI-019
async function loadSamples() {
  let entries;
  try {
    const response = await fetch("samples/manifest.json");
    if (!response.ok) return;
    entries = JSON.parse(await response.text());
  } catch {
    return;
  }
  if (!Array.isArray(entries) || entries.length === 0) return;

  const none = radio("none", "None — use uploaded file", true);
  none.input.addEventListener("change", () => onSampleSelected(null));
  ui.sampleOptions.append(none.label);
  entries.forEach((entry, i) => {
    const option = radio(String(i), entry.label, false);
    option.input.addEventListener("change", () => onSampleSelected(entry));
    ui.sampleOptions.append(option.label);
  });
  ui.sampleSection.hidden = false;
}

// ── Start-up ──────────────────────────────────────────────────────────────────

// @spec UI-UI-027, UI-UI-028, UI-UI-029, UI-UI-048, UI-UI-049
function warmUp() {
  if (!ENDPOINT) {
    log("ERROR", "api_endpoint_not_configured", {
      detail: "window.PHENO_API_ENDPOINT is empty or config.js did not load; only samples will work.",
    });
    return;
  }
  // Fire and forget: starts the Lambda's cold start while the visitor is still
  // choosing a file. The outcome is deliberately ignored and never logged.
  fetch(`${ENDPOINT}/health`).catch(() => {});
}

ui.file.addEventListener("change", () => {
  ui.uploadError.textContent = "";
  syncControls();
});
ui.run.addEventListener("click", onRun);

warmUp();
syncControls();
loadSamples();
