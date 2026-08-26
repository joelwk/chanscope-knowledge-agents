"""Lightweight query UI for /query and /nl_query endpoints."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="Knowledge Query UI",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

QUERY_UI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Knowledge Query Console</title>
  <style>
    :root{
      --bg:#0e1116;
      --panel:#11141b;
      --card:#161a23;
      --text:#eef2f6;
      --muted:#b3bbc6;
      --accent:#7fb0ff;
      --ok:#34c759;
      --warn:#ff9f0a;
      --err:#ff453a;
      --border:#242a36;
      --code:#0d1118;
      --chip:#1d2431;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 14px;
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
      background: var(--panel);
    }
    header h1 {
      font-size: 20px;
      margin: 0;
    }
    header p {
      margin: 2px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 12px;
      border-radius: 999px;
      border: 1px solid var(--border);
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--muted);
    }
    .state-idle .status-dot { background: var(--muted); }
    .state-loading .status-dot { background: var(--warn); }
    .state-success .status-dot { background: var(--ok); }
    .state-error .status-dot { background: var(--err); }

    main {
      max-width: 1220px;
      margin: 20px auto;
      padding: 0 16px 24px;
    }
    .page-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(250px, 320px);
      gap: 18px;
      align-items: start;
    }
    .flow {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
    }
    .card h2, .card h3 {
      margin: 0 0 10px 0;
      font-size: 16px;
    }
    .muted { color: var(--muted); }
    .small { font-size: 12px; }

    .step-heading {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 12px;
    }
    .step-badge {
      width: 30px;
      height: 30px;
      border-radius: 999px;
      background: var(--accent);
      color: var(--bg);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 13px;
      flex: 0 0 30px;
    }
    .step-heading .title-wrap h2 {
      margin: 0;
      font-size: 17px;
    }
    .step-heading .title-wrap p {
      margin: 2px 0 0;
      color: var(--muted);
      font-size: 13px;
    }

    .mode-toggle {
      display: grid;
      grid-template-columns: 1fr 1fr;
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      margin-bottom: 12px;
    }
    .mode-btn {
      border: 0;
      background: #121722;
      color: var(--muted);
      padding: 10px 10px;
      cursor: pointer;
      text-align: center;
      font-weight: 600;
      transition: background 0.2s ease, color 0.2s ease;
    }
    .mode-btn + .mode-btn { border-left: 1px solid var(--border); }
    .mode-btn[aria-pressed="true"] {
      background: rgba(127, 176, 255, 0.18);
      color: var(--text);
    }
    .mode-hint {
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 12px;
    }

    textarea, input[type="text"], input[type="number"] {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #0f1420;
      color: var(--text);
      padding: 11px 12px;
      font: inherit;
    }
    textarea { min-height: 92px; resize: vertical; }
    input:focus-visible, textarea:focus-visible, button:focus-visible, summary:focus-visible {
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }

    .field-label {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
      font-size: 13px;
    }
    .field-error {
      display: none;
      margin-top: 6px;
      color: var(--err);
      font-size: 12px;
    }
    .field-error.visible { display: block; }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 12px;
      align-items: center;
    }
    button {
      border: 1px solid var(--border);
      border-radius: 9px;
      background: #1a212f;
      color: var(--text);
      padding: 9px 14px;
      cursor: pointer;
      font-weight: 600;
    }
    button:hover { filter: brightness(1.08); }
    button.primary {
      background: var(--accent);
      color: #0f131c;
      border-color: rgba(127, 176, 255, 0.4);
    }
    button.secondary {
      background: #161d2a;
      color: var(--muted);
    }
    button.inline-btn {
      padding: 4px 8px;
      font-size: 12px;
      background: #1a2435;
      color: var(--muted);
      margin-left: 10px;
    }
    button[disabled] {
      opacity: 0.6;
      cursor: not-allowed;
      filter: none;
    }

    details.advanced {
      margin-top: 12px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #121825;
      overflow: hidden;
    }
    details.advanced > summary {
      cursor: pointer;
      padding: 10px 12px;
      font-weight: 600;
      list-style: none;
    }
    details.advanced > summary::-webkit-details-marker { display: none; }
    .advanced-body {
      border-top: 1px solid var(--border);
      padding: 10px 12px;
      display: grid;
      gap: 8px;
    }
    .checkbox-row {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .checkbox-row input[type="checkbox"] {
      width: 16px;
      height: 16px;
      accent-color: var(--accent);
    }
    .advanced-section { display: none; }
    .advanced-section.active { display: block; }
    .inline-number {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 8px;
      align-items: center;
    }
    .inline-number span {
      color: var(--muted);
      font-size: 13px;
    }

    .alert {
      margin-top: 12px;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      display: none;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      background: #151c29;
      color: var(--muted);
    }
    .alert.visible { display: flex; }
    .alert.error {
      border-color: rgba(255, 69, 58, 0.55);
      background: rgba(255, 69, 58, 0.12);
      color: #ffc7c3;
    }
    .alert.info {
      border-color: rgba(127, 176, 255, 0.55);
      background: rgba(127, 176, 255, 0.12);
      color: #d6e6ff;
    }

    .result-shell {
      min-height: 220px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .empty-state {
      border: 1px dashed var(--border);
      border-radius: 10px;
      padding: 16px;
      color: var(--muted);
      background: #121826;
    }
    .summary-panel {
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      background: #111824;
    }
    .summary-panel h3 {
      margin: 0 0 8px;
      font-size: 14px;
    }
    .summary-text {
      margin: 0;
      white-space: pre-wrap;
      color: var(--text);
      font-size: 13px;
    }

    .chunk-list {
      display: grid;
      gap: 10px;
    }
    .chunk-card {
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 11px;
      background: #121a27;
    }
    .chunk-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 6px;
    }
    .chunk-title {
      font-size: 13px;
      color: var(--muted);
      margin: 0;
    }
    .badge-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 8px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 11px;
      color: var(--muted);
      background: var(--chip);
    }
    .chunk-text {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 13px;
      color: var(--text);
    }
    .expand-wrap details {
      margin-top: 8px;
      border-top: 1px solid var(--border);
      padding-top: 8px;
    }
    .expand-wrap summary {
      cursor: pointer;
      color: var(--accent);
      font-size: 12px;
    }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .kpi {
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 9px 10px;
      background: #121a27;
    }
    .kpi .label {
      margin: 0;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .3px;
    }
    .kpi .value {
      margin: 4px 0 0;
      font-size: 15px;
      font-weight: 700;
      color: var(--text);
    }

    .chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .filter-chip {
      display: inline-flex;
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 12px;
      border: 1px solid var(--border);
      color: #d8e5ff;
      background: rgba(127, 176, 255, 0.12);
    }

    .table-wrap {
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: auto;
      background: #111825;
      max-height: 400px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      border-bottom: 1px solid var(--border);
      text-align: left;
      padding: 8px;
      vertical-align: top;
    }
    th {
      background: #131c2c;
      color: var(--muted);
      position: sticky;
      top: 0;
      z-index: 1;
    }
    td {
      color: var(--text);
      max-width: 360px;
      white-space: pre-wrap;
      word-break: break-word;
    }

    details.collapsible {
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      background: #111825;
    }
    details.collapsible + details.collapsible {
      margin-top: 8px;
    }
    details.collapsible > summary {
      cursor: pointer;
      padding: 9px 12px;
      font-size: 13px;
      color: var(--muted);
      user-select: none;
      list-style: none;
    }
    details.collapsible > summary::-webkit-details-marker { display: none; }
    .json-panel {
      margin: 0;
      padding: 0 12px 12px;
      color: #d7deea;
      font-family: Consolas, "Courier New", monospace;
      font-size: 12px;
      white-space: pre-wrap;
      overflow: auto;
      background: var(--code);
      border-top: 1px solid var(--border);
    }
    .status-trail {
      list-style: none;
      margin: 0;
      padding: 0 12px 12px;
      border-top: 1px solid var(--border);
    }
    .status-trail li {
      padding: 8px 0;
      border-bottom: 1px dashed var(--border);
      font-size: 12px;
      color: var(--muted);
    }
    .status-trail li:last-child { border-bottom: 0; }
    .status-time { color: #8ea0b6; margin-right: 6px; }

    .restriction-card {
      border: 1px solid rgba(255, 159, 10, 0.5);
      border-radius: 10px;
      background: rgba(255, 159, 10, 0.08);
      padding: 12px;
      color: #ffd6a6;
      font-size: 13px;
    }
    .restriction-card h4 {
      margin: 0 0 8px;
      font-size: 14px;
      color: #ffe5bf;
    }

    .help-card {
      position: sticky;
      top: 20px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .help-card p {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }
    .help-list {
      margin: 0;
      padding-left: 16px;
      color: var(--muted);
      font-size: 13px;
    }
    .example-grid {
      display: grid;
      gap: 8px;
    }
    .example-btn {
      width: 100%;
      text-align: left;
      font-size: 12px;
      padding: 8px 10px;
      border-radius: 8px;
      background: #111a29;
      color: #d7e5ff;
      border: 1px solid var(--border);
      cursor: pointer;
    }
    .example-btn:hover { background: #142036; }

    .hidden { display: none !important; }

    @media (max-width: 1000px) {
      .page-grid { grid-template-columns: 1fr; }
      .help-card { position: static; }
    }
    @media (max-width: 720px) {
      header { padding: 14px; align-items: flex-start; flex-direction: column; }
      main { padding: 0 12px 18px; }
      .kpi-grid { grid-template-columns: 1fr; }
      .inline-number { grid-template-columns: 1fr; }
      .mode-btn { font-size: 12px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Knowledge Query Console</h1>
      <p>Toggle between semantic retrieval and NL-to-SQL queries with one prompt box.</p>
    </div>
    <div id="result-state-pill" class="status-pill state-idle">
      <span class="status-dot"></span>
      <span id="result-state-text">Idle</span>
    </div>
  </header>

  <main>
    <div class="page-grid">
      <section class="flow">
        <div class="card">
          <div class="step-heading">
            <span class="step-badge">1</span>
            <div class="title-wrap">
              <h2>Choose mode and submit query</h2>
              <p>Default mode is Semantic Query for broad compatibility.</p>
            </div>
          </div>

          <div class="mode-toggle" role="tablist" aria-label="Query mode">
            <button id="mode-query" class="mode-btn" role="tab" aria-pressed="true" aria-controls="advanced-query">Semantic (/query)</button>
            <button id="mode-nl" class="mode-btn" role="tab" aria-pressed="false" aria-controls="advanced-nl">NL->SQL (/nl_query)</button>
          </div>
          <p id="mode-hint" class="mode-hint">Uses <code>/api/v1/query</code> and returns summaries with relevant chunks.</p>

          <label class="field-label" for="query-input">
            <span>Query</span>
            <span class="small muted">Required</span>
          </label>
          <textarea id="query-input" placeholder="Ask your question here..."></textarea>
          <div id="query-error" class="field-error">Please enter a query before submitting.</div>

          <div class="actions">
            <button id="submit-btn" class="primary" type="button">Run Query</button>
            <button id="refresh-status-btn" class="secondary hidden" type="button">Refresh status</button>
            <span id="task-id-inline" class="small muted hidden"></span>
          </div>

          <details class="advanced">
            <summary>Advanced options</summary>
            <div class="advanced-body">
              <div id="advanced-query" class="advanced-section active">
                <label class="checkbox-row">
                  <input id="q-use-background" type="checkbox" />
                  <span>Use background processing</span>
                </label>
                <label class="checkbox-row">
                  <input id="q-force-refresh" type="checkbox" />
                  <span>Force refresh</span>
                </label>
                <label class="checkbox-row">
                  <input id="q-skip-embeddings" type="checkbox" />
                  <span>Skip embeddings</span>
                </label>
              </div>

              <div id="advanced-nl" class="advanced-section">
                <div class="inline-number">
                  <span>Result limit</span>
                  <input id="nl-limit" type="number" min="1" max="1000" value="50" />
                </div>
                <label class="checkbox-row">
                  <input id="nl-format-for-llm" type="checkbox" checked />
                  <span>Format response for LLM</span>
                </label>
              </div>
            </div>
          </details>
        </div>

        <div class="card">
          <div class="step-heading">
            <span class="step-badge">2</span>
            <div class="title-wrap">
              <h2>Results</h2>
              <p>Clean output first. Technical details are collapsed below.</p>
            </div>
          </div>

          <div id="inline-alert" class="alert">
            <span id="inline-alert-text"></span>
            <button id="retry-btn" class="inline-btn hidden" type="button">Retry</button>
          </div>

          <div id="result-shell" class="result-shell">
            <div class="empty-state">
              No result yet. Submit a query to see summary, chunks, and metadata.
            </div>
          </div>
        </div>
      </section>

      <aside class="card help-card">
        <h3>How modes work</h3>
        <p><strong>Semantic (/query)</strong> returns a summary plus ranked chunks from stratified data.</p>
        <p><strong>NL->SQL (/nl_query)</strong> converts plain language to SQL and returns row-level records.</p>
        <p class="small">Endpoint targets: <code>/api/v1/query</code> and <code>/api/v1/nl_query</code>.</p>
        <p class="small">Note: <code>/nl_query</code> requires Replit/PostgreSQL support.</p>

        <h3>Try an example</h3>
        <div class="example-grid">
          <button class="example-btn" data-mode="query" data-query="Investment opportunities in renewable energy">Semantic: renewable energy trend</button>
          <button class="example-btn" data-mode="query" data-query="Current geopolitical events and financial market developments">Semantic: geopolitical + market overview</button>
          <button class="example-btn" data-mode="nl" data-query="Show posts about Bitcoin from the last 3 hours" >NL->SQL: time-filtered Bitcoin posts</button>
          <button class="example-btn" data-mode="nl" data-query="Find messages containing AI regulation from this week" >NL->SQL: content + time filters</button>
        </div>

        <h3>Polling behavior</h3>
        <ul class="help-list">
          <li>Background query polling runs every 2s for up to 60s.</li>
          <li>If still running, use the manual "Refresh status" button.</li>
        </ul>
      </aside>
    </div>
  </main>

  <script>
    (function () {
      "use strict";

      const API_BASE = "/api/v1";
      const QUERY_ENDPOINT = API_BASE + "/query";
      const NL_QUERY_ENDPOINT = API_BASE + "/nl_query";
      const POLL_INTERVAL_MS = 2000;
      const AUTO_POLL_DURATION_MS = 60000;
      const FETCH_TIMEOUT_MS = 65000;
      const QUERY_SYNC_TIMEOUT_MS = 300000;
      const NL_QUERY_TIMEOUT_MS = 90000;
      const DEFAULT_ROW_PREVIEW = 25;
      const CHUNK_TRUNCATE_LIMIT = 320;

      const elements = {
        modeQueryBtn: document.getElementById("mode-query"),
        modeNlBtn: document.getElementById("mode-nl"),
        modeHint: document.getElementById("mode-hint"),
        queryInput: document.getElementById("query-input"),
        queryError: document.getElementById("query-error"),
        submitBtn: document.getElementById("submit-btn"),
        refreshStatusBtn: document.getElementById("refresh-status-btn"),
        taskIdInline: document.getElementById("task-id-inline"),
        resultShell: document.getElementById("result-shell"),
        statePill: document.getElementById("result-state-pill"),
        stateText: document.getElementById("result-state-text"),
        inlineAlert: document.getElementById("inline-alert"),
        inlineAlertText: document.getElementById("inline-alert-text"),
        retryBtn: document.getElementById("retry-btn"),
        advancedQuery: document.getElementById("advanced-query"),
        advancedNl: document.getElementById("advanced-nl"),
        qUseBackground: document.getElementById("q-use-background"),
        qForceRefresh: document.getElementById("q-force-refresh"),
        qSkipEmbeddings: document.getElementById("q-skip-embeddings"),
        nlLimit: document.getElementById("nl-limit"),
        nlFormatForLlm: document.getElementById("nl-format-for-llm"),
      };

      const state = {
        mode: "query",
        loading: false,
        lastRequest: null,
        polling: {
          taskId: null,
          timer: null,
          startedAt: 0,
          lastStatus: null,
        },
        statusTrail: [],
      };

      function escapeHtml(value) {
        const source = String(value == null ? "" : value);
        return source
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#39;");
      }

      function prettyJson(value) {
        try {
          return JSON.stringify(value ?? {}, null, 2);
        } catch (error) {
          return "{}";
        }
      }

      function nowLabel() {
        return new Date().toLocaleTimeString();
      }

      function appendTrail(message) {
        state.statusTrail.unshift({ time: nowLabel(), message: message });
        state.statusTrail = state.statusTrail.slice(0, 40);
      }

      function renderTrailHtml() {
        if (!state.statusTrail.length) {
          return '<ul class="status-trail"><li>No status events yet.</li></ul>';
        }
        const items = state.statusTrail.map((item) => (
          '<li><span class="status-time">' + escapeHtml(item.time) + '</span>' + escapeHtml(item.message) + "</li>"
        )).join("");
        return '<ul class="status-trail">' + items + "</ul>";
      }

      function setResultState(nextState, label) {
        elements.statePill.classList.remove("state-idle", "state-loading", "state-success", "state-error");
        elements.statePill.classList.add("state-" + nextState);
        elements.stateText.textContent = label;
      }

      function setLoading(loading) {
        state.loading = loading;
        elements.submitBtn.disabled = loading;
        elements.submitBtn.textContent = loading ? "Running..." : "Run Query";
      }

      function setMode(mode) {
        state.mode = mode === "nl" ? "nl" : "query";
        const isQueryMode = state.mode === "query";
        elements.modeQueryBtn.setAttribute("aria-pressed", isQueryMode ? "true" : "false");
        elements.modeNlBtn.setAttribute("aria-pressed", isQueryMode ? "false" : "true");
        elements.advancedQuery.classList.toggle("active", isQueryMode);
        elements.advancedNl.classList.toggle("active", !isQueryMode);
        elements.modeHint.innerHTML = isQueryMode
          ? 'Uses <code>/api/v1/query</code> and returns summaries with relevant chunks.'
          : 'Uses <code>/api/v1/nl_query</code> and returns SQL-backed records.';
        hideInlineAlert();
      }

      function clearValidation() {
        elements.queryError.classList.remove("visible");
      }

      function showValidation(message) {
        elements.queryError.textContent = message;
        elements.queryError.classList.add("visible");
      }

      function showInlineAlert(message, kind, retryHandler) {
        elements.inlineAlert.classList.remove("error", "info");
        elements.inlineAlert.classList.add(kind === "error" ? "error" : "info");
        elements.inlineAlertText.textContent = message;
        elements.inlineAlert.classList.add("visible");
        if (typeof retryHandler === "function") {
          elements.retryBtn.classList.remove("hidden");
          elements.retryBtn.onclick = retryHandler;
        } else {
          elements.retryBtn.classList.add("hidden");
          elements.retryBtn.onclick = null;
        }
      }

      function hideInlineAlert() {
        elements.inlineAlert.classList.remove("visible", "error", "info");
        elements.inlineAlertText.textContent = "";
        elements.retryBtn.classList.add("hidden");
        elements.retryBtn.onclick = null;
      }

      function showTaskId(taskId) {
        if (!taskId) {
          elements.taskIdInline.classList.add("hidden");
          elements.taskIdInline.textContent = "";
          return;
        }
        elements.taskIdInline.textContent = "Task ID: " + taskId;
        elements.taskIdInline.classList.remove("hidden");
      }

      function stopPolling() {
        if (state.polling.timer) {
          clearInterval(state.polling.timer);
          state.polling.timer = null;
        }
        state.polling.taskId = null;
        state.polling.startedAt = 0;
        state.polling.lastStatus = null;
        elements.refreshStatusBtn.classList.add("hidden");
      }

      async function fetchJson(url, options, timeoutMs) {
        const opts = options || {};
        const timeout = timeoutMs || FETCH_TIMEOUT_MS;
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeout);
        try {
          const response = await fetch(url, {
            ...opts,
            signal: controller.signal,
          });
          const raw = await response.text();
          let body = {};
          if (raw) {
            try {
              body = JSON.parse(raw);
            } catch (parseError) {
              body = { raw: raw };
            }
          }
          if (!response.ok) {
            const error = new Error("Request failed");
            error.status = response.status;
            error.body = body;
            throw error;
          }
          return body;
        } catch (error) {
          if (error.name === "AbortError") {
            const timeoutError = new Error("Request timed out");
            timeoutError.status = 0;
            timeoutError.body = {};
            throw timeoutError;
          }
          throw error;
        } finally {
          clearTimeout(timer);
        }
      }

      function normalizeQueryPayload(payload) {
        const base = payload && typeof payload === "object" ? payload : {};
        let resultBlock = base;
        if (base.result && typeof base.result === "object") {
          resultBlock = base.result;
        } else if (Array.isArray(base.results) && base.results.length && typeof base.results[0] === "object") {
          resultBlock = base.results[0];
        }
        const chunks = Array.isArray(resultBlock.chunks) ? resultBlock.chunks : [];
        const summary = typeof resultBlock.summary === "string" ? resultBlock.summary : "";
        const metadata = resultBlock.metadata && typeof resultBlock.metadata === "object" ? resultBlock.metadata : {};
        return {
          status: base.status || resultBlock.status || "completed",
          taskId: base.task_id || base.taskId || metadata.task_id || "",
          summary: summary,
          chunks: chunks,
          metadata: metadata,
          raw: payload,
        };
      }

      function normalizeNlPayload(payload) {
        const base = payload && typeof payload === "object" ? payload : {};
        const description = base.description && typeof base.description === "object" ? base.description : {};
        const data = Array.isArray(base.data) ? base.data : [];
        const metadata = base.metadata && typeof base.metadata === "object" ? base.metadata : {};
        return {
          status: String(base.status || "success"),
          query: String(base.query || ""),
          description: description,
          sql: String(base.sql || ""),
          recordCount: Number.isFinite(Number(base.record_count)) ? Number(base.record_count) : data.length,
          data: data,
          executionTimeMs: Number.isFinite(Number(base.execution_time_ms)) ? Number(base.execution_time_ms) : 0,
          metadata: metadata,
          raw: payload,
        };
      }

      function toDisplayValue(value) {
        if (value == null) return "";
        if (typeof value === "string") return value;
        if (typeof value === "number" || typeof value === "boolean") return String(value);
        return prettyJson(value);
      }

      function chunkTextHtml(text) {
        const safe = String(text || "");
        if (safe.length <= CHUNK_TRUNCATE_LIMIT) {
          return '<p class="chunk-text">' + escapeHtml(safe) + "</p>";
        }
        const shortText = safe.slice(0, CHUNK_TRUNCATE_LIMIT).trimEnd() + "...";
        return (
          '<div class="expand-wrap">' +
            '<p class="chunk-text">' + escapeHtml(shortText) + "</p>" +
            "<details>" +
              "<summary>Expand full text</summary>" +
              '<p class="chunk-text">' + escapeHtml(safe) + "</p>" +
            "</details>" +
          "</div>"
        );
      }

      function buildChunkBadges(chunk) {
        const metadata = chunk && chunk.metadata && typeof chunk.metadata === "object" ? chunk.metadata : {};
        const entries = [];
        const sourceValue = metadata.source || metadata.channel || metadata.platform;
        const dateValue = metadata.timestamp || metadata.posted_date_time || metadata.date;
        const threadValue = metadata.thread_id || metadata.thread || metadata.id;
        const scoreValue = chunk && chunk.score != null ? chunk.score : metadata.score;
        if (sourceValue) entries.push("source: " + sourceValue);
        if (dateValue) entries.push("date: " + dateValue);
        if (threadValue) entries.push("thread: " + threadValue);
        if (scoreValue != null && scoreValue !== "") entries.push("score: " + scoreValue);
        if (!entries.length) return "";
        return '<div class="badge-row">' + entries.map((label) => (
          '<span class="badge">' + escapeHtml(label) + "</span>"
        )).join("") + "</div>";
      }

      function renderQueryResult(normalized) {
        const summaryHtml = normalized.summary
          ? (
              '<div class="summary-panel">' +
                "<h3>Summary</h3>" +
                '<p class="summary-text">' + escapeHtml(normalized.summary) + "</p>" +
              "</div>"
            )
          : (
              '<div class="summary-panel">' +
                "<h3>Summary</h3>" +
                '<p class="summary-text muted">No summary returned.</p>' +
              "</div>"
            );

        const chunkCards = normalized.chunks.map((chunk, index) => {
          const text = chunk && typeof chunk.text === "string"
            ? chunk.text
            : (chunk && typeof chunk.content === "string" ? chunk.content : prettyJson(chunk));
          return (
            '<article class="chunk-card">' +
              '<div class="chunk-head">' +
                '<p class="chunk-title">Chunk ' + (index + 1) + "</p>" +
              "</div>" +
              buildChunkBadges(chunk || {}) +
              chunkTextHtml(text) +
            "</article>"
          );
        }).join("");

        const chunkListHtml = normalized.chunks.length
          ? '<div class="chunk-list">' + chunkCards + "</div>"
          : '<div class="empty-state">No chunks were returned for this query.</div>';

        const detailsHtml =
          '<details class="collapsible">' +
            "<summary>Metadata JSON</summary>" +
            '<pre class="json-panel">' + escapeHtml(prettyJson(normalized.metadata)) + "</pre>" +
          "</details>" +
          '<details class="collapsible">' +
            "<summary>Task and status trail</summary>" +
            renderTrailHtml() +
          "</details>";

        elements.resultShell.innerHTML = summaryHtml + chunkListHtml + detailsHtml;
      }

      function descriptionChips(description) {
        const chips = [];
        if (Array.isArray(description.filters)) {
          description.filters.forEach((entry) => {
            if (entry) chips.push(String(entry));
          });
        }
        if (description.time_filter) chips.push("time: " + description.time_filter);
        if (description.content_filter) chips.push("content: " + description.content_filter);
        if (description.limit) chips.push("limit: " + description.limit);
        if (!chips.length) return '<p class="muted small">No structured filters returned.</p>';
        return '<div class="chip-row">' + chips.map((chip) => (
          '<span class="filter-chip">' + escapeHtml(chip) + "</span>"
        )).join("") + "</div>";
      }

      function buildNlTable(records, showAll) {
        if (!records.length) {
          return '<div class="empty-state">No rows returned for this query.</div>';
        }

        const visibleRecords = showAll ? records : records.slice(0, DEFAULT_ROW_PREVIEW);
        const columnsSet = new Set();
        records.forEach((row) => {
          if (row && typeof row === "object") {
            Object.keys(row).forEach((key) => columnsSet.add(key));
          }
        });
        const columns = Array.from(columnsSet);
        if (!columns.length) {
          return '<div class="empty-state">Rows returned, but no table columns were found.</div>';
        }

        const headHtml = columns.map((col) => "<th>" + escapeHtml(col) + "</th>").join("");
        const bodyHtml = visibleRecords.map((row) => {
          const cells = columns.map((col) => {
            const rawValue = row && typeof row === "object" ? row[col] : "";
            const printable = toDisplayValue(rawValue);
            if (typeof printable === "string" && printable.length > CHUNK_TRUNCATE_LIMIT) {
              const shortValue = printable.slice(0, CHUNK_TRUNCATE_LIMIT).trimEnd() + "...";
              return "<td>" +
                '<div class="expand-wrap"><p class="chunk-text">' + escapeHtml(shortValue) + "</p>" +
                "<details><summary>Expand full value</summary><p class=\"chunk-text\">" + escapeHtml(printable) + "</p></details></div>" +
              "</td>";
            }
            return "<td>" + escapeHtml(printable) + "</td>";
          }).join("");
          return "<tr>" + cells + "</tr>";
        }).join("");

        const showMoreButton = records.length > DEFAULT_ROW_PREVIEW
          ? (
              '<div class="actions">' +
                '<button id="toggle-rows-btn" class="secondary" type="button">' +
                  (showAll ? "Show first 25 rows" : "Show all rows (" + records.length + ")") +
                "</button>" +
              "</div>"
            )
          : "";

        return (
          '<div class="table-wrap"><table><thead><tr>' + headHtml + "</tr></thead><tbody>" + bodyHtml + "</tbody></table></div>" +
          showMoreButton
        );
      }

      function renderNlResult(normalized, showAllRows) {
        const kpiHtml =
          '<div class="kpi-grid">' +
            '<div class="kpi"><p class="label">Status</p><p class="value">' + escapeHtml(normalized.status) + "</p></div>" +
            '<div class="kpi"><p class="label">Record count</p><p class="value">' + escapeHtml(normalized.recordCount) + "</p></div>" +
            '<div class="kpi"><p class="label">Execution ms</p><p class="value">' + escapeHtml(normalized.executionTimeMs.toFixed(2)) + "</p></div>" +
          "</div>";

        const filtersHtml =
          '<div class="summary-panel">' +
            "<h3>Parsed filters</h3>" +
            descriptionChips(normalized.description) +
          "</div>";

        const tableHtml = buildNlTable(normalized.data, showAllRows);
        const detailsHtml =
          '<details class="collapsible">' +
            "<summary>Generated SQL</summary>" +
            '<pre class="json-panel">' + escapeHtml(normalized.sql || "No SQL returned.") + "</pre>" +
          "</details>" +
          '<details class="collapsible">' +
            "<summary>Metadata JSON</summary>" +
            '<pre class="json-panel">' + escapeHtml(prettyJson(normalized.metadata)) + "</pre>" +
          "</details>" +
          '<details class="collapsible">' +
            "<summary>Task and status trail</summary>" +
            renderTrailHtml() +
          "</details>";

        elements.resultShell.innerHTML = kpiHtml + filtersHtml + tableHtml + detailsHtml;

        const toggleRowsBtn = document.getElementById("toggle-rows-btn");
        if (toggleRowsBtn) {
          toggleRowsBtn.addEventListener("click", () => {
            renderNlResult(normalized, !showAllRows);
          });
        }
      }

      function renderEnvRestriction(errorBody) {
        const additional = errorBody && typeof errorBody === "object" ? errorBody.additional_info || {} : {};
        const message = additional.message || errorBody.detail || "Natural language query is not available in this environment.";
        const possibleSolution = additional.possible_solution || "Use /query mode or run in Replit with PostgreSQL configured.";
        elements.resultShell.innerHTML =
          '<div class="restriction-card">' +
            "<h4>NL->SQL unavailable in this environment</h4>" +
            "<p>" + escapeHtml(message) + "</p>" +
            "<p class=\"small\">" + escapeHtml(possibleSolution) + "</p>" +
          "</div>";
      }

      function parseErrorMessage(error) {
        if (!error) return "Unknown error";
        if (error.status === 0) return "Network timeout while waiting for response.";
        const body = error.body;
        if (body && typeof body === "object") {
          if (typeof body.detail === "string") return body.detail;
          if (body.detail && typeof body.detail === "object") {
            if (typeof body.detail.message === "string") return body.detail.message;
            return prettyJson(body.detail);
          }
          if (typeof body.message === "string") return body.message;
        }
        return "Request failed. Please retry.";
      }

      function isNlEnvironmentRestriction(error) {
        if (!error || error.status !== 400) return false;
        const body = error.body && typeof error.body === "object" ? error.body : {};
        const additional = body.additional_info && typeof body.additional_info === "object" ? body.additional_info : {};
        const code = additional.error || body.error_code || "";
        const message = additional.message || body.detail || "";
        return String(code).toUpperCase() === "ENVIRONMENT_RESTRICTION" || String(message).toLowerCase().includes("replit");
      }

      async function submitQueryMode(requestBody) {
        appendTrail("Submitting /query request");
        const payload = await fetchJson(QUERY_ENDPOINT, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestBody),
        }, QUERY_SYNC_TIMEOUT_MS);

        const normalized = normalizeQueryPayload(payload);
        if (requestBody.use_background && String(payload.status || "").toLowerCase() === "processing" && payload.task_id) {
          appendTrail("Background task started");
          showTaskId(payload.task_id);
          startPolling(payload.task_id);
          setResultState("loading", "Polling");
          showInlineAlert("Background task started. Polling status every 2 seconds.", "info");
          elements.resultShell.innerHTML = '<div class="empty-state">Task is running in background. Waiting for completion...</div>';
          return;
        }

        appendTrail("Query completed");
        setResultState("success", "Success");
        showTaskId(normalized.taskId || payload.task_id || "");
        renderQueryResult(normalized);
      }

      async function submitNlMode(requestBody) {
        appendTrail("Submitting /nl_query request");
        const payload = await fetchJson(NL_QUERY_ENDPOINT, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestBody),
        }, NL_QUERY_TIMEOUT_MS);
        const normalized = normalizeNlPayload(payload);
        appendTrail("NL query completed");
        setResultState("success", "Success");
        showTaskId(normalized.metadata && normalized.metadata.task_id ? normalized.metadata.task_id : "");
        renderNlResult(normalized, false);
      }

      async function pollOnce(taskId) {
        const payload = await fetchJson(API_BASE + "/batch_status/" + encodeURIComponent(taskId), {
          method: "GET",
        }, 30000);

        const status = String(payload.status || "").toLowerCase();
        if (status && status !== state.polling.lastStatus) {
          appendTrail("Task status: " + status);
          state.polling.lastStatus = status;
        }

        if (status === "completed") {
          stopPolling();
          setResultState("success", "Success");
          showInlineAlert("Background task completed.", "info");
          const normalized = normalizeQueryPayload(payload);
          renderQueryResult(normalized);
          return true;
        }
        if (status === "failed" || status === "expired") {
          stopPolling();
          setResultState("error", "Error");
          const message = payload.error || payload.message || ("Task " + status + ".");
          showInlineAlert(message, "error", () => retryLastRequest());
          elements.resultShell.innerHTML = '<div class="empty-state">' + escapeHtml(message) + "</div>";
          return true;
        }

        const elapsed = Date.now() - state.polling.startedAt;
        if (elapsed > AUTO_POLL_DURATION_MS) {
          if (state.polling.timer) {
            clearInterval(state.polling.timer);
            state.polling.timer = null;
          }
          elements.refreshStatusBtn.classList.remove("hidden");
          showInlineAlert("Still processing. Automatic polling stopped after 60s. Use Refresh status.", "info");
          appendTrail("Automatic polling stopped after 60 seconds");
          return false;
        }
        return false;
      }

      function startPolling(taskId) {
        stopPolling();
        state.polling.taskId = taskId;
        state.polling.startedAt = Date.now();
        elements.refreshStatusBtn.classList.remove("hidden");
        elements.refreshStatusBtn.onclick = async function () {
          if (!state.polling.taskId) return;
          try {
            await pollOnce(state.polling.taskId);
          } catch (error) {
            setResultState("error", "Error");
            showInlineAlert(parseErrorMessage(error), "error", () => retryLastRequest());
          }
        };

        pollOnce(taskId).catch((error) => {
          setResultState("error", "Error");
          showInlineAlert(parseErrorMessage(error), "error", () => retryLastRequest());
        });

        state.polling.timer = setInterval(() => {
          if (!state.polling.taskId) return;
          pollOnce(state.polling.taskId).catch((error) => {
            stopPolling();
            setResultState("error", "Error");
            showInlineAlert(parseErrorMessage(error), "error", () => retryLastRequest());
          });
        }, POLL_INTERVAL_MS);
      }

      function currentRequestBody() {
        const queryText = elements.queryInput.value.trim();
        if (state.mode === "query") {
          return {
            mode: "query",
            payload: {
              query: queryText,
              use_background: !!elements.qUseBackground.checked,
              force_refresh: !!elements.qForceRefresh.checked,
              skip_embeddings: !!elements.qSkipEmbeddings.checked,
            },
          };
        }
        return {
          mode: "nl",
          payload: {
            query: queryText,
            limit: Number(elements.nlLimit.value || 50),
            format_for_llm: !!elements.nlFormatForLlm.checked,
          },
        };
      }

      async function runSubmit() {
        clearValidation();
        hideInlineAlert();

        const requestData = currentRequestBody();
        const queryText = requestData.payload.query;
        if (!queryText) {
          showValidation("Please enter a query before submitting.");
          setResultState("error", "Validation");
          return;
        }

        stopPolling();
        state.statusTrail = [];
        showTaskId("");
        setLoading(true);
        setResultState("loading", "Loading");
        appendTrail("Request started");
        state.lastRequest = requestData;

        try {
          if (requestData.mode === "query") {
            await submitQueryMode(requestData.payload);
          } else {
            await submitNlMode(requestData.payload);
          }
        } catch (error) {
          setResultState("error", "Error");
          const message = parseErrorMessage(error);
          appendTrail("Request failed: " + message);

          if (requestData.mode === "nl" && isNlEnvironmentRestriction(error)) {
            renderEnvRestriction(error.body || {});
            showInlineAlert("NL->SQL mode requires Replit/PostgreSQL in this deployment.", "info");
          } else if (requestData.mode === "query" && error.status === 0) {
            showInlineAlert(
              "Query timed out in the browser. Enable 'Use background processing' in Advanced options and retry.",
              "error",
              async () => {
                elements.qUseBackground.checked = true;
                await retryLastRequest();
              }
            );
            elements.resultShell.innerHTML = '<div class="empty-state">The request exceeded the browser timeout window. Retry in background mode for long-running queries.</div>';
          } else {
            showInlineAlert(message, "error", () => retryLastRequest());
            elements.resultShell.innerHTML = '<div class="empty-state">' + escapeHtml(message) + "</div>";
          }
        } finally {
          setLoading(false);
        }
      }

      async function retryLastRequest() {
        if (!state.lastRequest) return;
        elements.queryInput.value = state.lastRequest.payload.query || elements.queryInput.value;
        if (state.lastRequest.mode === "query") {
          setMode("query");
          elements.qUseBackground.checked = !!state.lastRequest.payload.use_background;
          elements.qForceRefresh.checked = !!state.lastRequest.payload.force_refresh;
          elements.qSkipEmbeddings.checked = !!state.lastRequest.payload.skip_embeddings;
        } else {
          setMode("nl");
          elements.nlLimit.value = Number(state.lastRequest.payload.limit || 50);
          elements.nlFormatForLlm.checked = !!state.lastRequest.payload.format_for_llm;
        }
        await runSubmit();
      }

      function bindEvents() {
        elements.modeQueryBtn.addEventListener("click", () => setMode("query"));
        elements.modeNlBtn.addEventListener("click", () => setMode("nl"));
        elements.submitBtn.addEventListener("click", runSubmit);
        elements.queryInput.addEventListener("input", clearValidation);

        document.querySelectorAll(".example-btn").forEach((btn) => {
          btn.addEventListener("click", () => {
            const mode = btn.getAttribute("data-mode") === "nl" ? "nl" : "query";
            const query = btn.getAttribute("data-query") || "";
            setMode(mode);
            elements.queryInput.value = query;
            elements.queryInput.focus();
          });
        });
      }

      setMode("query");
      bindEvents();
      setResultState("idle", "Idle");
    })();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def query_ui_home() -> HTMLResponse:
    """Serve query UI page."""
    return HTMLResponse(content=QUERY_UI_HTML)
