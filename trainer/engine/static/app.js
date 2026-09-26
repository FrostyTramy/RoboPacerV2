// RoboPacerV2 Trainer frontend. Talks to server.py's /api/* routes only -
// see trainer/engine/server.py for the route contract.

const logEl = document.getElementById("log");
const statusText = document.getElementById("status-text");
const statusElapsed = document.getElementById("status-elapsed");
const statusLoss = document.getElementById("status-loss");
const statusModels = document.getElementById("status-models");
const stopBtn = document.getElementById("stop-btn");

let evtSource = null;
let jobStartMs = null;
let elapsedTimer = null;

// ── Mode (Standard/Custom) + action tab switching ───────────────────────

document.querySelectorAll("[data-mode-btn]").forEach(btn => {
  btn.addEventListener("click", () => {
    document.body.dataset.mode = btn.dataset.modeBtn;
    document.querySelectorAll("[data-mode-btn]").forEach(b => b.classList.toggle("active", b === btn));
  });
});

document.querySelectorAll("[data-action-btn]").forEach(btn => {
  btn.addEventListener("click", () => switchAction(btn.dataset.actionBtn));
});

function switchAction(action) {
  document.body.dataset.action = action;
  renderPrereqs();
  document.querySelectorAll("[data-action-btn]").forEach(b => b.classList.toggle("active", b.dataset.actionBtn === action));
  document.querySelectorAll(".action-panel").forEach(p => p.classList.toggle("active", p.dataset.action === action));
}

// A job's "action" (as reported by /api/status.current_job.action) doesn't
// always match a visible tab 1:1 - retry-compile/smoketest live inside the
// "full" tab, smoketest-compile lives inside "compile-only".
function panelForJobAction(action) {
  if (action === "retry-compile" || action === "smoketest") return "full";
  if (action === "smoketest-compile") return "compile-only";
  return action;
}

// ── Defaults prefill ─────────────────────────────────────────────────────

function applyDefaults(defaults) {
  document.querySelectorAll("[data-field]").forEach(el => {
    const key = el.dataset.field;
    if (!(key in defaults)) return;
    if (el.type === "checkbox") {
      el.checked = !!defaults[key];
    } else {
      el.value = defaults[key];
    }
  });
}

// ── Path validation on blur ──────────────────────────────────────────────

document.querySelectorAll("[data-validate-kind]").forEach(input => {
  input.addEventListener("blur", () => validatePath(input));
});

async function validatePath(input) {
  const kind = input.dataset.validateKind;
  const resultEl = input.closest(".field").querySelector(".validate-result");
  const path = input.value.trim();
  if (!path) { resultEl.textContent = ""; resultEl.className = "validate-result"; return; }
  resultEl.textContent = "Checking...";
  resultEl.className = "validate-result";
  try {
    const res = await fetch(`/api/validate?kind=${kind}&path=${encodeURIComponent(path)}`);
    const data = await res.json();
    if (data.ok) {
      resultEl.className = "validate-result ok";
      if (kind === "dataset") {
        resultEl.textContent = `✓ ${data.record_count} frames, ${data.format} (${data.frame_stack_n}-frame stack)`;
      } else if (kind === "pth") {
        resultEl.textContent = `✓ checkpoint OK (frame_stack_n=${data.frame_stack_n})`;
        const panel = input.closest(".action-panel");
        // Training saves <name>_calib_data_nhwc.npy next to the .pth - pre-fill it.
        const calibInput = panel.querySelector('[data-field="calib_npy"]');
        if (calibInput && data.calib_npy && autoFillable(calibInput)) {
          autoFill(calibInput, data.calib_npy);
          validatePath(calibInput);
        }
        // Output name = the .pth's own name (models/<name>.pth -> <name>.hef).
        const nameInput = panel.querySelector('[data-field="model_name"]');
        const stem = path.split(/[\\/]/).pop().replace(/\.pth$/i, "");
        if (nameInput && stem && autoFillable(nameInput)) {
          autoFill(nameInput, stem);
        }
      } else if (kind === "npy") {
        resultEl.textContent = `✓ ${data.record_count} calibration samples (${data.frame_stack_n}-frame stack)`;
      }
    } else {
      resultEl.className = "validate-result error";
      resultEl.textContent = `✗ ${data.error}`;
    }
  } catch (e) {
    resultEl.className = "validate-result error";
    resultEl.textContent = "✗ validation request failed";
  }
}

// A field filled in from the picked .pth follows the next .pth picked too -
// unless the user has typed their own value into it since.
function autoFillable(el) {
  return !el.value.trim() || el.value === el.dataset.autoFilled;
}

function autoFill(el, value) {
  el.value = value;
  el.dataset.autoFilled = value;
}

// ── Folder/file browse buttons ───────────────────────────────────────────

document.querySelectorAll("[data-browse]").forEach(btn => {
  btn.addEventListener("click", async () => {
    const input = document.getElementById(btn.dataset.browse);
    const isFolder = btn.dataset.browseType === "folder";
    const url = isFolder ? "/api/browse/folder" : `/api/browse/file?type=${btn.dataset.fileType || "pth"}`;
    const res = await fetch(url);
    const data = await res.json();
    if (data.path) {
      input.value = data.path;
      input.dispatchEvent(new Event("blur"));
    }
  });
});

// ── Config collection + job launching ────────────────────────────────────

function collectConfig(panel) {
  const config = {};
  panel.querySelectorAll("[data-field]").forEach(el => {
    const key = el.dataset.field;
    if (el.type === "checkbox") {
      config[key] = el.checked;
    } else if (el.type === "number") {
      if (el.value !== "") config[key] = parseFloat(el.value);
    } else if (el.value.trim() !== "") {
      config[key] = el.value.trim();
    }
  });
  return config;
}

async function launch(route, panelName) {
  const panel = document.querySelector(`.action-panel[data-action="${panelName}"]`);
  const config = collectConfig(panel);
  if (route !== "/api/run/train-only") {
    // Everything except Train only ends in a Docker compile - check fresh
    // (Docker may have just been started) and stop BEFORE a long training run.
    const p = await refreshPrereqs();
    if (p && !(p.docker_running && p.wheel_present)) {
      appendLog(prereqProblem(p) + " (Train only doesn't need Docker.)", "error");
      return;
    }
  }
  if (route === "/api/run/compile-only" && !config.calib_npy) {
    appendLog("Pick the calibration .npy (saved by training next to the .pth).", "error");
    return;
  }
  logEl.textContent = "";
  statusLoss.textContent = "";
  let res, data;
  try {
    res = await fetch(route, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    data = await res.json();
  } catch (e) {
    appendLog("Request failed: " + e, "error");
    return;
  }
  if (!res.ok) {
    appendLog(data.error || "Failed to start.", "error");
    return;
  }
  setRunning(true, Date.now() / 1000);
  startStream();
}

document.getElementById("full_start").addEventListener("click", () => launch("/api/run/full", "full"));
document.getElementById("full_retry_compile").addEventListener("click", () => launch("/api/run/retry-compile", "full"));
document.getElementById("full_smoketest").addEventListener("click", () => launch("/api/run/smoketest", "full"));
document.getElementById("trainonly_start").addEventListener("click", () => launch("/api/run/train-only", "train-only"));
document.getElementById("compileonly_start").addEventListener("click", () => launch("/api/run/compile-only", "compile-only"));
document.getElementById("compileonly_smoketest").addEventListener("click", () => launch("/api/run/smoketest-compile", "compile-only"));

stopBtn.addEventListener("click", () => fetch("/api/stop", { method: "POST" }));

// ── Docker / Hailo compiler reminder (always visible) ────────────────────

const prereqBanner = document.getElementById("prereq-banner");
let lastPrereqs = null;

function prereqProblem(p) {
  if (!p.docker_installed) return "Docker Desktop is not installed - compiling needs it (see INSTALL.md).";
  if (!p.docker_running) return "Docker Desktop is not running - start it before Full or Compile.";
  if (!p.wheel_present) return "Hailo compiler wheel missing in engine/compile/resources/ (see INSTALL.md).";
  return null;
}

function renderPrereqs() {
  if (!lastPrereqs) return;
  const problem = prereqProblem(lastPrereqs);
  const trainOnly = document.body.dataset.action === "train-only";
  if (!problem) {
    prereqBanner.className = "prereq-banner ok";
    prereqBanner.textContent = "✓ Docker running · Hailo compiler ready";
  } else if (trainOnly) {
    prereqBanner.className = "prereq-banner info";
    prereqBanner.textContent = problem + " (Train only doesn't need it.)";
  } else {
    prereqBanner.className = "prereq-banner warn";
    prereqBanner.textContent = "⚠ " + problem;
  }
}

async function refreshPrereqs() {
  try {
    lastPrereqs = await (await fetch("/api/prereqs")).json();
    renderPrereqs();
  } catch (e) { /* server busy/restarting - keep the last known state */ }
  return lastPrereqs;
}

refreshPrereqs();
setInterval(() => { if (!document.hidden) refreshPrereqs(); }, 5000);

// ── Status bar / elapsed timer ───────────────────────────────────────────

function setRunning(running, startedAtSeconds) {
  stopBtn.disabled = !running;
  document.querySelectorAll(".start-btn").forEach(b => b.disabled = running);
  if (running) {
    jobStartMs = startedAtSeconds ? startedAtSeconds * 1000 : Date.now();
    statusText.textContent = "Running";
    clearInterval(elapsedTimer);
    elapsedTimer = setInterval(updateElapsed, 1000);
    updateElapsed();
  } else {
    clearInterval(elapsedTimer);
  }
}

function updateElapsed() {
  const secs = Math.max(0, Math.floor((Date.now() - jobStartMs) / 1000));
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  statusElapsed.textContent = `${m}m ${s}s`;
}

async function refreshModels() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    statusModels.textContent = data.models.length ? `Compiled: ${data.models.join(", ")}` : "";
  } catch (e) { /* ignore */ }
}

// ── SSE stream handling ──────────────────────────────────────────────────

function startStream() {
  if (evtSource) evtSource.close();
  evtSource = new EventSource("/api/stream");
  evtSource.onmessage = (e) => handleEvent(JSON.parse(e.data));
  evtSource.onerror = () => { /* EventSource retries automatically */ };
}

function handleEvent(ev) {
  switch (ev.type) {
    case "heartbeat":
      appendLog("...", "heartbeat");
      break;
    case "log":
      appendLog(ev.text, ev.level);
      break;
    case "split":
      appendLog(`Split: ${ev.train} train / ${ev.val} val / ${ev.total} total`, "info");
      break;
    case "epoch":
      appendLog(`Epoch ${ev.epoch}/${ev.total} - train ${ev.train}  val ${ev.val}${ev.best ? "  (best)" : ""}`,
                 ev.best ? "success" : "info");
      statusLoss.textContent = `Epoch ${ev.epoch}/${ev.total} - train ${ev.train} / val ${ev.val}`;
      break;
    case "file":
      appendLog(`Saved: ${ev.name}`, "success");
      break;
    case "done":
      finishJob();
      break;
  }
}

function finishJob() {
  if (evtSource) { evtSource.close(); evtSource = null; }
  setRunning(false);
  statusText.textContent = "Done";
  refreshModels();
}

function appendLog(text, level) {
  const line = document.createElement("div");
  line.className = `log-${level || "info"}`;
  line.textContent = text;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

// ── Startup: load defaults, restore in-progress job if any ──────────────

async function init() {
  try {
    const defaults = await (await fetch("/api/defaults")).json();
    applyDefaults(defaults);
  } catch (e) { /* server not reachable yet - fields keep their HTML defaults */ }

  await refreshModels();

  try {
    const status = await (await fetch("/api/status")).json();
    if (status.running && status.current_job) {
      const panel = panelForJobAction(status.current_job.action);
      switchAction(panel);
      document.querySelectorAll(`[data-action-btn="${panel}"]`).forEach(b => b.classList.add("active"));
      if (status.current_job.model_name) {
        document.querySelectorAll(`.action-panel[data-action="${panel}"] [data-field="model_name"]`)
          .forEach(el => el.value = status.current_job.model_name);
      }
      setRunning(true, status.current_job.started_at);
      startStream();
    }
  } catch (e) { /* server not reachable yet */ }
}

init();
