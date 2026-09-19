"use strict";

const $ = (id) => document.getElementById(id);
const MAX_RECORD_MS = 5 * 60 * 1000;
const QUEUE_KEY = "kikuyu.pendingCorrections";

const state = {
  file: null,
  recorder: null,
  chunks: [],
  startedAt: 0,
  timer: null,
  lastResult: null,
  busy: false,
};

/* ---------- shell ---------- */

function setStatus(text, kind) {
  $("status-text").textContent = text;
  $("status").className = "status" + (kind ? " " + kind : "");
}

function showError(message) {
  const box = $("error");
  box.textContent = message;
  box.hidden = !message;
}

function setBusy(busy, message) {
  state.busy = busy;
  $("progress").hidden = !busy;
  if (message) $("progress-text").textContent = message;
  $("send-text").disabled = busy;
  $("send-file").disabled = busy || !state.file;
  const record = $("record");
  if (record) record.disabled = busy;
}

function selectTab(name) {
  for (const tab of ["speak", "type"]) {
    const selected = tab === name;
    $("tab-" + tab).setAttribute("aria-selected", String(selected));
    $("panel-" + tab).hidden = !selected;
  }
}

$("tab-speak").addEventListener("click", () => selectTab("speak"));
$("tab-type").addEventListener("click", () => selectTab("type"));

// #type opens the typing tab directly, which is handy for a home-screen shortcut.
if (location.hash === "#type") selectTab("type");

/* ---------- health ---------- */

async function checkHealth() {
  if (!navigator.onLine) {
    setStatus("Offline — the app shell is cached", "off");
    return;
  }
  try {
    const response = await fetch("health", { cache: "no-store" });
    if (!response.ok) throw new Error("HTTP " + response.status);
    const health = await response.json();
    const missing = [];
    if (!health.asr_model) missing.push("speech model");
    if (!health.translation_model) missing.push("translation model");
    if (missing.length) setStatus("Ready, but no " + missing.join(" or "), "off");
    else setStatus("Ready — running offline on this machine", "ok");
  } catch (error) {
    setStatus("Cannot reach the server", "off");
  }
}

window.addEventListener("online", () => { checkHealth(); flushQueue(); });
window.addEventListener("offline", () => setStatus("Offline — the app shell is cached", "off"));

/* ---------- results ---------- */

function renderResult(payload) {
  state.lastResult = payload;
  $("out-kikuyu").textContent = payload.kikuyu || "(nothing recognised)";
  // Typed Kikuyu often lacks its tilde vowels; show what was actually translated.
  const readAs = $("read-as");
  readAs.hidden = !payload.read_as;
  readAs.textContent = payload.read_as ? `Read as ${payload.read_as} — the tilde vowels were filled in.` : "";
  $("out-english").textContent = payload.english || "(no translation)";
  $("fix-text").value = payload.english || "";
  $("fix-status").textContent = "";

  const match = payload.bible_match;
  $("verse").hidden = !match;
  if (match) {
    $("verse-ref").textContent = `${match.book} ${match.chapter}:${match.verse}`;
    $("verse-english").textContent = match.english || "";
  }

  const audio = $("tts");
  const url = payload.audio;
  audio.hidden = !url;
  $("speak").hidden = !url;
  if (url) audio.src = url;

  const links = [];
  const urls = payload.file_urls || {};
  if (urls["subtitles.srt"]) links.push(`<a href="${urls["subtitles.srt"]}" download>Kikuyu subtitles</a>`);
  if (urls["english_subtitles.srt"]) links.push(`<a href="${urls["english_subtitles.srt"]}" download>English subtitles</a>`);
  $("downloads").innerHTML = links.join("");

  $("result").hidden = false;
  $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function readError(response) {
  try {
    const body = await response.json();
    if (body && body.detail) return String(body.detail);
  } catch (error) { /* fall through to the status line */ }
  return `Server returned HTTP ${response.status}`;
}

/* ---------- translate ---------- */

async function translateBlob(blob, filename) {
  showError("");
  setBusy(true, "Transcribing and translating… long recordings take a while.");
  const form = new FormData();
  form.append("file", blob, filename);
  try {
    const response = await fetch("translate", { method: "POST", body: form });
    if (!response.ok) throw new Error(await readError(response));
    renderResult(await response.json());
  } catch (error) {
    showError(navigator.onLine
      ? String(error.message || error)
      : "You are offline. Translating audio needs the server that runs the models.");
  } finally {
    setBusy(false);
  }
}

async function translateText() {
  const text = $("text-input").value.trim();
  if (!text) return;
  showError("");
  setBusy(true, "Translating…");
  try {
    const response = await fetch("translate-text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) throw new Error(await readError(response));
    const payload = await response.json();
    renderResult({ kikuyu: payload.kikuyu, english: payload.english, bible_match: payload.bible_match });
  } catch (error) {
    showError(navigator.onLine
      ? String(error.message || error)
      : "You are offline. Translating needs the server that runs the models.");
  } finally {
    setBusy(false);
  }
}

$("send-text").addEventListener("click", translateText);
$("send-file").addEventListener("click", () => {
  if (state.file) translateBlob(state.file, state.file.name || "recording.wav");
});

$("file").addEventListener("change", (event) => {
  state.file = event.target.files[0] || null;
  $("filename").textContent = state.file ? state.file.name : "";
  $("send-file").disabled = !state.file || state.busy;
});

/* ---------- recording ---------- */

function pickMimeType() {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4", "audio/aac"];
  for (const type of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(type)) return type;
  }
  return "";
}

function extensionFor(mimeType) {
  if (mimeType.includes("webm")) return "webm";
  if (mimeType.includes("ogg")) return "ogg";
  if (mimeType.includes("mp4")) return "m4a";
  if (mimeType.includes("aac")) return "aac";
  return "webm";
}

function tickTimer() {
  const seconds = Math.floor((Date.now() - state.startedAt) / 1000);
  const minutes = String(Math.floor(seconds / 60)).padStart(2, "0");
  $("timer").textContent = `${minutes}:${String(seconds % 60).padStart(2, "0")} — release to translate`;
  if (Date.now() - state.startedAt >= MAX_RECORD_MS) stopRecording();
}

async function startRecording() {
  if (state.recorder || state.busy) return;
  showError("");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    const mimeType = pickMimeType();
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    state.recorder = recorder;
    state.chunks = [];
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data && event.data.size) state.chunks.push(event.data);
    });
    recorder.addEventListener("stop", () => {
      for (const track of stream.getTracks()) track.stop();
      const type = recorder.mimeType || mimeType || "audio/webm";
      const blob = new Blob(state.chunks, { type });
      state.recorder = null;
      if (blob.size > 1000) translateBlob(blob, "recording." + extensionFor(type));
      else showError("That recording was too short. Hold the button while you speak.");
    });
    recorder.start();
    state.startedAt = Date.now();
    state.timer = setInterval(tickTimer, 250);
    $("record").classList.add("recording");
    $("record-label").textContent = "Recording… release to stop";
    tickTimer();
  } catch (error) {
    showError("Microphone permission was refused, or no microphone is available. "
      + "Use “Choose or record an audio file” instead.");
  }
}

function stopRecording() {
  clearInterval(state.timer);
  $("timer").textContent = "";
  $("record").classList.remove("recording");
  $("record-label").textContent = "Hold to record";
  if (state.recorder && state.recorder.state !== "inactive") state.recorder.stop();
}

function setupRecording() {
  const canRecord = window.isSecureContext
    && navigator.mediaDevices
    && typeof navigator.mediaDevices.getUserMedia === "function"
    && window.MediaRecorder;
  $("record-area").hidden = !canRecord;
  $("insecure-note").hidden = canRecord;
  if (!canRecord) return;

  const record = $("record");
  // Pointer events cover mouse and touch, and never fire twice for one press.
  record.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    record.setPointerCapture(event.pointerId);
    startRecording();
  });
  for (const name of ["pointerup", "pointercancel", "pointerleave"]) {
    record.addEventListener(name, stopRecording);
  }
  // Keyboard users get press-and-hold via space or enter.
  record.addEventListener("keydown", (event) => {
    if ((event.key === " " || event.key === "Enter") && !event.repeat) {
      event.preventDefault();
      startRecording();
    }
  });
  record.addEventListener("keyup", (event) => {
    if (event.key === " " || event.key === "Enter") stopRecording();
  });
}

/* ---------- corrections, with an offline queue ---------- */

function readQueue() {
  try {
    return JSON.parse(localStorage.getItem(QUEUE_KEY) || "[]");
  } catch (error) {
    return [];
  }
}

function writeQueue(items) {
  try {
    localStorage.setItem(QUEUE_KEY, JSON.stringify(items));
  } catch (error) { /* private mode or a full quota; the correction is lost, not the app */ }
  const note = $("queue-note");
  note.hidden = items.length === 0;
  note.textContent = items.length
    ? `${items.length} correction(s) saved on this phone, waiting to sync.`
    : "";
}

async function postCorrection(item) {
  const response = await fetch("corrections", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(item),
  });
  if (!response.ok) throw new Error(await readError(response));
}

async function flushQueue() {
  const pending = readQueue();
  if (!pending.length || !navigator.onLine) return;
  const remaining = [];
  for (const item of pending) {
    try {
      await postCorrection(item);
    } catch (error) {
      remaining.push(item);
    }
  }
  writeQueue(remaining);
}

$("save-fix").addEventListener("click", async () => {
  const result = state.lastResult;
  const corrected = $("fix-text").value.trim();
  if (!result || !corrected) return;
  const item = { source: result.kikuyu || "", machine: result.english || "", corrected };
  if (!item.source) {
    $("fix-status").textContent = "Nothing to correct yet.";
    return;
  }
  $("save-fix").disabled = true;
  try {
    await postCorrection(item);
    $("fix-status").textContent = "Saved. It will be used the next time you train.";
    await flushQueue();
  } catch (error) {
    const pending = readQueue();
    pending.push(item);
    writeQueue(pending);
    $("fix-status").textContent = "Saved on this phone. It will sync when the server is reachable.";
  } finally {
    $("save-fix").disabled = false;
  }
});

/* ---------- small actions ---------- */

$("copy").addEventListener("click", async () => {
  const text = $("out-english").textContent;
  try {
    await navigator.clipboard.writeText(text);
    $("copy").textContent = "Copied";
    setTimeout(() => { $("copy").textContent = "Copy"; }, 1500);
  } catch (error) {
    showError("Copying needs HTTPS or localhost. Select the text and copy it by hand.");
  }
});

$("speak").addEventListener("click", () => {
  const audio = $("tts");
  if (audio.src) audio.play().catch(() => showError("Could not play the audio."));
});

/* ---------- install prompt ---------- */

let installPrompt = null;
window.addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  installPrompt = event;
  $("install").hidden = false;
});
$("install").addEventListener("click", async () => {
  if (!installPrompt) return;
  installPrompt.prompt();
  await installPrompt.userChoice;
  installPrompt = null;
  $("install").hidden = true;
});

/* ---------- boot ---------- */

if ("serviceWorker" in navigator && window.isSecureContext) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("sw.js").catch(() => { /* offline shell is optional */ });
  });
}

setupRecording();
writeQueue(readQueue());
flushQueue();
checkHealth();
