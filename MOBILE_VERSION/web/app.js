"use strict";

// The whole app talks to a server running on this same phone, so every request
// is to localhost and none of it needs a network.

const $ = (id) => document.getElementById(id);
const STORE_KEY = "kikuyu.language";

const state = {
  languages: [],
  current: null,
  health: null,
  recorder: null,
  chunks: [],
  recordedBlob: null,
  pickedFile: null,
  timer: null,
  startedAt: 0,
  polling: null,
  last: null,
};

// -- helpers ---------------------------------------------------------------

const show = (element, visible) => { element.hidden = !visible; };

function setStatus(kind, text) {
  const status = $("status");
  status.className = `status ${kind}`;
  $("status-text").textContent = text;
}

function showError(message) {
  const box = $("error");
  box.textContent = message;
  show(box, Boolean(message));
}

function setBusy(busy, message) {
  show($("progress"), busy);
  if (message) $("progress-text").textContent = message;
  $("send-text").disabled = busy;
  $("send-file").disabled = busy || !currentUpload();
}

const currentUpload = () => state.recordedBlob || state.pickedFile;

// -- startup ---------------------------------------------------------------

async function loadHealth() {
  try {
    const response = await fetch("api/health", { cache: "no-store" });
    if (!response.ok) throw new Error(String(response.status));
    state.health = await response.json();
  } catch (error) {
    setStatus("bad", "Cannot reach the translator on this phone.");
    showError("The server is not running. Start it in Termux with ./start.sh");
    return;
  }

  state.languages = state.health.languages;
  const saved = localStorage.getItem(STORE_KEY);
  const preferred = state.languages.find((language) => language.code === saved);
  state.current = preferred || state.languages[0];
  renderLanguages();
  applyLanguage();
}

function renderLanguages() {
  const nav = $("langs");
  nav.replaceChildren();
  for (const language of state.languages) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.code = language.code;
    const name = document.createElement("span");
    name.textContent = language.name;
    const native = document.createElement("span");
    native.className = "native";
    native.textContent = language.native;
    button.append(name, native);
    button.addEventListener("click", () => {
      state.current = language;
      localStorage.setItem(STORE_KEY, language.code);
      applyLanguage();
      show($("result"), false);
      showError("");
    });
    nav.append(button);
  }
}

function applyLanguage() {
  const language = state.current;
  $("lang-name").textContent = language.name;
  $("source-heading").textContent = language.name;
  $("text-input").placeholder = language.placeholder;
  for (const button of $("langs").children) {
    button.setAttribute("aria-current", button.dataset.code === language.code ? "true" : "false");
  }

  // Say which tier will answer, so nobody is surprised by a word-by-word gloss.
  if (language.neural) {
    setStatus("ok", "Full translation model ready — works with no internet.");
  } else if (language.verses && language.gloss) {
    setStatus("warn", "Offline mode: exact Bible verses, plus word-by-word for anything else.");
  } else if (language.verses) {
    setStatus("warn", "Offline mode: exact Bible verses only.");
  } else {
    setStatus("bad", `${language.name} needs the translation model — none of the offline data covers it.`);
  }

  const speechOff = $("speech-off");
  show($("record-area"), language.speech && window.isSecureContext && Boolean(navigator.mediaDevices));
  if (language.speech) {
    show(speechOff, false);
    $("send-file").disabled = !currentUpload();
  } else {
    speechOff.textContent = state.health.speech_installed
      ? `Speech recognition for ${language.name} was not copied to this phone.`
      : "Speech recognition is not installed on this phone. Typed text still translates.";
    show(speechOff, true);
    $("send-file").disabled = true;
  }
}

// -- tabs ------------------------------------------------------------------

function selectTab(which) {
  const isType = which === "type";
  $("tab-type").setAttribute("aria-selected", String(isType));
  $("tab-speak").setAttribute("aria-selected", String(!isType));
  show($("panel-type"), isType);
  show($("panel-speak"), !isType);
}

$("tab-type").addEventListener("click", () => selectTab("type"));
$("tab-speak").addEventListener("click", () => selectTab("speak"));

// -- jobs ------------------------------------------------------------------

async function startJob(request) {
  showError("");
  show($("result"), false);
  setBusy(true, "Working…");
  try {
    const response = await fetch(request.url, request.options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
    pollJob(payload.job);
  } catch (error) {
    setBusy(false);
    showError(error.message);
  }
}

function pollJob(jobId) {
  clearTimeout(state.polling);
  const tick = async () => {
    try {
      const response = await fetch(`api/job/${jobId}`, { cache: "no-store" });
      if (!response.ok) throw new Error("That translation expired.");
      const job = await response.json();
      if (job.progress) $("progress-text").textContent = job.progress;
      if (job.state === "done") {
        setBusy(false);
        renderResult(job.result);
        return;
      }
      if (job.state === "failed") {
        setBusy(false);
        showError(job.error || "The translation failed.");
        return;
      }
      state.polling = setTimeout(tick, 900);
    } catch (error) {
      setBusy(false);
      showError(error.message);
    }
  };
  state.polling = setTimeout(tick, 400);
}

// -- rendering -------------------------------------------------------------

const BADGES = {
  verse: ["verse", "Published verse"],
  neural: ["neural", "Translated"],
  gloss: ["gloss", "Word-by-word"],
  none: ["none", "Not translated"],
};

function renderResult(data) {
  state.last = data;
  $("source-heading").textContent = data.language_name;
  $("out-source").textContent = data.source || "—";

  const readAs = $("read-as");
  if (data.read_as) {
    readAs.textContent = `Read as: ${data.read_as}`;
    show(readAs, true);
  } else {
    show(readAs, false);
  }

  const warning = $("warning");
  if (data.warning) {
    warning.textContent = data.warning;
    show(warning, true);
  } else {
    show(warning, false);
  }

  $("out-english").textContent = data.english || "—";
  const [badgeClass, badgeText] = BADGES[data.engine] || BADGES.none;
  const badge = $("engine-badge");
  badge.className = `badge ${badgeClass}`;
  badge.textContent = badgeText;

  const note = $("engine-note");
  if (data.note) {
    note.textContent = data.note;
    show(note, true);
  } else {
    show(note, false);
  }

  if (data.verse) {
    $("verse-ref").textContent = data.verse.reference;
    $("verse-english").textContent = data.verse.english;
    show($("verse"), true);
  } else {
    show($("verse"), false);
  }

  if (data.near_verse) {
    $("near-ref").textContent = data.near_verse.reference;
    $("near-english").textContent = data.near_verse.english;
    show($("near"), true);
  } else {
    show($("near"), false);
  }

  $("fix-text").value = data.english || "";
  $("fix-status").textContent = "";
  show($("result"), true);
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

// -- text ------------------------------------------------------------------

$("send-text").addEventListener("click", () => {
  const text = $("text-input").value.trim();
  if (!text) {
    showError("Type something to translate.");
    return;
  }
  startJob({
    url: "api/translate",
    options: {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, lang: state.current.code }),
    },
  });
});

$("text-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) $("send-text").click();
});

// -- recording -------------------------------------------------------------

$("file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  state.pickedFile = file || null;
  state.recordedBlob = null;
  $("filename").textContent = file ? `${file.name} (${(file.size / 1048576).toFixed(1)} MB)` : "";
  $("send-file").disabled = !file || !state.current.speech;
});

$("record").addEventListener("click", async () => {
  if (state.recorder && state.recorder.state === "recording") {
    state.recorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.chunks = [];
    state.recorder = new MediaRecorder(stream);
    state.recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) state.chunks.push(event.data);
    });
    state.recorder.addEventListener("stop", () => {
      stream.getTracks().forEach((track) => track.stop());
      state.recordedBlob = new Blob(state.chunks, { type: state.recorder.mimeType || "audio/webm" });
      state.pickedFile = null;
      $("record").classList.remove("on");
      $("record-label").textContent = "Tap to record";
      clearInterval(state.timer);
      $("filename").textContent = `Recording ready (${(state.recordedBlob.size / 1048576).toFixed(1)} MB)`;
      $("send-file").disabled = false;
    });
    state.recorder.start();
    state.startedAt = Date.now();
    $("record").classList.add("on");
    $("record-label").textContent = "Tap to stop";
    state.timer = setInterval(() => {
      const seconds = Math.floor((Date.now() - state.startedAt) / 1000);
      $("timer").textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
    }, 250);
  } catch (error) {
    showError("The microphone was not allowed. Grant Termux microphone permission in Android settings.");
  }
});

$("send-file").addEventListener("click", () => {
  const upload = currentUpload();
  if (!upload) {
    showError("Choose or record something first.");
    return;
  }
  const form = new FormData();
  const name = state.pickedFile ? state.pickedFile.name : "recording.webm";
  form.append("audio", upload, name);
  form.append("lang", state.current.code);
  startJob({ url: "api/audio", options: { method: "POST", body: form } });
});

// -- extras ----------------------------------------------------------------

$("copy").addEventListener("click", async () => {
  const text = $("out-english").textContent;
  try {
    await navigator.clipboard.writeText(text);
    $("copy").textContent = "Copied";
    setTimeout(() => { $("copy").textContent = "Copy"; }, 1400);
  } catch (error) {
    showError("Could not copy. Select the text and copy it by hand.");
  }
});

$("save-fix").addEventListener("click", async () => {
  if (!state.last) return;
  const corrected = $("fix-text").value.trim();
  if (!corrected) return;
  try {
    const response = await fetch("api/correction", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        lang: state.last.language,
        source: state.last.source,
        machine: state.last.english,
        corrected,
      }),
    });
    $("fix-status").textContent = response.ok ? "Saved on this phone." : "Could not save that.";
  } catch (error) {
    $("fix-status").textContent = "Could not save that.";
  }
});

loadHealth();
