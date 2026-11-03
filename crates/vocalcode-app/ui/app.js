"use strict";

const { invoke } = window.__TAURI__.core;
const $ = (id) => document.getElementById(id);

const RATE = 24000;
const MAX_S = 30;
const STEP = 0.1;              // une barre = 100 ms
const BARS = MAX_S / STEP;
const css = getComputedStyle(document.documentElement);
const C = {
  ink: css.getPropertyValue("--ink").trim(),
  pink: css.getPropertyValue("--pink").trim(),
  line: css.getPropertyValue("--line").trim(),
};

const fmt = (s) => {
  s = Math.max(0, Math.floor(s + 1e-6));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};
const mm = (v) => v.toFixed(2).replace(".", ",");

// ---------- la bande de 30 s ----------

class Band {
  constructor(canvas, onSeek) {
    this.cv = canvas;
    this.env = new Float32Array(BARS);
    this.duration = 0;
    this.head = null;
    new ResizeObserver(() => this.draw()).observe(canvas);
    canvas.addEventListener("pointerdown", (e) => {
      if (!this.duration || !onSeek) return;
      const r = canvas.getBoundingClientRect();
      const t = ((e.clientX - r.left) / r.width) * MAX_S;
      onSeek(Math.min(t, this.duration));
    });
  }
  set(env, duration) {
    this.env = Float32Array.from(env);
    this.duration = duration;
    this.head = null;
    this.draw();
  }
  draw() {
    const cv = this.cv, dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const slot = w / BARS, mid = h / 2;
    const filled = Math.ceil(this.duration / STEP - 1e-6);

    // capacité restante : un point toutes les 0,5 s, un repère plus haut toutes les 10 s
    g.fillStyle = C.line;
    for (let i = Math.ceil(filled / 5) * 5; i <= BARS; i += 5) {
      const x = Math.min(w - 2, (i / BARS) * w);
      const tall = i % 100 === 0 && i > 0 && i < BARS;
      g.fillRect(x - 1, tall ? mid - 9 : mid - 1.5, 2, tall ? 18 : 3);
    }

    // barres alignées sur les pixels physiques : on regroupe les tranches si la bande est étroite
    const group = Math.max(1, Math.ceil(3 / (slot * dpr)));
    const px = 1 / dpr;
    for (let i = 0; i < filled; i += group) {
      let v = 0;
      for (let j = i; j < Math.min(i + group, filled); j++) v = Math.max(v, this.env[j] || 0);
      const x0 = Math.round((i * slot) * dpr) * px;
      const x1 = Math.round((Math.min(i + group, BARS) * slot) * dpr) * px;
      const bwPx = Math.max(px, Math.round((x1 - x0) * 0.6 * dpr) * px);
      const bh = Math.round(Math.max(2, (0.06 + 0.94 * Math.sqrt(v)) * (h - 14)) * dpr) * px;
      g.fillStyle = this.head !== null && i * STEP < this.head ? C.ink : C.pink;
      g.fillRect(x0, Math.round((mid - bh / 2) * dpr) * px, bwPx, bh);
    }
  }
}

// ---------- lecteur ----------

class Player {
  constructor(band, button, clock) {
    this.band = band;
    this.button = button;
    this.clock = clock;
    this.audio = new Audio();
    this.url = null;
    this.audio.addEventListener("ended", () => this.stopped());
    button.addEventListener("click", () => this.toggle());
  }
  load(bytes) {
    if (this.url) URL.revokeObjectURL(this.url);
    this.url = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
    this.audio.src = this.url;
    this.button.disabled = false;
  }
  get ready() { return !this.button.disabled; }
  toggle() { this.audio.paused ? this.play() : this.pause(); }
  play() {
    if (!this.ready) return;
    this.audio.play();
    this.button.classList.add("playing");
    this.button.setAttribute("aria-label", "Pause");
    const tick = () => {
      if (this.audio.paused) return;
      this.band.head = this.audio.currentTime;
      this.band.draw();
      this.clock.textContent = fmt(this.audio.currentTime);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  pause() {
    this.audio.pause();
    this.button.classList.remove("playing");
    this.button.setAttribute("aria-label", "Écouter");
  }
  seek(t) {
    this.audio.currentTime = t;
    this.band.head = t;
    this.band.draw();
    this.clock.textContent = fmt(t);
    if (this.audio.paused) this.play();
  }
  stopped() {
    this.pause();
    this.band.head = null;
    this.band.draw();
    this.clock.textContent = fmt(this.band.duration);
  }
  reset() {
    this.pause();
    this.button.disabled = true;
  }
}

function setStatus(el, text, error = false) {
  el.textContent = text;
  el.classList.toggle("error", error);
}

const errText = (e) => (typeof e === "string" ? e : e?.message || String(e));

// ---------- modes ----------

function setMode(mode) {
  document.body.dataset.mode = mode;
  for (const b of document.querySelectorAll(".modes button")) {
    const on = b.dataset.mode === mode;
    b.setAttribute("aria-selected", on);
    b.tabIndex = on ? 0 : -1;
  }
  $("create").hidden = mode !== "create";
  $("read").hidden = mode !== "read";
  createPlayer.pause();
  readPlayer.pause();
  requestAnimationFrame(() => { createBand.draw(); readBand.draw(); });
}

document.querySelectorAll(".modes button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
document.querySelector(".modes").addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
    const next = document.body.dataset.mode === "create" ? "read" : "create";
    setMode(next);
    $(next === "create" ? "tab-create" : "tab-read").focus();
  }
});

// ---------- Créer ----------

const createBand = new Band($("c-band"), (t) => createPlayer.seek(t));
const createPlayer = new Player(createBand, $("c-play"), $("c-time"));
const cStatus = $("c-status");
let sizeCm = 8;
let gridN = 0;
let previewUrl = null;

function updateSizes() {
  for (const b of document.querySelectorAll(".sizes button")) {
    const cm = Number(b.dataset.cm);
    b.setAttribute("aria-checked", cm === sizeCm);
    b.dataset.tip = gridN ? `Modules de ${mm((cm * 10) / (gridN + 8))} mm` : "Largeur imprimée, marge comprise";
  }
  if (gridN) {
    const m = (sizeCm * 10) / (gridN + 8);
    setStatus(cStatus, m < 0.25 ? `À ${sizeCm} cm, les modules font ${mm(m)} mm : une imprimante fine est nécessaire.` : cStatus.dataset.keep || "");
  }
}
document.querySelectorAll(".sizes button").forEach((b) =>
  b.addEventListener("click", () => { sizeCm = Number(b.dataset.cm); updateSizes(); }));
updateSizes();

function setBusy(busy) {
  $("rec").disabled = busy;
  $("c-file").disabled = busy;
  document.body.classList.toggle("busy", busy);
}

async function encode(bytes, headers) {
  setBusy(true);
  createPlayer.reset();
  setStatus(cStatus, "Création du code…");
  try {
    const info = await invoke("encode_audio", bytes, { headers });
    gridN = info.n;
    const png = await invoke("take_preview");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(new Blob([png], { type: "image/png" }));
    const img = $("code");
    img.src = previewUrl;
    img.hidden = false;
    img.classList.remove("fresh"); void img.offsetWidth; img.classList.add("fresh");
    $("code-empty").hidden = true;
    createPlayer.load(await invoke("take_audio"));
    createBand.set(info.envelope, info.seconds);
    $("c-time").textContent = fmt(info.seconds);
    $("save-png").disabled = false;
    $("print").disabled = false;
    cStatus.dataset.keep = info.truncated ? "Coupé à 30 s." : "";
    setStatus(cStatus, cStatus.dataset.keep);
    updateSizes();
  } catch (e) {
    setStatus(cStatus, errText(e), true);
  } finally {
    setBusy(false);
  }
}

// enregistrement au micro
let rec = null;

async function startRec() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch {
    setStatus(cStatus, "Micro inaccessible. Vérifie qu'un micro est branché et autorisé.", true);
    return;
  }
  let ctx;
  try { ctx = new AudioContext({ sampleRate: RATE }); } catch { ctx = new AudioContext(); }
  const src = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(2048, 1, 1);
  const ratio = ctx.sampleRate / RATE;
  const env = new Float32Array(BARS);
  const chunks = [];
  let total = 0, acc = 0, accN = 0;
  rec = { stream, ctx, src, proc, chunks };

  createPlayer.reset();
  createBand.set(env, 0);
  setStatus(cStatus, "");
  document.body.classList.add("recording");
  $("rec").setAttribute("aria-label", "Arrêter");

  proc.onaudioprocess = (e) => {
    if (!rec) return;
    let d = e.inputBuffer.getChannelData(0);
    if (ratio !== 1) {  // rééchantillonnage linéaire si le contexte n'est pas à 24 kHz
      const out = new Float32Array(Math.floor(d.length / ratio));
      for (let i = 0; i < out.length; i++) {
        const p = i * ratio, k = Math.floor(p), f = p - k;
        out[i] = d[k] * (1 - f) + (d[Math.min(k + 1, d.length - 1)] * f);
      }
      d = out;
    }
    const room = MAX_S * RATE - total;
    if (d.length > room) d = d.subarray(0, room);
    chunks.push(new Float32Array(d));
    for (const v of d) {
      acc += v * v; accN++;
      if (accN === RATE * STEP) {
        const i = Math.floor(total / (RATE * STEP));
        if (i < BARS) env[i] = Math.sqrt(acc / accN);
        acc = 0; accN = 0;
      }
      total++;
    }
    const peak = Math.max(1e-3, ...env);
    createBand.env = env.map((v) => v / peak);
    createBand.duration = total / RATE;
    createBand.draw();
    $("c-time").textContent = fmt(total / RATE);
    if (total >= MAX_S * RATE) stopRec();
  };
  src.connect(proc);
  proc.connect(ctx.destination);
}

function stopRec() {
  if (!rec) return;
  const { stream, ctx, src, proc, chunks } = rec;
  rec = null;
  proc.disconnect(); src.disconnect();
  stream.getTracks().forEach((t) => t.stop());
  ctx.close();
  document.body.classList.remove("recording");
  $("rec").setAttribute("aria-label", "Enregistrer");
  const n = chunks.reduce((a, c) => a + c.length, 0);
  const pcm = new Float32Array(n);
  let o = 0;
  for (const c of chunks) { pcm.set(c, o); o += c.length; }
  encode(new Uint8Array(pcm.buffer), { "x-kind": "pcm" });
}

$("rec").addEventListener("click", () => (rec ? stopRec() : startRec()));

async function encodeFile(file) {
  if (!file) return;
  const ext = (file.name.split(".").pop() || "").toLowerCase();
  const bytes = new Uint8Array(await file.arrayBuffer());
  encode(bytes, { "x-kind": "file", "x-ext": ext });
}
$("c-file").addEventListener("change", (e) => { encodeFile(e.target.files[0]); e.target.value = ""; });

$("save-png").addEventListener("click", async () => {
  try {
    const path = await invoke("plugin:dialog|save", {
      options: { defaultPath: "message.png", filters: [{ name: "Image PNG", extensions: ["png"] }] },
    });
    if (!path) return;
    await invoke("save_png", { path, widthCm: sizeCm });
    setStatus(cStatus, "Image enregistrée.");
  } catch (e) {
    setStatus(cStatus, errText(e), true);
  }
});

function printCode() {
  if (!previewUrl) return;
  const img = $("print-img");
  img.src = previewUrl;
  img.style.width = `${sizeCm}cm`;
  img.onload = () => window.print();
}
$("print").addEventListener("click", printCode);

// ---------- Lire ----------

const readBand = new Band($("r-band"), (t) => readPlayer.seek(t));
const readPlayer = new Player(readBand, $("r-play"), $("r-time"));
const rStatus = $("r-status");
let photoUrl = null;

async function decodeFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus(rStatus, "Ce fichier n'est pas une image.", true);
    return;
  }
  const bytes = new Uint8Array(await file.arrayBuffer());
  if (photoUrl) URL.revokeObjectURL(photoUrl);
  photoUrl = URL.createObjectURL(file);
  $("photo-img").src = photoUrl;
  $("photo").hidden = false;
  $("drop-empty").hidden = true;
  $("read-drop").classList.add("has-photo", "busy");
  $("read-drop").dataset.tip = "Clique ou glisse une autre photo pour la lire";
  $("r-again").hidden = false;
  $("outline").setAttribute("points", "");
  readPlayer.reset();
  readBand.set([], 0);
  $("r-time").textContent = "0:00";
  $("r-total").textContent = "/0:00";
  $("save-wav").disabled = true;
  setStatus(rStatus, "Lecture du code…");
  try {
    const info = await invoke("decode_photo", bytes);
    const svg = $("photo-svg");
    svg.setAttribute("viewBox", `0 0 ${info.width} ${info.height}`);
    const poly = $("outline");
    poly.setAttribute("points", info.outline.map((p) => p.join(",")).join(" "));
    poly.classList.remove("drawn"); void poly.getBBox(); poly.classList.add("drawn");
    readPlayer.load(await invoke("take_audio"));
    readBand.set(info.envelope, info.seconds);
    $("r-total").textContent = `/${fmt(info.seconds)}`;
    $("save-wav").disabled = false;
    setStatus(rStatus, info.usage > 0.75 ? "Lu de justesse : une photo plus nette serait plus sûre." : "");
    readPlayer.play();
  } catch (e) {
    setStatus(rStatus, errText(e), true);
  } finally {
    $("read-drop").classList.remove("busy");
  }
}

$("r-file").addEventListener("change", (e) => { decodeFile(e.target.files[0]); e.target.value = ""; });

$("r-again").addEventListener("click", () => $("r-file").click());

$("save-wav").addEventListener("click", async () => {
  try {
    const path = await invoke("plugin:dialog|save", {
      options: { defaultPath: "message.wav", filters: [{ name: "Son WAV", extensions: ["wav"] }] },
    });
    if (!path) return;
    await invoke("save_wav", { path });
    setStatus(rStatus, "Son enregistré.");
  } catch (e) {
    setStatus(rStatus, errText(e), true);
  }
});

document.addEventListener("paste", (e) => {
  if (document.body.dataset.mode !== "read") return;
  const item = [...e.clipboardData.items].find((i) => i.type.startsWith("image/"));
  if (item) decodeFile(item.getAsFile());
});

// ---------- glisser-déposer (n'importe où dans la fenêtre, selon le mode) ----------

const dropTarget = () => (document.body.dataset.mode === "create" ? $("create-drop") : $("read-drop"));
let dragDepth = 0;
window.addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; dropTarget().classList.add("over"); });
window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; dropTarget().classList.remove("over"); } });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  dropTarget().classList.remove("over");
  const file = e.dataTransfer.files[0];
  if (!file) return;
  if (file.type.startsWith("image/")) {
    setMode("read");
    decodeFile(file);
  } else if (document.body.dataset.mode === "create" && !rec) {
    encodeFile(file);
  } else {
    setMode("create");
    encodeFile(file);
  }
});

// ---------- clavier ----------

document.addEventListener("keydown", (e) => {
  const onControl = e.target.closest("button, input, label");
  const mode = document.body.dataset.mode;
  if (e.key === " " && !onControl) {
    e.preventDefault();
    (mode === "create" ? createPlayer : readPlayer).toggle();
  } else if ((e.key === "r" || e.key === "R") && mode === "create" && !e.ctrlKey && !e.metaKey && !onControl) {
    rec ? stopRec() : startRec();
  } else if (e.key === "p" && (e.ctrlKey || e.metaKey) && mode === "create") {
    e.preventDefault();
    printCode();
  }
});

// ---------- infobulles ----------

const tip = $("tip");
let tipTimer = null;
function showTip(el) {
  const text = el.dataset.tip;
  if (!text) return;
  tip.textContent = text;
  tip.hidden = false;
  const r = el.getBoundingClientRect(), t = tip.getBoundingClientRect();
  let top = r.top - t.height - 8;
  if (top < 8) top = r.bottom + 8;
  const left = Math.min(Math.max(8, r.left + r.width / 2 - t.width / 2), innerWidth - t.width - 8);
  tip.style.top = `${top}px`;
  tip.style.left = `${left}px`;
}
function hideTip() { clearTimeout(tipTimer); tip.hidden = true; }
document.addEventListener("pointerover", (e) => {
  const el = e.target.closest("[data-tip]");
  hideTip();
  if (el) tipTimer = setTimeout(() => showTip(el), 450);
});
document.addEventListener("pointerdown", hideTip);
document.addEventListener("focusin", (e) => {
  const el = e.target.closest("[data-tip]");
  hideTip();
  if (el && e.target.matches(":focus-visible")) showTip(el);
});
document.addEventListener("focusout", hideTip);

// ---------- modèle audio ----------

(async function watchModel() {
  const el = $("model");
  for (;;) {
    let s;
    try { s = await invoke("model_status"); } catch { s = "loading"; }
    if (s === "ready") { el.hidden = true; return; }
    el.hidden = false;
    if (s !== "loading") { el.textContent = s; el.dataset.tip = ""; return; }
    await new Promise((r) => setTimeout(r, 400));
  }
})();

createBand.draw();
readBand.draw();

// fichier passé à l'exécutable (« Ouvrir avec », glisser sur l'icône)
(async function openInitial() {
  try {
    const name = await invoke("initial_file_name");
    if (!name) return;
    const bytes = await invoke("initial_file");
    const ext = name.split(".").pop().toLowerCase();
    const image = ["jpg", "jpeg", "png", "webp"].includes(ext);
    const type = image ? `image/${ext === "jpg" ? "jpeg" : ext}` : "audio/*";
    const file = new File([bytes], name, { type });
    if (image) { setMode("read"); decodeFile(file); } else { encodeFile(file); }
  } catch (e) { /* aucun fichier */ }
})();
