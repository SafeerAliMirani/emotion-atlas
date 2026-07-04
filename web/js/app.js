// app.js - orchestrates the Emotion Atlas: load the real model, fetch real
// tweets, embed them on WebGPU, PCA-project to 3D, render the cloud, and let
// you drop a live-embedded sentence into the space with its real neighbours.
import { Renderer } from "./render-core.js";
import { OrbitCamera } from "./camera.js";
import { fetchEmotionTweets, EMOTIONS } from "./emotion-data.js";
import { loadEmbedder, embed, embedOne } from "./embedder.js";
import { pca, projectAll, projectOne } from "./pca.js";
import { nearest, voteEmotion } from "./knn.js";
import { getCached, putCached } from "./cache.js";

const EMO_HEX = ["#3887e5", "#f1b32b", "#e87ba4", "#e34948", "#9085e9", "#1baf7a"];
const canvas = document.getElementById("gpu");
const TOTAL = 16000;                 // full dair-ai/emotion train split
const state = { pointSize: 2.6, starSize: 13, time: 0 };
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
let renderer, camera, N = 0, ready = false, lastInteract = 0;
let texts = [], labels = null, embeddings = null, positions = null, pcaModel = null, projScale = 1;
let pointData = null;
const visible = [true, true, true, true, true, true];

const $ = (id) => document.getElementById(id);
function markInteract() { lastInteract = performance.now(); }
function setLoading(msg, frac) {
  const t = $("loading-text"); if (t) t.textContent = msg;
  const bar = $("bar"); if (bar && frac != null) bar.style.width = Math.round(frac * 100) + "%";
}
function hideLoading() { const o = $("loading"); if (o) o.classList.add("hidden"); }
function showNoGPU(msg) { const o = $("nogpu"); if (o) o.classList.remove("hidden"); if (msg) { const m = $("nogpu-msg"); if (m) m.textContent = msg; } }
function setStat(id, v) { const el = $(id); if (el) el.textContent = v; }

async function boot() {
  if (!(await Renderer.supported())) { showNoGPU(); return; }
  renderer = new Renderer(canvas);
  try { await renderer.init(); } catch (e) { console.error(e); showNoGPU(e.message); return; }
  camera = new OrbitCamera();
  camera.distance = 3.2; camera.minDist = 0.45; camera.maxDist = 12;
  markInteract();
  setupInput();
  loop();

  try {
    const CACHE_KEY = "dair-ai-emotion|all-MiniLM-L6-v2|" + TOTAL + "|v1";
    setLoading("Loading model, first load only", 0.02);
    const modelP = loadEmbedder((frac) => setLoading("Downloading model on WebGPU", Math.max(0.02, frac)));
    const cached = await getCached(CACHE_KEY);
    if (cached && cached.embeddings && cached.texts) {
      texts = cached.texts; labels = cached.labels; embeddings = cached.embeddings; N = texts.length;
      setLoading("Loading model for live queries", 0.6);
      await modelP;
    } else {
      const data = await fetchEmotionTweets(TOTAL, (f) => setLoading("Fetching " + TOTAL.toLocaleString() + " real tweets", f));
      texts = data.texts; labels = data.labels; N = texts.length;
      await modelP;
      setLoading("Embedding " + N.toLocaleString() + " tweets on WebGPU, one time, then cached", 0);
      embeddings = await embed(texts, (f) => setLoading("Embedding on WebGPU, cached after this", f));
      putCached(CACHE_KEY, { texts, labels, embeddings });
    }
    setLoading("Projecting to 3D (PCA)", 0.94);
    pcaModel = pca(embeddings, N, 384, 3);
    const pr = projectAll(embeddings, N, 384, pcaModel);
    positions = pr.pos; projScale = pr.scale;
    buildPoints();
    buildLegend();
    setStat("stat-points", N.toLocaleString());
    hideLoading();
    ready = true;
    markInteract();
  } catch (e) {
    console.error(e);
    setLoading("Could not load: " + (e.message || e), 0);
  }
}

function buildPoints() {
  pointData = new Float32Array(N * 4);
  for (let i = 0; i < N; i++) {
    pointData[i * 4] = positions[i * 3];
    pointData[i * 4 + 1] = positions[i * 3 + 1];
    pointData[i * 4 + 2] = positions[i * 3 + 2];
    pointData[i * 4 + 3] = labels[i];
  }
  applyFilter();
}

function applyFilter() {
  if (!pointData) return;
  const keep = [];
  for (let i = 0; i < N; i++) if (visible[labels[i]]) keep.push(i);
  const arr = new Float32Array(keep.length * 4);
  keep.forEach((idx, k) => { arr.set(pointData.subarray(idx * 4, idx * 4 + 4), k * 4); });
  renderer.setPoints(arr, keep.length);
}

function buildLegend() {
  const box = $("legend"); if (!box) return;
  const counts = [0, 0, 0, 0, 0, 0];
  for (let i = 0; i < N; i++) counts[labels[i]]++;
  box.innerHTML = "";
  EMOTIONS.forEach((name, e) => {
    const row = document.createElement("button");
    row.className = "legend-row";
    row.innerHTML = '<span class="dot" style="background:' + EMO_HEX[e] + '"></span><span class="lname">' + name + '</span><span class="lcount">' + counts[e].toLocaleString() + '</span>';
    row.onclick = () => { visible[e] = !visible[e]; row.classList.toggle("off", !visible[e]); applyFilter(); markInteract(); };
    box.appendChild(row);
  });
}

async function runQuery(text) {
  if (!embeddings) return;
  markInteract();
  const btn = $("embed-btn"); if (btn) btn.disabled = true;
  try {
    const q = await embedOne(text);
    const nb = nearest(q, embeddings, N, 8);
    const vote = voteEmotion(nb, labels, 6);
    const star = projectOne(q, pcaModel, projScale);
    renderer.setOverlay(Float32Array.of(star[0], star[1], star[2], 6), 1);
    const lines = new Float32Array(nb.length * 6);
    nb.forEach((o, i) => {
      lines[i * 6] = star[0]; lines[i * 6 + 1] = star[1]; lines[i * 6 + 2] = star[2];
      lines[i * 6 + 3] = positions[o.i * 3]; lines[i * 6 + 4] = positions[o.i * 3 + 1]; lines[i * 6 + 5] = positions[o.i * 3 + 2];
    });
    renderer.setLines(lines);
    showResult(text, vote, nb);
  } catch (e) { console.error(e); }
  if (btn) btn.disabled = false;
}

function showResult(text, vote, nb) {
  const box = $("result"); if (!box) return;
  const e = vote.emotion;
  const nn = nb.slice(0, 5).map((o) =>
    '<div class="nn"><span class="nn-dot" style="background:' + EMO_HEX[labels[o.i]] + '"></span><span class="nn-txt">' +
    escapeHtml(texts[o.i]) + '</span><span class="nn-sim">' + o.sim.toFixed(2) + '</span></div>').join("");
  box.innerHTML =
    '<div class="pred">predicted <b style="color:' + EMO_HEX[e] + '">' + EMOTIONS[e] + '</b> · ' + Math.round(vote.confidence * 100) + '%</div>' +
    '<div class="nn-title">nearest real tweets</div>' + nn;
  box.classList.remove("hidden");
}

function escapeHtml(s) { return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

function pick(px, py) {
  if (!positions || !N) return;
  const vp = camera.viewProj(canvas.clientWidth / canvas.clientHeight);
  let best = -1, bestD = 22 * 22;
  for (let i = 0; i < N; i++) {
    if (!visible[labels[i]]) continue;
    const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
    const cx = vp[0] * x + vp[4] * y + vp[8] * z + vp[12];
    const cy = vp[1] * x + vp[5] * y + vp[9] * z + vp[13];
    const cw = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
    if (cw <= 0) continue;
    const sx = (cx / cw * 0.5 + 0.5) * canvas.clientWidth;
    const sy = (1 - (cy / cw * 0.5 + 0.5)) * canvas.clientHeight;
    const d = (sx - px) * (sx - px) + (sy - py) * (sy - py);
    if (d < bestD) { bestD = d; best = i; }
  }
  if (best >= 0) {
    const box = $("result"); if (!box) return;
    box.innerHTML = '<div class="pred">this tweet · <b style="color:' + EMO_HEX[labels[best]] + '">' + EMOTIONS[labels[best]] + '</b></div><div class="nn"><span class="nn-txt">' + escapeHtml(texts[best]) + '</span></div>';
    box.classList.remove("hidden");
  }
}

function setupInput() {
  let dragging = false, moved = 0, lx = 0, ly = 0;
  canvas.addEventListener("pointerdown", (e) => { dragging = true; moved = 0; lx = e.clientX; ly = e.clientY; camera.beginDrag(); markInteract(); canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - lx, dy = e.clientY - ly; lx = e.clientX; ly = e.clientY; moved += Math.abs(dx) + Math.abs(dy);
    camera.rotateByPixels(dx, dy, canvas.clientHeight); markInteract();
  });
  canvas.addEventListener("pointerup", (e) => { dragging = false; camera.endDrag(); markInteract(); if (moved < 5) pick(e.clientX, e.clientY); });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); camera.zoomBy(Math.exp(e.deltaY * 0.001)); markInteract(); }, { passive: false });

  const inp = $("query"), btn = $("embed-btn");
  if (btn) btn.onclick = () => { const v = inp.value.trim(); if (v) runQuery(v); };
  if (inp) inp.addEventListener("keydown", (e) => { if (e.key === "Enter") { const v = inp.value.trim(); if (v) runQuery(v); } });

  const ab = $("about-btn"), about = $("about"), abx = $("about-close");
  if (ab && about) ab.onclick = () => about.classList.toggle("hidden");
  if (abx && about) abx.onclick = () => about.classList.add("hidden");

  window.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement !== inp) { e.preventDefault(); if (inp) inp.focus(); }
    else if (e.key === "Escape") { if (about && !about.classList.contains("hidden")) about.classList.add("hidden"); else if (inp) inp.blur(); }
  });
}

let last = 0;
function loop(now) {
  now = now || 0;
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (!document.hidden) {
    state.time += dt;
    if (ready && !reducedMotion && performance.now() - lastInteract > 4000) camera.azimuth += dt * 0.12;
    camera.update(dt);
    renderer.resize();
    const vp = camera.viewProj(canvas.clientWidth / Math.max(1, canvas.clientHeight));
    renderer.render(vp, state);
  } else { last = now; }
  requestAnimationFrame(loop);
}

boot();
