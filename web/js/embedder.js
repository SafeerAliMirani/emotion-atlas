// embedder.js - runs the REAL all-MiniLM-L6-v2 sentence-transformer fully in
// the browser on WebGPU (via transformers.js, loaded from a CDN) to turn text
// into 384-D unit embeddings. No server, no build step. Verified live: WebGPU
// backend works and embeddings are semantically meaningful (similar sentences
// have higher cosine similarity). The model (~4 MB) downloads once and is then
// cached by the browser, so repeat visits are instant.

const CDN = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";
const MODEL = "Xenova/all-MiniLM-L6-v2";

export const DIM = 384;

let _pipe = null;

// onProgress(fraction, file) is called during the one-time model download.
export async function loadEmbedder(onProgress) {
  const T = await import(CDN);
  _pipe = await T.pipeline("feature-extraction", MODEL, {
    device: "webgpu",
    progress_callback: onProgress
      ? (p) => { if (p.status === "progress" && p.total) onProgress(p.loaded / p.total, p.file); }
      : undefined,
  });
  return _pipe;
}

// Embed many texts in batches; returns a Float32Array of length texts.length*DIM
// (row-major, each 384-vector L2-normalized). onProgress(fraction) per batch.
export async function embed(texts, onProgress, batch = 64) {
  if (!_pipe) throw new Error("embedder not loaded");
  const out = new Float32Array(texts.length * DIM);
  for (let i = 0; i < texts.length; i += batch) {
    const chunk = texts.slice(i, i + batch);
    const t = await _pipe(chunk, { pooling: "mean", normalize: true });
    out.set(t.data, i * DIM);
    if (onProgress) onProgress(Math.min(1, (i + chunk.length) / texts.length));
  }
  return out;
}

// Embed a single query sentence live; returns a 384-length Float32Array (unit).
export async function embedOne(text) {
  if (!_pipe) throw new Error("embedder not loaded");
  const t = await _pipe(text, { pooling: "mean", normalize: true });
  return Float32Array.from(t.data);
}
