// knn.js - cosine nearest-neighbour search of a query embedding against the
// corpus. Embeddings are L2-normalised, so cosine similarity is just the dot
// product. O(N*D) per query (~1M ops here), effectively instant. Returns the
// top-k indices with their similarity, plus a kNN emotion vote for a "predicted emotion".

const DIM = 384;

export function nearest(query, data, N, k = 8) {
  const sims = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const b = i * DIM; let d = 0;
    for (let j = 0; j < DIM; j++) d += query[j] * data[b + j];
    sims[i] = d;
  }
  const idx = Array.from({ length: N }, (_, i) => i);
  idx.sort((a, b) => sims[b] - sims[a]);
  const top = idx.slice(0, k);
  return top.map((i) => ({ i, sim: sims[i] }));
}

export function voteEmotion(neighbours, labels, nEmotions = 6) {
  const w = new Float32Array(nEmotions);
  for (const { i, sim } of neighbours) w[labels[i]] += Math.max(0, sim);
  let best = 0, bi = 0, tot = 0;
  for (let e = 0; e < nEmotions; e++) { tot += w[e]; if (w[e] > best) { best = w[e]; bi = e; } }
  return { emotion: bi, confidence: tot > 0 ? best / tot : 0 };
}
