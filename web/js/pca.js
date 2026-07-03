// pca.js - real principal-component analysis of the 384-D embeddings so they
// can be projected to 3-D for display. Power iteration on the implicit
// covariance (v <- Sum_i x_i (x_i . v)) with Gram-Schmidt deflation for the top
// k components. Pure JS, no dependencies. Returns the data mean and a k*D basis
// so the SAME projection is applied to the corpus and to a live query point.

function normalise(v) {
  let s = 0; for (let j = 0; j < v.length; j++) s += v[j] * v[j];
  s = Math.sqrt(s) || 1; for (let j = 0; j < v.length; j++) v[j] /= s;
}

export function pca(data, N, D, k = 3, iters = 40) {
  const mean = new Float32Array(D);
  for (let i = 0; i < N; i++) { const b = i * D; for (let j = 0; j < D; j++) mean[j] += data[b + j]; }
  for (let j = 0; j < D; j++) mean[j] /= N;

  const X = new Float32Array(N * D);
  for (let i = 0; i < N; i++) { const b = i * D; for (let j = 0; j < D; j++) X[b + j] = data[b + j] - mean[j]; }

  const comps = [];
  for (let c = 0; c < k; c++) {
    let v = new Float32Array(D);
    for (let j = 0; j < D; j++) v[j] = Math.random() - 0.5;
    normalise(v);
    for (let it = 0; it < iters; it++) {
      const nv = new Float32Array(D);
      for (let i = 0; i < N; i++) {
        const b = i * D; let d = 0;
        for (let j = 0; j < D; j++) d += X[b + j] * v[j];
        for (let j = 0; j < D; j++) nv[j] += X[b + j] * d;
      }
      for (const u of comps) { let d = 0; for (let j = 0; j < D; j++) d += nv[j] * u[j]; for (let j = 0; j < D; j++) nv[j] -= d * u[j]; }
      normalise(nv);
      v = nv;
    }
    comps.push(v);
  }

  const basis = new Float32Array(k * D);
  for (let c = 0; c < k; c++) basis.set(comps[c], c * D);
  return { mean, basis, k, D };
}

// Project the whole corpus -> Float32Array(N*3), auto-scaled into a ~[-1,1] box.
export function projectAll(data, N, D, { mean, basis, k }) {
  const pos = new Float32Array(N * k);
  for (let i = 0; i < N; i++) {
    const b = i * D;
    for (let c = 0; c < k; c++) {
      let d = 0; const bo = c * D;
      for (let j = 0; j < D; j++) d += (data[b + j] - mean[j]) * basis[bo + j];
      pos[i * k + c] = d;
    }
  }
  let max = 1e-6;
  for (let i = 0; i < pos.length; i++) { const a = Math.abs(pos[i]); if (a > max) max = a; }
  const s = 1 / max;
  for (let i = 0; i < pos.length; i++) pos[i] *= s;
  return { pos, scale: s };
}

// Project one query vector with the same basis + mean + scale -> [x,y,z].
export function projectOne(vec, { mean, basis, k }, scale) {
  const out = new Float32Array(k);
  for (let c = 0; c < k; c++) {
    let d = 0; const bo = c * mean.length;
    for (let j = 0; j < mean.length; j++) d += (vec[j] - mean[j]) * basis[bo + j];
    out[c] = d * scale;
  }
  return out;
}
