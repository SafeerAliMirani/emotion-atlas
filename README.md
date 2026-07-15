<h1 align="center">Emotion Atlas</h1>
<p align="center"><b>A real language model runs in your browser on WebGPU, reads thousands of real tweets, and turns their feelings into a 3D map you can fly through. Zero install, no backend.</b></p>

<p align="center">
  <a href="https://emotion-atlas-dww.pages.dev"><img src="https://img.shields.io/badge/Live%20Demo-emotion--atlas--dww.pages.dev-1baf7a?style=for-the-badge&logo=cloudflare&logoColor=white" alt="Live Demo" /></a>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/WebGPU-005A9C?style=for-the-badge&logo=webgpu&logoColor=white" alt="WebGPU" />
  <img src="https://img.shields.io/badge/WGSL-1a1a2e?style=for-the-badge" alt="WGSL" />
  <img src="https://img.shields.io/badge/transformers.js-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="transformers.js" />
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge&logo=opensourceinitiative&logoColor=white" alt="MIT License" />
</p>

<p align="center"><b>Live demo:</b> <a href="https://emotion-atlas-dww.pages.dev">emotion-atlas-dww.pages.dev</a> (best in Chrome/Edge 113+ or desktop Safari 18+)</p>

## What it does

Emotion Atlas loads **all-MiniLM-L6-v2**, a real sentence-transformer, directly in your browser and runs it on **WebGPU** via transformers.js. No server ever sees your text. It fetches the full 16,000-tweet train split of the public **dair-ai/emotion** dataset live from the Hugging Face datasets-server, embeds every tweet into a 384-dimensional meaning vector, and projects that space down to 3D with a hand-written PCA. The result is a point cloud, colored by emotion, that you can rotate, zoom, and click through, rendered with hand-written WGSL rather than a charting library.

Type your own sentence and it gets embedded live, on the GPU, placed in the same 3D space by the same projection, and connected to its true nearest neighbors by cosine similarity, with a k-NN vote for its predicted emotion. Nothing here is precomputed, bundled, or synthetic: the model and every tweet load at run time, and the first embedding run happens live in front of you.

It's also a bit of a homecoming for me. My MS thesis was fine-grained emotion detection from text, so this is that problem reborn as a live, GPU-accelerated, browser-native tool.

## How it works

**1. In-browser embedding on WebGPU.** `embedder.js` loads all-MiniLM-L6-v2 through transformers.js with `device: "webgpu"`. The corpus is embedded in batches of 64 texts, each producing a 384-D, L2-normalized vector. A typed query goes through the exact same pipeline, one sentence at a time, so it lands in a genuinely comparable space.

**2. One PCA basis for everything.** `pca.js` runs power iteration on the (implicit) covariance of the 384-D embeddings, with Gram-Schmidt deflation to pull out the top 3 orthogonal components, no external linear-algebra library. The mean and basis are computed once from the corpus and reused for every projection afterward, including a live query, so a typed sentence lands consistently instead of warping the whole map every time you type.

**3. Cosine k-NN retrieval.** Because every embedding is already unit-length, cosine similarity between two vectors is just their dot product. `knn.js` scores a query against all 16,000 corpus vectors (about a million multiply-adds, over in a blink), returns the top-k, and turns those neighbors into a weighted emotion vote for the "predicted emotion" readout.

**4. WGSL instanced points.** `shaders.js` and `render-core.js` hold the whole renderer: a storage buffer of `vec4<f32>` (xyz position + emotion label) is pulled per-instance in the vertex shader to draw screen-space discs, six vertices per point, no vertex buffer needed for the cloud itself. It runs with a real depth buffer and 4x MSAA, and the same pipeline draws the highlighted query star and the neighbor lines. There's no three.js, no D3, no WebGL fallback: it's raw `navigator.gpu`.

**5. Camera and math from scratch.** `camera.js` is an arcball orbit camera with drag momentum and eased zoom; `mat.js` is a small column-major 4x4 matrix library written to match WGSL's `mat4x4<f32>` layout and WebGPU's `[0, 1]` clip-space depth convention (not OpenGL's `[-1, 1]`), which is exactly the kind of detail that silently breaks a renderer if you get it wrong.

## Tech highlights

- **Real client-side inference.** A transformer model runs on WebGPU in the tab, no API call, no server-side GPU.
- **Hand-rolled WebGPU renderer.** Instanced point rendering via storage-buffer vertex pulling in raw WGSL, with a proper depth buffer and 4x MSAA, built without any rendering library.
- **From-scratch dimensionality reduction.** Power-iteration PCA with Gram-Schmidt deflation, implemented in plain JavaScript, sharing one basis between a static corpus and live, one-off queries.
- **Real retrieval, not a lookup table.** Cosine k-NN over live embeddings drives both the neighbor lines you see and the emotion prediction.
- **Resilient live data loading.** The corpus is paged from the HF datasets-server (100 rows per page) with bounded concurrency and retries, overlapped with the model download so the first visit doesn't feel like two sequential waits.
- **IndexedDB caching.** The embedded corpus is cached client-side after the first run, keyed by dataset, model, and size, so a second visit skips straight to rendering.

## Run it locally

No build step, no bundler, no package manager. It's plain ES modules loaded straight by the browser.

```bash
git clone https://github.com/SafeerAliMirani/emotion-atlas.git
cd emotion-atlas/web
python serve.py
# open http://localhost:8080
```

`serve.py` is a tiny stdlib-only server that disables caching, so every refresh loads the latest files instead of a stale cached module. Any static file server works too, just serve the `web/` folder over HTTP (not `file://`, since WebGPU and module imports need a real origin).

You'll need a WebGPU-capable browser (Chrome or Edge 113+, or desktop Safari 18+) and an internet connection, since the model and the tweets both load live from Hugging Face. The very first load takes up to a minute while the ~4 MB model downloads and the corpus gets embedded, with a progress bar the whole time. After that it's cached and loads instantly.

## Data & credits

| What | Source |
|---|---|
| Model | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (ONNX port: [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2)), Apache-2.0 |
| Inference runtime | [transformers.js](https://github.com/huggingface/transformers.js) |
| Data | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion), fetched live via the Hugging Face datasets-server |

Both the model and the dataset are the property of their respective authors and are loaded at run time, not redistributed in this repository.

## Honest by design

This app tells you its own limits, right in the "about" panel: the 3D positions are a **PCA shadow** of a 384-dimensional space, so some structure is unavoidably lost in the projection, and the "predicted emotion" is a **nearest-neighbor vote**, not a separately trained classifier. It's showing you the geometry of what the model actually learned, not claiming a benchmark score. I'd rather ship a tool that's upfront about what it is than one that oversells itself.

## License

[MIT](LICENSE) © 2026 Dr. Safeer Ali Mirani. The all-MiniLM-L6-v2 model (Apache-2.0) and the dair-ai/emotion dataset remain the property of their respective authors and are not redistributed here.

---

<p align="center">Built by <b>Dr. Safeer Ali Mirani</b>, GPU / XR / real-time visualisation engineer and computational neuroscientist (PhD).<br />
<a href="mailto:safeer.ali.mirani@gmail.com">safeer.ali.mirani@gmail.com</a> · <a href="https://safeeralimirani.pages.dev">Portfolio</a> · <a href="https://github.com/SafeerAliMirani">GitHub</a> · <a href="https://www.linkedin.com/in/safeeralimirani">LinkedIn</a></p>
