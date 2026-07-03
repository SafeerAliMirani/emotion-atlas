# Emotion Atlas

**A real language model runs in your browser on WebGPU, reads thousands of real tweets, and turns their feelings into a 3D map you can explore — zero install, no backend.**

_By **Dr. Safeer Ali Mirani** — GPU / XR / real-time visualisation engineer and computational neuroscientist (PhD)._

**Live demo → [emotion-atlas.netlify.app](https://emotion-atlas.netlify.app)** — best in a WebGPU browser (Chrome/Edge 113+ or desktop Safari 18+). The first visit embeds all 16k tweets live (~1–2 min), then caches — later visits are instant.

Emotion Atlas loads the **all-MiniLM-L6-v2** sentence-transformer directly in the browser (via transformers.js on WebGPU), fetches ~2,400 real tweets from the public **dair-ai/emotion** dataset live, and embeds every one into a 384-dimensional meaning vector. Those vectors are projected to 3D with principal component analysis and drawn as a point cloud — coloured by emotion — using hand-written WGSL. Type your own sentence and it's embedded live and dropped into the same space, with lines to its true nearest neighbours and a k-NN emotion prediction. Nothing is precomputed, bundled, or synthetic: the model and every tweet load at run time.

It's also a homecoming: my MS thesis was fine-grained emotion detection from text, so this is that idea reborn as a live, GPU-accelerated, browser-native tool.

## Features

- **Real ML, in your browser** — all-MiniLM-L6-v2 runs client-side on WebGPU (transformers.js). The ~4 MB model downloads once, then the browser caches it.
- **Real data, fetched live** — ~2,400 real emotion-labelled tweets streamed from the Hugging Face datasets-server (dair-ai/emotion), paged with bounded concurrency.
- **Hand-written WebGPU rendering** — the point cloud is instanced screen-space discs drawn from a storage buffer in raw WGSL, with a depth buffer and 4× MSAA. No three.js, no plotting library.
- **Live sentence embedding** — type anything; it's embedded on the GPU, placed by the same PCA projection, and linked to its real nearest neighbours (cosine similarity) with a k-NN emotion vote.
- **Explore** — arcball drag to rotate, scroll to zoom, click any point to read its actual tweet, filter emotions from the legend, gentle idle auto-rotate.
- **Honest provenance panel** — what the model is, where the data comes from, and the limits of the method.

## Real, public data — and honest about the method

Nothing here is synthetic or precomputed. At run time the browser talks to two public sources:

| Source | What | How |
|---|---|---|
| **Hugging Face model hub** | the all-MiniLM-L6-v2 sentence-transformer | loaded and run in-browser by transformers.js on the WebGPU backend |
| **dair-ai/emotion** (via HF datasets-server) | ~2,400 real tweets + their emotion labels | fetched live as JSON, paged 100 rows at a time |

Two honest caveats, stated in the app itself: the 3D positions are a **PCA shadow** of 384-D space, so some structure is unavoidably lost; and the "predicted emotion" is a **nearest-neighbour vote**, not a separately trained classifier — it's showing you the geometry, not claiming state-of-the-art accuracy.

## Prior art & what's different

Latent-space explorers exist — the best known is TensorFlow's Embedding Projector — and so do in-browser ML demos (transformers.js ships many). What's uncommon is the combination: the embeddings are computed **live in the browser on WebGPU** from a **real labelled corpus** (not precomputed and uploaded), the cloud is rendered in **hand-written WGSL** (not three.js or a charting library), and you can **embed your own sentence live** and see its true neighbours. The differentiator is the same as the rest of my portfolio: it's built against the raw GPU API, and every byte of model and data is real and loaded at run time.

## Run it

```bash
cd web
python serve.py        # stdlib-only, no-cache static server
# open http://localhost:8080
```

Requirements: a WebGPU browser (Chrome/Edge 113+ or Safari 18+) and an internet connection (the model and tweets load live). The first load runs for up to a minute while the model downloads and the corpus is embedded, with a progress bar throughout; it's cached afterwards.

## Architecture

Plain ES modules, no bundler, no build step.

| Module | Role |
|---|---|
| `web/index.html` | shell: HUD, control panel, loading / about / no-WebGPU overlays |
| `web/js/app.js` | orchestrator: load, fetch, embed, project, render loop, input, query, picking |
| `web/js/embedder.js` | transformers.js wrapper: load MiniLM on WebGPU, batch-embed the corpus, embed live queries |
| `web/js/emotion-data.js` | dair-ai/emotion loader via the HF datasets-server (paged, bounded concurrency) |
| `web/js/pca.js` | power-iteration PCA (top-3 components) + project corpus and query with one basis |
| `web/js/knn.js` | cosine nearest-neighbour search + k-NN emotion vote |
| `web/js/shaders.js` | WGSL: instanced emotion-coloured points and neighbour lines |
| `web/js/render-core.js` | WebGPU device, pipelines, depth + 4× MSAA, per-frame encoding |
| `web/js/camera.js` | arcball orbit camera with momentum and eased zoom |
| `web/js/mat.js` | column-major 4×4 matrix / vec3 math, WebGPU [0,1] depth convention |
| `web/serve.py` | no-cache static dev server (port 8080) |

```
HF model hub                 dair-ai/emotion (datasets-server)
     │                                 │
 embedder.js  ◄── WebGPU          emotion-data.js
 (MiniLM, 384-D)                       │
     └───────────────┬─────────────────┘
                     ▼
                  pca.js  (384-D → 3-D)
                     ▼
   render-core.js ◄── shaders.js (WGSL)      knn.js (live query neighbours)
   instanced points · neighbour lines           ▲
                     ▲                           │
    camera.js ─► app.js ◄───────────────────────┘
```

## Tech highlights

- **Real in-browser inference** — a sentence-transformer runs client-side on WebGPU; corpus embedding is batched, and query embedding is live and interactive.
- **Hand-rolled GPU point cloud** — instanced discs via storage-buffer vertex pulling in WGSL, depth buffer + 4× MSAA, arcball camera and matrix math to the WebGPU depth convention — no three.js, no plotting lib.
- **Real projection maths** — power-iteration PCA with Gram-Schmidt deflation, one basis shared by the corpus and by live queries so a typed sentence lands consistently.
- **Real retrieval** — cosine k-NN over the live embeddings drives both the neighbour lines and the emotion prediction.
- **Resilient live data** — the corpus is paged from the datasets-server with bounded concurrency and retries, overlapped with the model download to shorten first load.

## Data sources & credits

- **Model** — [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), ONNX port [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2), run with [transformers.js](https://github.com/huggingface/transformers.js). Apache-2.0.
- **Data** — [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion), via the Hugging Face datasets-server.

## Author

**Dr. Safeer Ali Mirani** — GPU / XR / real-time visualisation engineer and computational neuroscientist (PhD).
[safeer.ali.mirani@gmail.com](mailto:safeer.ali.mirani@gmail.com) · [Portfolio](https://safeeralimirani.netlify.app) · [GitHub](https://github.com/SafeerAliMirani) · [LinkedIn](https://www.linkedin.com/in/safeeralimirani)

## License

[MIT](LICENSE) © 2026 Dr. Safeer Ali Mirani. The model (Apache-2.0) and the dair-ai/emotion dataset are the property of their respective authors.
