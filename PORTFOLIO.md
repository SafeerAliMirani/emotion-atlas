# Emotion Atlas — portfolio & CV framing

**Dr. Safeer Ali Mirani** · GPU / XR / real-time visualisation engineer · computational neuroscientist (PhD)
[safeer.ali.mirani@gmail.com](mailto:safeer.ali.mirani@gmail.com) · [Portfolio](https://safeeralimirani.netlify.app) · [GitHub](https://github.com/SafeerAliMirani) · [LinkedIn](https://www.linkedin.com/in/safeeralimirani)

Reusable copy for a CV, portfolio site, or LinkedIn. All claims are accurate and verifiable.

## Résumé bullets

- Built **Emotion Atlas**, a zero-install web app that runs a real sentence-transformer (**all-MiniLM-L6-v2**) fully **in the browser on WebGPU**, embeds ~2,400 live-fetched real tweets into 384-D space, projects them to 3D with hand-written PCA, and renders the cloud as an instanced point system in **hand-written WGSL** — no three.js, no backend, no build step.
- Added **live interactive retrieval**: a user-typed sentence is embedded on the GPU in real time, placed by the same PCA basis, and linked to its true cosine nearest neighbours with a k-NN emotion prediction — real inference and real retrieval, entirely client-side.

## Portfolio blurb (2–3 sentences)

Emotion Atlas turns a language model's sense of feeling into something you can fly through: a real sentence-transformer runs in your browser on WebGPU, reads thousands of real emotion-labelled tweets, and lays them out in 3D by meaning, coloured by emotion. Type your own sentence and it's embedded live and dropped into the map beside its nearest neighbours. It pairs real in-browser machine learning with a hand-written WebGPU renderer, and is honest about its method — a PCA shadow of high-dimensional space and a nearest-neighbour vote, not a black box.

## Interview talking points (the non-obvious engineering)

1. **Real inference on the client.** A sentence-transformer runs on WebGPU via transformers.js; the corpus is embedded in batches and a typed query is embedded live, all with no server.
2. **Hand-written GPU rendering, not a chart library.** The point cloud is instanced screen-space discs pulled from a storage buffer in WGSL, with a real depth buffer and 4× MSAA — the same raw-WebGPU approach as my earthquake globe, applied to ML data.
3. **One projection basis for everything.** Power-iteration PCA (with Gram-Schmidt deflation) yields a basis shared by the corpus and by live queries, so a typed sentence lands consistently rather than being re-fit each time.
4. **Real retrieval.** Cosine k-NN over the live embeddings drives both the neighbour lines and the predicted emotion; because embeddings are L2-normalised, similarity is a single dot product.
5. **Honest by design.** The app states its own limits — positions are a PCA shadow, the prediction is a neighbour vote — which is the right posture for a scientific-visualisation tool.

## What to emphasise, by role

**Applied ML / ML engineering** — lead with real client-side inference: loading and running a transformer on WebGPU, batching, live query embedding, k-NN retrieval, and honest evaluation framing.

**Graphics / WebGPU / rendering** — lead with the hand-written WGSL point system, storage-buffer instancing, depth/MSAA, and the arcball camera and matrix maths to the WebGPU depth convention.

**General frontend / full-stack** — lead with the product: a zero-install, no-backend tool that streams a real model and real data at run time and stays responsive while embedding thousands of items.

## One-line version

*Real sentence-transformer running in your browser on WebGPU, mapping thousands of real tweets into an explorable 3D emotion space — hand-written WGSL, live query retrieval, zero backend. — Dr. Safeer Ali Mirani*
