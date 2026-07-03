# Deploying Emotion Atlas

Emotion Atlas is a static site with **no build step**. The model (from Hugging
Face) and the tweets (from the HF datasets-server) load live in the browser, so
it runs from any static host over HTTPS. `web/serve.py` is only for local dev.

The app lives in the **`web/`** folder (the publish directory). The included
`netlify.toml` sets this, so you don't configure it by hand.

Visitors need a WebGPU browser (Chrome/Edge 113+, desktop Safari 18+). The first
load downloads the ~4 MB model once (then it's cached); a progress bar covers it.

## Recommended: private GitHub repo → Netlify (auto-deploys on every push)

### 1. Create the repo
- github.com → **New repository** → name `emotion-atlas` → **Private** → **Create repository** (leave it empty).

### 2. Push this folder
Open a terminal in this `EmotionAtlas` folder and run:

```bash
git init -b main
git add .
git commit -m "Emotion Atlas — in-browser WebGPU embedding explorer"
git remote add origin https://github.com/SafeerAliMirani/emotion-atlas.git
git push -u origin main
```

(Prefer clicking? GitHub Desktop → Add local repository → this folder → Publish, keep it private.)

### 3. Connect on Netlify
- app.netlify.com → **Add new site → Import an existing project → GitHub** → pick `emotion-atlas`.
- Settings auto-fill from `netlify.toml`: build command blank, publish directory `web`.
- **Deploy.** Set the site name to `emotion-atlas` for `https://emotion-atlas.netlify.app`.

## Quick alternative: drag & drop
app.netlify.com → **Add new site → Deploy manually** → drag the **`web`** folder (the one with `index.html`) onto the upload area.

## After it's live
Add the URL to `PORTFOLIO.md`, your résumé, and your portfolio site.
