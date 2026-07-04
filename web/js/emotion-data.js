// emotion-data.js - loads real tweets and emotion labels from the public
// dair-ai/emotion dataset via the Hugging Face datasets-server rows API
// (CORS-enabled JSON, no backend). Each row is { text, label }, where label
// is an index into EMOTIONS. Pages are capped at 100 rows each, so a larger
// corpus is fetched with bounded concurrency (each row keeps its own label,
// so out-of-order page completion is harmless).

const BASE = "https://datasets-server.huggingface.co/rows?dataset=dair-ai/emotion&config=split&split=train";

export const EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"];

export async function fetchEmotionTweets(total = 2400, onProgress) {
  const PAGE = 100;
  const CONC = 6;
  const pages = Math.ceil(total / PAGE);
  const texts = [];
  const labels = [];
  let done = 0;
  let next = 0;

  async function worker() {
    while (true) {
      const p = next++;
      if (p >= pages) return;
      const off = p * PAGE;
      const len = Math.min(PAGE, total - off);
      if (len <= 0) return;
      const url = `${BASE}&offset=${off}&length=${len}`;
      let rows = null;
      for (let a = 0; a < 2 && !rows; a++) {
        try {
          const r = await fetch(url);
          if (r.ok) rows = (await r.json()).rows;
        } catch (e) { /* retry */ }
        if (!rows) await new Promise((res) => setTimeout(res, 400));
      }
      if (rows) {
        for (const it of rows) {
          const row = it.row || it;
          if (row && typeof row.text === "string" && row.label != null) {
            texts.push(row.text);
            labels.push(row.label | 0);
          }
        }
      }
      done += len;
      if (onProgress) onProgress(Math.min(1, done / total));
    }
  }

  await Promise.all(Array.from({ length: CONC }, worker));
  return { texts, labels: Int8Array.from(labels), emotions: EMOTIONS };
}
