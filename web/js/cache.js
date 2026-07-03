// cache.js - persists the embedded corpus in IndexedDB so the full dataset is
// fetched + embedded once and then loads instantly on every later visit. Keyed
// by dataset + model + size, so changing any of them invalidates cleanly.

const DB = "emotion-atlas";
const STORE = "corpus";

function openDB() {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE);
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}

export async function getCached(key) {
  try {
    const db = await openDB();
    return await new Promise((res) => {
      const req = db.transaction(STORE, "readonly").objectStore(STORE).get(key);
      req.onsuccess = () => res(req.result || null);
      req.onerror = () => res(null);
    });
  } catch (e) { return null; }
}

export async function putCached(key, val) {
  try {
    const db = await openDB();
    await new Promise((res) => {
      const req = db.transaction(STORE, "readwrite").objectStore(STORE).put(val, key);
      req.onsuccess = () => res();
      req.onerror = () => res();
    });
  } catch (e) { /* cache is best-effort */ }
}
