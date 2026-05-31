import express from "express";
import { createServer } from "http";
import { WebSocketServer } from "ws";
import { watch } from "chokidar";
import { readFileSync, existsSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const SOURCES = {
  accounting: resolve(__dirname, "../outputs/accounting.results.jsonl"),
  accounting_tuned: resolve(__dirname, "../outputs/accounting.results.tuned.jsonl"),
  chess: resolve(__dirname, "../outputs/chess.results.jsonl"),
  chess_tuned: resolve(__dirname, "../outputs/chess.results.tuned.jsonl")
};
const PORT = process.env.PORT || 3000;

// ── Express + HTTP ──────────────────────────────────────────────────────
const app = express();
app.use(express.static(resolve(__dirname, "public")));
const server = createServer(app);

// ── WebSocket ───────────────────────────────────────────────────────────
const wss = new WebSocketServer({ server });

/** Parse the JSONL file and return an array of objects */
function readResults(filePath) {
  if (!existsSync(filePath)) return [];
  const raw = readFileSync(filePath, "utf-8");
  return raw
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line, idx) => {
      try {
        return JSON.parse(line);
      } catch {
        console.warn(`Skipping malformed line ${idx + 1} in ${filePath}`);
        return null;
      }
    })
    .filter(Boolean);
}

/** Broadcast a message to every connected client */
function broadcast(data) {
  const payload = JSON.stringify(data);
  for (const client of wss.clients) {
    if (client.readyState === 1 /* OPEN */) {
      client.send(payload);
    }
  }
}

// Track how many lines we've already sent so we only push deltas
const knownLineCounts = {
  accounting: 0,
  accounting_tuned: 0,
  chess: 0,
  chess_tuned: 0
};

// On new connection, send the full snapshot for each source separately
wss.on("connection", (ws) => {
  console.log("Client connected");
  for (const [key, filePath] of Object.entries(SOURCES)) {
    const results = readResults(filePath);
    knownLineCounts[key] = results.length;
    ws.send(JSON.stringify({ type: "snapshot", source: key, data: results }));
  }
});

// ── File watcher ────────────────────────────────────────────────────────
const watcher = watch(Object.values(SOURCES), {
  persistent: true,
  usePolling: true,       // reliable for files written by other processes
  interval: 500,
  awaitWriteFinish: {
    stabilityThreshold: 300,
    pollInterval: 100,
  },
});

watcher.on("change", (filePath) => {
  const sourceKey = Object.keys(SOURCES).find(k => SOURCES[k] === filePath);
  if (!sourceKey) return;

  const results = readResults(filePath);
  const known = knownLineCounts[sourceKey] || 0;

  if (results.length > known) {
    const newRows = results.slice(known);
    knownLineCounts[sourceKey] = results.length;
    broadcast({ type: "append", source: sourceKey, data: newRows });
    console.log(`Pushed ${newRows.length} new result(s) for ${sourceKey} → ${results.length} total`);
  } else if (results.length < known) {
    // File was truncated / replaced — send full snapshot
    knownLineCounts[sourceKey] = results.length;
    broadcast({ type: "snapshot", source: sourceKey, data: results });
    console.log(`File reset detected for ${sourceKey} — sent full snapshot`);
  }
});

watcher.on("add", (filePath) => {
  console.log(`Watching ${filePath}`);
});

// ── Start ───────────────────────────────────────────────────────────────
server.listen(PORT, () => {
  console.log(`\n  🚀  Dashboard  →  http://localhost:${PORT}\n`);
});
