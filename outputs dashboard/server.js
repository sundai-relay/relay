import express from "express";
import { createServer } from "http";
import { WebSocketServer } from "ws";
import { watch } from "chokidar";
import { readFileSync, existsSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const RESULTS_PATH = resolve(__dirname, "../outputs/results.jsonl");
const PORT = process.env.PORT || 3000;

// ── Express + HTTP ──────────────────────────────────────────────────────
const app = express();
app.use(express.static(resolve(__dirname, "public")));
const server = createServer(app);

// ── WebSocket ───────────────────────────────────────────────────────────
const wss = new WebSocketServer({ server });

/** Parse the JSONL file and return an array of objects */
function readResults() {
  if (!existsSync(RESULTS_PATH)) return [];
  const raw = readFileSync(RESULTS_PATH, "utf-8");
  return raw
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line, idx) => {
      try {
        return JSON.parse(line);
      } catch {
        console.warn(`Skipping malformed line ${idx + 1}`);
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
let knownLineCount = 0;

// On new connection, send the full snapshot
wss.on("connection", (ws) => {
  console.log("Client connected");
  const results = readResults();
  knownLineCount = results.length;
  ws.send(JSON.stringify({ type: "snapshot", data: results }));
});

// ── File watcher ────────────────────────────────────────────────────────
const watcher = watch(RESULTS_PATH, {
  persistent: true,
  usePolling: true,       // reliable for files written by other processes
  interval: 500,
  awaitWriteFinish: {
    stabilityThreshold: 300,
    pollInterval: 100,
  },
});

watcher.on("change", () => {
  const results = readResults();
  if (results.length > knownLineCount) {
    const newRows = results.slice(knownLineCount);
    knownLineCount = results.length;
    broadcast({ type: "append", data: newRows });
    console.log(`Pushed ${newRows.length} new result(s) → ${knownLineCount} total`);
  } else if (results.length < knownLineCount) {
    // File was truncated / replaced — send full snapshot
    knownLineCount = results.length;
    broadcast({ type: "snapshot", data: results });
    console.log("File reset detected — sent full snapshot");
  }
});

watcher.on("add", () => {
  console.log(`Watching ${RESULTS_PATH}`);
});

// ── Start ───────────────────────────────────────────────────────────────
server.listen(PORT, () => {
  console.log(`\n  🚀  Dashboard  →  http://localhost:${PORT}\n`);
});
