/* ═══════════════════════════════════════════════════════════════════════
   Relay Dashboard — Regrounding Strategy Comparison
   ═══════════════════════════════════════════════════════════════════════ */

(() => {
  "use strict";

  // ── Condition config ────────────────────────────────────────────────
  const CONDITIONS = {
    naive:            { label: "Naive",             css: "naive",    order: 0 },
    always_reground:  { label: "Always Reground",   css: "always",   order: 1 },
    random_at_budget: { label: "Random at Budget",  css: "random",   order: 2 },
    adaptive:         { label: "Adaptive",          css: "adaptive", order: 3 },
  };

  const COND_COLORS = {
    naive:            "#6b7280",
    always_reground:  "#60a5fa",
    random_at_budget: "#f59e0b",
    adaptive:         "#34d399",
  };

  // ── State ───────────────────────────────────────────────────────────
  let allResults = {
    accounting: [],
    accounting_tuned: [],
    chess: [],
    chess_tuned: []
  };
  let ws = null;
  let reconnectTimer = null;
  let hasConnectedOnce = false;
  let staticMode = false;
  const RECONNECT_DELAY = 2000;

  // ── DOM refs ────────────────────────────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const connectionBar   = $("#connection-bar");
  const connectionLabel = $("#connection-label");
  const resultCounter   = $("#result-counter");
  const leaderboardCards = $("#leaderboard-cards");
  const gateSection     = $("#gate-section");
  const gateVerdictBadge = $("#gate-verdict-badge");
  const gateDetails     = $("#gate-details");
  const scoreBars       = $("#score-bars");
  const interventionBars = $("#intervention-bars");
  const efficiencyBars  = $("#efficiency-bars");
  const distributionCanvas = $("#distribution-chart");
  const filterTask      = $("#filter-task");
  const filterBoundary  = $("#filter-boundary");
  const taskGrid        = $("#task-comparison-grid");
  const resultsBody     = $("#results-body");
  const emptyState      = $("#empty-state");
  const leaderboard     = $("#leaderboard");
  const chartsSection   = $("#charts-section");
  const taskSection     = $("#task-section");
  const tableSection    = $("#table-section");
  const mainSections    = [leaderboard, gateSection, chartsSection, taskSection, tableSection];

  // ── Helpers ─────────────────────────────────────────────────────────
  function scoreClass(s) {
    if (s >= 1)    return "score-perfect";
    if (s >= 0.95) return "score-good";
    if (s >= 0.8)  return "score-mid";
    return "score-low";
  }

  function getFilteredResults() {
    const activeDomains = Array.from(document.querySelectorAll("#filter-domain-pills .filter-pill.active")).map(el => el.dataset.value);
    const activeTunings = Array.from(document.querySelectorAll("#filter-tuning-pills .filter-pill.active")).map(el => el.dataset.value);

    let merged = [];
    if (activeDomains.includes("accounting")) {
      if (activeTunings.includes("base")) merged.push(...(allResults.accounting || []));
      if (activeTunings.includes("tuned")) merged.push(...(allResults.accounting_tuned || []));
    }
    if (activeDomains.includes("chess")) {
      if (activeTunings.includes("base")) merged.push(...(allResults.chess || []));
      if (activeTunings.includes("tuned")) merged.push(...(allResults.chess_tuned || []));
    }
    return merged;
  }

  /** Group results by condition → aggregate stats */
  function aggregateByCondition(results) {
    const groups = {};
    for (const r of results) {
      if (!groups[r.condition]) groups[r.condition] = [];
      groups[r.condition].push(r);
    }

    const summaries = {};
    for (const [cond, rows] of Object.entries(groups)) {
      // Group by task to find final step info (max boundary)
      const taskRows = {};
      for (const r of rows) {
        const tid = r.sample_id || r.task_id || r.episode_id || "unknown";
        if (!taskRows[tid]) taskRows[tid] = [];
        taskRows[tid].push(r);
      }

      const taskScores = {};
      const taskInterventions = {};
      const taskCosts = {};
      
      let totalInterventions = 0;
      let totalSteps = rows.length;

      for (const [tid, trows] of Object.entries(taskRows)) {
        // Sort by boundary to find the last row
        trows.sort((a, b) => {
          const ab = a.boundary !== undefined ? a.boundary : (a.round_trip_index !== undefined ? a.round_trip_index : 0);
          const bb = b.boundary !== undefined ? b.boundary : (b.round_trip_index !== undefined ? b.round_trip_index : 0);
          return ab - bb;
        });
        
        const lastRow = trows[trows.length - 1];
        
        const score = Math.max(0, 1 - (lastRow.inv_dev_after !== undefined ? lastRow.inv_dev_after : (lastRow.invariant_deviation ?? 0))) * (lastRow.parse_health !== undefined ? lastRow.parse_health : 1.0);
        taskScores[tid] = score;
        
        const cost = lastRow.tokens_cum !== undefined ? lastRow.tokens_cum : (lastRow.cost_proxy !== undefined ? lastRow.cost_proxy : 0);
        taskCosts[tid] = cost;

        let taskInt = 0;
        for (const r of trows) {
          if (r.intervened) {
            taskInt++;
            totalInterventions++;
          }
        }
        taskInterventions[tid] = taskInt;
      }

      const scores = Object.values(taskScores);
      const costs = Object.values(taskCosts);
      const nTasks = scores.length;
      
      const avgScore = nTasks > 0 ? scores.reduce((a, b) => a + b, 0) / nTasks : 0;
      const avgCost = nTasks > 0 ? costs.reduce((a, b) => a + b, 0) / nTasks : 0;

      summaries[cond] = {
        condition: cond,
        avgScore,
        nTasks,
        totalSteps,
        totalInterventions,
        interventionRate: totalSteps > 0 ? totalInterventions / totalSteps : 0,
        avgCost,
        scorePerCost: avgCost > 0 ? avgScore / avgCost : 0,
        scores,
        allScores: rows.map(r => Math.max(0, 1 - (r.inv_dev_after !== undefined ? r.inv_dev_after : (r.invariant_deviation ?? 0))) * (r.parse_health !== undefined ? r.parse_health : 1.0)),
      };
    }
    return summaries;
  }

  /** Group results by task → by condition → final score */
  function groupByTask(results) {
    const tasks = {};
    for (const r of results) {
      const tid = r.sample_id || r.task_id || r.episode_id || "unknown";
      if (!tasks[tid]) tasks[tid] = {};
      if (!tasks[tid][r.condition]) tasks[tid][r.condition] = { steps: 0, score: 0, rounds: 0 };
      const t = tasks[tid][r.condition];
      t.steps++;
      const score = Math.max(0, 1 - (r.inv_dev_after !== undefined ? r.inv_dev_after : (r.invariant_deviation ?? 0))) * (r.parse_health !== undefined ? r.parse_health : 1.0);
      t.score = score;
      const boundary = r.boundary !== undefined ? r.boundary : (r.round_trip_index !== undefined ? r.round_trip_index : 0);
      t.rounds = Math.max(t.rounds, boundary + 1);
    }
    return tasks;
  }

  // ── Render: Leaderboard ─────────────────────────────────────────────
  function renderLeaderboard(summaries) {
    const sorted = Object.values(summaries).sort((a, b) => b.avgScore - a.avgScore);

    leaderboardCards.innerHTML = sorted.map((s, i) => {
      const conf = CONDITIONS[s.condition] || { label: s.condition, css: "naive" };
      return `
        <div class="lb-card ${conf.css}">
          <div class="rank">#${i + 1}</div>
          <div class="condition-name">${conf.label}</div>
          <div class="score-big">${s.avgScore.toFixed(4)}</div>
          <div class="lb-metrics">
            <div class="lb-metric"><span class="label">Tasks</span><span class="value">${s.nTasks}</span></div>
            <div class="lb-metric"><span class="label">Steps</span><span class="value">${s.totalSteps}</span></div>
            <div class="lb-metric"><span class="label">Interventions</span><span class="value">${s.totalInterventions}</span></div>
            <div class="lb-metric"><span class="label">Interv. Rate</span><span class="value">${(s.interventionRate * 100).toFixed(1)}%</span></div>
            <div class="lb-metric"><span class="label">Avg Cost (Tok)</span><span class="value">${s.avgCost.toFixed(0)}</span></div>
            <div class="lb-metric"><span class="label">Score/Cost (*1k)</span><span class="value">${(s.scorePerCost * 1000).toFixed(4)}</span></div>
          </div>
        </div>
      `;
    }).join("");
  }

  // ── Render: Gate verdict ────────────────────────────────────────────
  function renderGate(summaries) {
    const naive = summaries.naive;
    const always = summaries.always_reground;
    const adaptive = summaries.adaptive;
    const random = summaries.random_at_budget;

    if (!naive || !always) {
      gateSection.classList.add("hidden");
      return;
    }

    gateSection.classList.remove("hidden");
    const gap = always.avgScore - naive.avgScore;
    let verdict, verdictIcon;
    if (gap >= 0.15) { verdict = "green"; verdictIcon = "✓"; }
    else if (gap >= 0.05) { verdict = "yellow"; verdictIcon = "!"; }
    else { verdict = "red"; verdictIcon = "✗"; }

    gateVerdictBadge.className = verdict;
    gateVerdictBadge.textContent = verdictIcon;

    let details = `<strong>always_reground − naive = <code>${gap >= 0 ? "+" : ""}${gap.toFixed(4)}</code></strong>`;
    if (verdict === "green") {
      details += ` — Degradation is real and recovery works.`;
    } else if (verdict === "yellow") {
      details += ` — Marginal gap; consider more round-trips or harder edits.`;
    } else {
      details += ` — Gap too small; increase difficulty before building adaptive.`;
    }

    if (adaptive && random) {
      const adpGap = adaptive.avgScore - random.avgScore;
      details += `<br/><strong>adaptive − random = <code>${adpGap >= 0 ? "+" : ""}${adpGap.toFixed(4)}</code></strong>`;
      if (adpGap > 0) {
        details += ` — The risk signal picks better moments than chance. ✓`;
      } else {
        details += ` — Signal does not outperform random at same budget.`;
      }
    }

    gateDetails.innerHTML = details;
  }

  // ── Render: Bar charts ──────────────────────────────────────────────
  function renderBarChart(container, summaries, valueKey, format, maxOverride) {
    const entries = Object.values(summaries).sort(
      (a, b) => (CONDITIONS[a.condition]?.order ?? 99) - (CONDITIONS[b.condition]?.order ?? 99)
    );
    const maxVal = maxOverride ?? Math.max(...entries.map((e) => e[valueKey]), 0.001);

    container.innerHTML = entries.map((s) => {
      const conf = CONDITIONS[s.condition] || { label: s.condition, css: "naive" };
      const pct = (s[valueKey] / maxVal) * 100;
      const formatted = typeof format === "function" ? format(s[valueKey]) : s[valueKey].toFixed(4);
      return `
        <div class="bar-row">
          <span class="bar-label">${conf.label}</span>
          <div class="bar-track">
            <div class="bar-fill ${conf.css}" style="width:${Math.max(pct, 2)}%"></div>
            <span class="bar-value">${formatted}</span>
          </div>
        </div>
      `;
    }).join("");
  }

  // ── Render: Score distribution (canvas) ─────────────────────────────
  function renderDistribution(summaries) {
    const canvas = distributionCanvas;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const w = rect.width;
    const h = rect.height;
    const pad = { top: 12, right: 14, bottom: 28, left: 44 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);
      ctx.stroke();
    }

    // X-axis labels: 0 to 1
    ctx.fillStyle = "#4e5870";
    ctx.font = "10px JetBrains Mono, monospace";
    ctx.textAlign = "center";
    for (let i = 0; i <= 5; i++) {
      const x = pad.left + (plotW / 5) * i;
      ctx.fillText((i / 5).toFixed(1), x, h - 6);
    }

    const buckets = 25;
    const condEntries = Object.entries(summaries).sort(
      (a, b) => (CONDITIONS[a[0]]?.order ?? 99) - (CONDITIONS[b[0]]?.order ?? 99)
    );

    // Find global max count for Y scaling
    let globalMax = 1;
    const allCounts = {};
    for (const [cond, s] of condEntries) {
      const counts = new Array(buckets).fill(0);
      for (const score of s.scores) {
        const idx = Math.min(Math.floor(score * buckets), buckets - 1);
        counts[idx]++;
      }
      allCounts[cond] = counts;
      globalMax = Math.max(globalMax, ...counts);
    }

    // Y labels
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const val = globalMax - (globalMax / 4) * i;
      const y = pad.top + (plotH / 4) * i;
      ctx.fillText(Math.round(val), pad.left - 6, y + 3);
    }

    // Draw lines for each condition
    for (const [cond, counts] of Object.entries(allCounts)) {
      const color = COND_COLORS[cond] || "#888";

      ctx.beginPath();
      for (let i = 0; i < buckets; i++) {
        const x = pad.left + ((i + 0.5) / buckets) * plotW;
        const y = pad.top + plotH - (counts[i] / globalMax) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.stroke();

      // Fill under line
      ctx.lineTo(pad.left + ((buckets - 0.5) / buckets) * plotW, pad.top + plotH);
      ctx.lineTo(pad.left + (0.5 / buckets) * plotW, pad.top + plotH);
      ctx.closePath();
      
      // Use hex to rgba conversion
      const r = parseInt(color.slice(1, 3), 16);
      const g = parseInt(color.slice(3, 5), 16);
      const b = parseInt(color.slice(5, 7), 16);
      ctx.fillStyle = `rgba(${r},${g},${b},0.06)`;
      ctx.fill();
    }
  }

  // ── Render: Per-task comparison cards ───────────────────────────────
  function renderTaskCards(results) {
    const taskFilter = filterTask.value;
    const boundaryFilter = filterBoundary.value;

    let filtered = results;
    if (boundaryFilter !== "") {
      filtered = filtered.filter((r) => {
        const boundary = r.boundary !== undefined ? r.boundary : (r.round_trip_index !== undefined ? r.round_trip_index : 0);
        return String(boundary) === boundaryFilter;
      });
    }

    const tasks = groupByTask(filtered);
    const taskIds = Object.keys(tasks).sort();

    // Update task filter options
    const currentVal = filterTask.value;
    filterTask.innerHTML = '<option value="">All Tasks</option>';
    for (const tid of taskIds) {
      const opt = document.createElement("option");
      opt.value = tid;
      opt.textContent = tid;
      filterTask.appendChild(opt);
    }
    if (taskIds.includes(currentVal)) {
      filterTask.value = currentVal;
    } else {
      filterTask.value = "";
    }

    const displayTasks = taskFilter ? { [taskFilter]: tasks[taskFilter] } : tasks;

    taskGrid.innerHTML = Object.entries(displayTasks)
      .filter(([, v]) => v)
      .map(([tid, conds]) => {
        const condEntries = Object.entries(conds).sort(
          (a, b) => (CONDITIONS[a[0]]?.order ?? 99) - (CONDITIONS[b[0]]?.order ?? 99)
        );
        const maxRounds = Math.max(...condEntries.map(([, c]) => c.rounds), 0);

        const rows = condEntries.map(([cond, data]) => {
          const conf = CONDITIONS[cond] || { label: cond, css: "naive" };
          const pct = (data.score / 1.0) * 100; // scores are 0-1
          return `
            <div class="task-condition-row">
              <div class="task-cond-dot ${conf.css}"></div>
              <div class="task-cond-name">${conf.label}</div>
              <div class="task-cond-bar-track">
                <div class="task-cond-bar-fill ${conf.css}" style="width:${pct}%"></div>
              </div>
              <div class="task-cond-score">${data.score.toFixed(4)}</div>
            </div>
          `;
        }).join("");

        return `
          <div class="task-card">
            <div class="task-card-header">
              <div class="task-card-title">${tid}</div>
              <div class="task-card-rounds">${maxRounds} round${maxRounds !== 1 ? "s" : ""}</div>
            </div>
            ${rows}
          </div>
        `;
      }).join("");
  }

  // ── Render: Results table ───────────────────────────────────────────
  function renderTable(results, animate) {
    resultsBody.innerHTML = results.map((r, i) => {
      const conf = CONDITIONS[r.condition] || { label: r.condition, css: "naive" };
      const cls = animate && i >= results.length - 1 ? ' class="row-enter"' : "";
      
      const score = Math.max(0, 1 - (r.inv_dev_after !== undefined ? r.inv_dev_after : (r.invariant_deviation ?? 0))) * (r.parse_health !== undefined ? r.parse_health : 1.0);
      const boundary = r.boundary !== undefined ? r.boundary : (r.round_trip_index !== undefined ? r.round_trip_index : 0);
      const edit = r.edit || r.edit_name || "none";
      const opsHtml = (r.ops || []).map(op => `<span style="background:rgba(255,255,255,0.03);color:var(--text-secondary);padding:1px 6px;border-radius:4px;font-size:0.65rem;margin-right:4px">${op}</span>`).join('');
      const risk = r.risk !== undefined ? r.risk : (r.runtime_risk !== undefined ? r.runtime_risk : 0.0);
      const drift = r.embedding_drift !== undefined ? r.embedding_drift : (r.value_drift_rate !== undefined ? r.value_drift_rate : 0.0);
      const cost = r.tokens_cum !== undefined ? r.tokens_cum : (r.cost_proxy !== undefined ? r.cost_proxy : 0);
      const tid = r.sample_id || r.task_id || r.episode_id || "unknown";

      return `
        <tr${cls}>
          <td>${i + 1}</td>
          <td style="color:var(--text-primary);font-family:'Inter', system-ui, -apple-system, sans-serif;font-weight:600">${tid}</td>
          <td><span class="cond-tag ${r.condition}">${conf.label}</span></td>
          <td>Boundary ${boundary}</td>
          <td style="font-family:'Inter', system-ui, -apple-system, sans-serif">${edit}</td>
          <td><div style="display:flex;flex-wrap:wrap;gap:2px">${opsHtml}</div></td>
          <td><span class="score-pill ${scoreClass(score)}">${score.toFixed(4)}</span></td>
          <td>${risk.toFixed(4)}</td>
          <td>${drift.toFixed(4)}</td>
          <td><span class="flag-${!!r.intervened}">${r.intervened ? "Yes" : "No"}</span></td>
          <td>${cost}</td>
        </tr>
      `;
    }).join("");
  }

  // ── Master render ───────────────────────────────────────────────────
  function render(animate = false) {
    const filtered = getFilteredResults();
    resultCounter.textContent = `${filtered.length} results`;

    if (filtered.length === 0) {
      emptyState.classList.remove("hidden");
      mainSections.forEach((s) => s.classList.add("hidden"));
      return;
    }

    emptyState.classList.add("hidden");
    mainSections.forEach((s) => s.classList.remove("hidden"));

    const summaries = aggregateByCondition(filtered);

    renderLeaderboard(summaries);
    renderGate(summaries);
    renderBarChart(scoreBars, summaries, "avgScore", (v) => v.toFixed(4), 1.0);
    renderBarChart(interventionBars, summaries, "interventionRate", (v) => (v * 100).toFixed(1) + "%");
    renderBarChart(efficiencyBars, summaries, "scorePerCost", (v) => (v * 1000).toFixed(4));
    renderDistribution(summaries);
    renderTaskCards(filtered);
    renderTable(filtered, animate);
  }

  // ── Static Results Fallback ─────────────────────────────────────────
  async function loadStaticResults() {
    connectionBar.className = "static";
    connectionLabel.textContent = "Static Mode";

    const filenames = {
      accounting: "accounting.results.jsonl",
      accounting_tuned: "accounting.results.tuned.jsonl",
      chess: "chess.results.jsonl",
      chess_tuned: "chess.results.tuned.jsonl"
    };

    const prefixes = [
      "../../outputs/",
      "../outputs/",
      "outputs/",
      "./outputs/"
    ];

    // Find the first prefix that works by trying to fetch the accounting file
    let workingPrefix = null;
    for (const prefix of prefixes) {
      try {
        const testUrl = `${prefix}${filenames.accounting}`;
        const response = await fetch(testUrl, { method: "HEAD" });
        if (response.ok) {
          workingPrefix = prefix;
          console.log(`Found working outputs directory at relative path: ${workingPrefix}`);
          break;
        }
      } catch (e) {
        // Continue
      }
    }

    // Fallback: try standard fetch if HEAD is blocked or fails
    if (!workingPrefix) {
      for (const prefix of prefixes) {
        try {
          const testUrl = `${prefix}${filenames.accounting}`;
          const response = await fetch(testUrl);
          if (response.ok) {
            workingPrefix = prefix;
            console.log(`Found working outputs directory via GET at relative path: ${workingPrefix}`);
            break;
          }
        } catch (e) {
          // Continue
        }
      }
    }

    if (!workingPrefix) {
      console.error("Could not find outputs directory at any expected relative path.");
      connectionLabel.textContent = "Static (Error)";
      return;
    }

    let loadedAny = false;
    for (const [key, filename] of Object.entries(filenames)) {
      const path = `${workingPrefix}${filename}`;
      try {
        const response = await fetch(path);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const text = await response.text();
        const parsed = text
          .split("\n")
          .filter(line => line.trim().length > 0)
          .map((line, idx) => {
            try {
              return JSON.parse(line);
            } catch (err) {
              console.warn(`Skipping malformed line ${idx + 1} in static fetch of ${key}`);
              return null;
            }
          })
          .filter(Boolean);
        
        allResults[key] = parsed;
        loadedAny = true;
      } catch (err) {
        console.error(`Failed to load static results for ${key} from ${path}:`, err);
      }
    }

    if (loadedAny) {
      render(false);
      connectionLabel.textContent = "Static Mode";
    } else {
      connectionLabel.textContent = "Static (Empty)";
    }
  }

  // ── WebSocket ───────────────────────────────────────────────────────
  function connect() {
    clearTimeout(reconnectTimer);
    if (ws) {
      try {
        ws.close();
      } catch (e) {}
    }

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    
    // If running under file protocol, location.host is empty, so WebSocket will fail.
    // Immediately fall back to static mode.
    if (location.protocol === "file:") {
      console.log("Running from file protocol, falling back to static mode.");
      if (!staticMode) {
        staticMode = true;
        loadStaticResults();
      }
      return;
    }

    ws = new WebSocket(`${proto}//${location.host}`);

    // Set a timeout to check if we can connect. If it takes too long and we haven't connected before, fallback.
    const fallbackTimeout = setTimeout(() => {
      if (!hasConnectedOnce && !staticMode) {
        console.log("WebSocket connection timed out. Falling back to static mode.");
        staticMode = true;
        loadStaticResults();
      }
    }, 1500);

    ws.addEventListener("open", () => {
      clearTimeout(fallbackTimeout);
      hasConnectedOnce = true;
      staticMode = false;
      connectionBar.className = "connected";
      connectionLabel.textContent = "Live";
      clearTimeout(reconnectTimer);
    });

    ws.addEventListener("message", (evt) => {
      const msg = JSON.parse(evt.data);
      if (msg.type === "snapshot") {
        allResults[msg.source] = msg.data;
        render(false);
      } else if (msg.type === "append") {
        if (!allResults[msg.source]) allResults[msg.source] = [];
        allResults[msg.source].push(...msg.data);
        render(true);
      }
    });

    ws.addEventListener("close", () => {
      clearTimeout(fallbackTimeout);
      
      if (!hasConnectedOnce && !staticMode) {
        console.log("WebSocket connection closed before opening. Falling back to static mode.");
        staticMode = true;
        loadStaticResults();
      }

      if (hasConnectedOnce) {
        // If we connected once and then disconnected, the server was running but stopped.
        // Show disconnected status and retry immediately.
        connectionBar.className = "disconnected";
        connectionLabel.textContent = "Reconnecting…";
        reconnectTimer = setTimeout(connect, RECONNECT_DELAY);
      } else {
        // If we are in static mode (or never connected), try to reconnect occasionally in the background
        // unless we are hosted on a standard static GitHub Page environment.
        const isGitHubPages = location.hostname.endsWith(".github.io");
        if (!isGitHubPages) {
          reconnectTimer = setTimeout(connect, 10000);
        }
      }
    });

    ws.addEventListener("error", () => {
      ws.close();
    });
  }

  // ── Filter listeners ───────────────────────────────────────────────
  [filterTask, filterBoundary].forEach((el) => {
    el.addEventListener("change", () => render(false));
  });

  // ── Pill Filter listeners ──────────────────────────────────────────
  document.querySelectorAll(".filter-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      pill.classList.toggle("active");
      render(false);
    });
  });

  // ── Resize ─────────────────────────────────────────────────────────
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => render(false), 150);
  });

  // ── Boot ────────────────────────────────────────────────────────────
  connect();
})();
