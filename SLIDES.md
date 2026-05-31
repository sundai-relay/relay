# Relay — Pitch Deck Outline
### Multi-Agent Orchestration Build Day · The Engine, Cambridge · May 31, 2026

> **Format you're presenting into:** Round 1 = **3 min to a judge panel + optional 2-min Q&A**;
> finalists re-present to the full room. Scored on **Agent Orchestration · Utility ·
> Technical Execution · Creativity · Sponsor Usage**. Top prize is literally
> *"Most Sophisticated Harness"* — that is your category. *"Best Use of Weave"* is a
> separate prize you are built to win.
>
> **How to use this file:** 8 core slides for the 3-minute run + 4 appendix slides
> (Q&A / finalist / withheld). Each slide lists **[ON SCREEN]** (keep it sparse — judges
> read *or* listen, not both) and **[SAY]** (your spoken track, ~timed). `‹SWAP›` marks
> every place to drop your **final run numbers** before 7pm. Numbers shown are the real
> `delegate_relay` accounting-domain results — replace only if your final run differs.

---

## THE SPINE (memorize — every slide and every answer routes back here)

> **Re-grounding a drifting agent chain obviously helps. Our claim is narrower and harder:
> a cheap, key-free signal picks *better moments* to re-ground than random does — at the
> *same* intervention budget.**

Say **"at the same budget"** out loud at least twice. It's the phrase that turns a demo
into a result. Beating *naive* proves only that re-grounding helps (obvious). Beating
*random-at-budget* proves the **signal has decision value** — that's the whole project.

---

# CORE DECK (3 minutes, 8 slides — ~22s each)

---

## Slide 1 — Title / hook  ·  ~0:00–0:20
**[ON SCREEN]**
- **RELAY** — selective intervention for lossy multi-agent hand-offs
- one line: *"A flight recorder + circuit breaker for agent hand-offs."*
- Team name · members · `github.com/…` · ▶ 90-sec demo video
- Small logos: W&B Weave · W&B Inference · MCP · Microsoft DELEGATE52

**[SAY]**
> "We're Relay. Long agent chains silently lose the detail that mattered — and in
> production there's no answer key to tell you it happened. Relay makes that loss
> *observable* and *selectively repairable*."

*Hits:* sets up Utility + names the "harness" framing the top prize rewards.

---

## Slide 2 — The problem, made visceral  ·  ~0:20–0:45
**[ON SCREEN]**
- A 4-node chain: `Source → Agent 1 → Agent 2 → Agent 3 → Answer`
- Detail badge that survives hop 1, **drops at hop 2** (turns red), wrong answer at the end
- Caption: *"Summarize → hand off → repeat. The detail that mattered is gone, and nothing flagged it."*

**[SAY]**
> "Every hand-off re-summarizes state. A precise detail — a negation, a number, an
> exception — survives one hop and vanishes the next. The chain looks healthy the whole
> time. Today's fix is to dump the full source into *every* hop: accurate, but the cost
> scales with the chain."

*Hits:* Utility (real, felt-in-the-room pain), sets up why *selective* matters.

---

## Slide 3 — The harness: a Conductor that watches every hand-off  ·  ~0:45–1:10
**[ON SCREEN]**
- Same chain, now with a **Conductor** above it holding the source, reading a checksum after each hop
- Two signal chips:
  - **① Question-conditioned drift** — distance to the *question-relevant* source chunks (local Sentence-BERT). Key-free.
  - **② Answer instability (shadow answerer)** — does the answer the memo supports *flip*? Key-free.
- `risk = f(drift, flip) → intervene only when it spikes`

**[SAY]**
> "A **Conductor** keeps the source and scores every hand-off with two *cheap, key-free*
> signals — how far the memo drifted from the question-relevant source, and whether the
> answer it supports just flipped. When risk spikes, it re-grounds — re-injects the
> relevant source — and *only* then. Gold answers are used **only** for final scoring,
> so nothing's circular."

*Hits:* Agent Orchestration (Conductor acting on *intermediate* inter-agent state) +
Technical Execution (honest, non-circular design).

---

## Slide 4 — THE DEMO: watch a wrong answer become right  ·  ~1:10–1:50  ⭐ centerpiece
**[ON SCREEN]** (a single recorded item, 4 stills or a looping clip)
1. Source sentence with the critical detail highlighted
2. Agent 1 memo — detail kept ✅ · Agent 2 memo — detail dropped ❌
3. **Two needles spike together:** drift ↑ **and** shadow answer flips **C → B**
4. Conductor injects the source sentence → memo repaired → answer back to **C** ✅

**[SAY]**
> "One item, live trace. Agent 1 keeps the detail; Agent 2 drops it. Watch — drift spikes
> *and* the shadow answer flips from C to B. That flip is the point: the hand-off changed
> *what the next agent would do*. The Conductor injects the one source sentence, the memo's
> repaired, and the answer snaps back to C — the correct one."

*Hits:* Technical Execution + Orchestration. **This is your strongest 40 seconds — rehearse it cold.**
*Logistics: play a clean **pre-recorded** run and narrate it. Do not gamble live API latency on your centerpiece.*

---

## Slide 5 — THE MONEY GRAPH: four conditions, same items, same budget  ·  ~1:50–2:25
**[ON SCREEN]** — cost (tokens, x) vs fidelity (y) scatter, 4 labelled points:

| Condition | Fidelity | Avg interventions | Avg tokens |
|---|---|---|---|
| naive | `‹0.127›` | 0.00 | `‹5252›` |
| random_at_budget | `‹0.452›` | 0.83 | `‹6628›` |
| **adaptive (Relay)** | **`‹0.711›`** | 0.83 | `‹6498›` |
| always_reground | `‹0.934›` | 4.00 | `‹11394›` |

- Big callout arrow: **adaptive − random = `‹+0.259›` at the SAME 0.83 interventions**
- Footnote: *always-reground buys the 0.934 ceiling for ~75% more tokens*

**[SAY]**
> "Four conditions, identical items and edits — only the policy changes. Naive is the
> floor; always-reground is the ceiling but costs ~75% more tokens. The two that matter
> sit at the *same* budget: random re-grounding gets `‹0.45›`, our adaptive signal gets
> `‹0.71›` — **plus `‹0.26›` fidelity for the same number of interventions.** That gap is
> the entire project: the signal picks *better moments* than chance."

*Hits:* Creativity (a controlled experiment, not a demo) + Technical Execution.
*‹SWAP all four rows + the delta with your final run. If you also have the chess domain
(adaptive `‹0.896›` vs random `‹0.761›`, `‹+0.135›`), keep it for Slide 7/Q&A as a second datapoint.›*

---

## Slide 6 — Built on Weave: a guardrail on *intermediate* agent state  ·  ~2:25–2:45
**[ON SCREEN]**
- **W&B Inference** — every agent call, OpenAI-compatible, open models, traced by default
- **W&B Weave** — (1) traces every turn/step/LM call · (2) `weave.Evaluation` + exact-match
  scorer → **4-condition leaderboard** · (3) per-hop **drift / instability / intervention /
  cost** as custom **Signals**
- One line: *"observe → score → intervene, applied to inter-agent hand-offs, not final output"*
- **MCP** servers expose the grounding-retrieval tool, the DELEGATE52 loader, and the scorer

**[SAY]**
> "All of it runs on **W&B Inference** and is observable in **Weave** — every hop traced,
> the four conditions published as an Evaluation leaderboard, and drift, instability, and
> each intervention logged as custom Signals. It's a *handoff-fidelity guardrail* built on
> Weave's observe-score-intervene model — applied to the state *between* agents. MCP serves
> the grounding tool, the benchmark loader, and the scorer."

*Hits:* Sponsor Usage — specific enough to win **Best Use of Weave**, not just "we used it."
*Claim A2A only if you actually wired hand-offs over it; otherwise say "MCP for tools, a control loop for hand-offs."*

---

## Slide 7 — Why it generalizes + closer  ·  ~2:45–3:00
**[ON SCREEN]**
- Two domains, same harness: **DELEGATE52 ledgers** (`‹+0.259›`) · **chess PGN** (`‹+0.135›`)
- Vision line: *"Like AlphaProof selecting a solver per domain — select the **eval** per domain. Tailoring the eval compounds the gains."*
- Closer: *"The agentic analogue of a person relaying what an AI told them — same drift, now measurable."*

**[SAY]**
> "Same harness, a real Microsoft benchmark, two domains. And tailoring the eval *per
> domain* widened the gap — that's where this goes: select the eval to fit the problem,
> like AlphaProof selects a solver. This is the measurable version of what happens when a
> person passes along what an AI told them. That's what we're building next."

*Hits:* Creativity + the BD/vision signal that wins finalist rounds.

---

## Slide 8 — Recap card (leave this up during Q&A)  ·  static
**[ON SCREEN]**
- **Problem:** agent chains lose the detail that mattered; no answer key in prod.
- **Relay:** a Conductor scores each hand-off with cheap key-free signals, re-grounds only on a spike.
- **Result:** adaptive beats budget-matched random by **`‹+0.259›`** fidelity at equal cost.
- **Stack:** W&B Inference · Weave (traces + Evaluation leaderboard + Signals) · MCP · DELEGATE52.
- **The claim:** *the signal picks better moments than random — at the same budget.*

---

# APPENDIX (do not present; pull up as needed)

---

## A1 — Q&A landmines (the 2-minute round)
Keep answers to two sentences; every one ends by routing back to the spine.

- **"Isn't this just Weave?"** → "Weave is the observability layer. Relay is a guardrail
  built on it that scores *intermediate* hand-off state and proves the trigger beats random."
- **"How do you know drift means factual loss?"** → "We don't assume it. We measure whether
  it predicts downstream answer failure — and whether acting on it beats random at equal budget."
- **"Sentence-BERT is weak."** → "Intentionally — it's a cheap runtime feature, not the judge.
  The shadow-answerer makes it operational, and we validate the whole risk model against gold."
- **"Is this really multi-agent?"** → "Yes — agents exchange compressed state and a Conductor
  observes, scores, and intervenes on the hand-off itself. The orchestration *is* the point."
- **"Why not always include the source?"** → "Cost, latency, context window, privacy, attention
  budget. Always-reground is our ceiling and costs ~75% more tokens; Relay is selective."
- **"n is small."** → "Proof of concept on a small slice — descriptive results, no p-values on
  tiny n. The design is the contribution; the effect is large and at equal budget."

---

## A2 — Architecture (one diagram, if a judge asks "how's it built?")
- **Agents (W&B Inference, OpenAI-compatible):** Explainer (sees source, writes memo) →
  Relay agents (see only prior memo) → Answerer (answers from final memo).
- **Conductor:** retains source · computes `risk = f(drift, flip)` per hop · re-grounds above
  threshold tuned to a target intervention rate (~30%) · **preserves the legitimate edit, never resets to seed.**
- **Signals:** Sentence-BERT all-MiniLM-L6-v2 (local) for question-conditioned drift; shadow
  answerer (same model, no gold) for answer instability. *(If the Inference endpoint returns
  logprobs, risk = flip + relay top-2 margin drop; else flip + drift.)*
- **Scoring:** deterministic exact-match / round-trip fidelity ∈ [0,1] vs gold — **measurement only.**
- **Four conditions = one loop, four trigger modes:** off / forced-on / random-at-budget / risk-driven.

---

## A3 — Finalist (full-room) adjustments
- Open ~10s bigger on the **problem** — a room of builders has all felt context-loss; make them nod.
- Demo on the big screen; let the C→B→C flip breathe.
- End on the **vision** (Slide 7) not the mechanics — select-the-eval-per-domain is the venture.
- Same spine, same money graph. Don't add new claims under pressure.

---

## A4 — The sauce (withhold; tease only)
Show: drift/instability curves, the fidelity/cost frontier, gold accuracy.
Withhold: typed cognitive-level divergence per learner against a curated reference.
If pushed: *"How would you diagnose *which* understanding broke, per learner, at what
cognitive level? That's the venture — let's talk offline."*

---

## PRE-FLIGHT CHECKLIST (before 7pm draft / 8pm final)
- [ ] `‹SWAP›` every bracketed number with your **final** run; re-screenshot the money graph.
- [ ] Demo video **< 2 min** (submission rule), clean recorded run, the C→B→C moment visible.
- [ ] Repo **public** (eligibility rule); README links the four-condition run command.
- [ ] Submission copy filled: team name, members + emails/socials, X/LinkedIn handles,
      2–3 sentence summary, problem, **how it's built (name MCP + frameworks)**, sponsor-tool list.
- [ ] Only claim what you built — A2A only if wired; else "MCP for tools, control loop for hand-offs."
- [ ] One slide visible per beat; you speak the rest. Practice the demo narration cold, twice.
