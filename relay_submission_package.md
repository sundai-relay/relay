# Relay — Submission Package (final)
### Multi-Agent Orchestration Build Day · The Engine, Cambridge · May 31, 2026

**What it is.** A runtime observability harness — a flight recorder plus circuit breaker — for lossy multi-agent hand-offs. As an explanation is relayed down a chain of agents, a Conductor estimates each hop's *handoff-degradation risk* from cheap, key-free signals and selectively re-grounds the chain only when risk spikes. We validate whether those signals actually predict task loss against gold answers, and show in Weave that adaptive re-grounding beats naive relay, always-reground, **and a budget-matched random control** on a fidelity/cost frontier.

**Honest scope.** We are not claiming to detect truth or "solve drift." The signals are cheap runtime features, not a factuality detector; the scientific claim is operational — *does this risk model pick better re-grounding moments than random at the same budget?*

---

## The two key-free signals (what the Conductor watches)
1. **Question-conditioned drift** — retrieve the source chunks most relevant to the question, then measure cosine distance between each relayed memo and *those* chunks (not the whole source, which would just punish compression). Sentence-BERT (all-MiniLM-L6-v2, local). A coarse semantic-displacement feature.
2. **Answer instability (shadow answerer)** — at each hop, answer the MCQ from the current memo *without the gold answer*, and track whether the predicted choice flips or confidence drops. This measures whether the handoff changed *what the downstream agent would do* — the operational concern, and far harder to dismiss than an embedding number.

Risk = a transparent combination of the two; intervene above a threshold tuned to a target intervention rate. Gold answers are used **only** for final scoring — neither signal touches them, so nothing is circular.

## The result (the money graph)
Same items, four conditions:

| Condition | Proves | Cost | Fidelity |
|---|---|---|---|
| Naive relay | loss exists | low | low |
| Always-reground | upper bound | high | high |
| **Random-at-budget** | re-grounding itself helps | = adaptive | modest |
| **Adaptive (Relay)** | the signal has *decision value* | = random | best per token |

The whole project is the **adaptive-vs-random** comparison: does the cheap signal pick better moments than chance at the same cost? A cost/fidelity scatter with four labelled points is the demo.

---

## PART A — SUBMISSION COPY (paste into AGI House)
**Team / members / repo / demo video:** `[fill]`

**1. Summary (2–3 sentences).** Relay is a runtime observability harness for lossy multi-agent hand-offs. A Conductor estimates each hop's handoff-degradation risk from two cheap key-free signals — question-conditioned semantic drift and shadow-answer instability — and selectively re-grounds the chain only when risk spikes, logging the full fidelity/cost tradeoff in Weave. We validate whether the signals predict real answer loss by comparing adaptive re-grounding against naive relay, always-reground, and a budget-matched random baseline.

**2. What it does / problem it solves.** Long agent pipelines silently drop the detail that mattered as state is summarized and handed off, and in production there's no answer key to tell you it happened. Relay makes that loss observable and selectively repairable: cheap key-free signals flag degradation in flight, and the Conductor re-injects question-relevant source content only when risk is high. We don't assume the signals mean factual loss — we measure whether they predict downstream answer failure and whether acting on them beats random intervention at the same cost.

**3. How it's built — protocols, frameworks, tools.** A chain of LLM agents on W&B Inference (OpenAI-compatible): an Explainer (sees the source, writes a handoff memo), Relay agents (see only the prior memo), an Answerer (answers from the final memo), and a Conductor that retains the source, scores each hop, and decides whether to re-ground. Signals: Sentence-BERT (all-MiniLM-L6-v2, local) for question-conditioned drift; a shadow-answerer (same model) for answer instability. Scoring: deterministic exact-match MCQ vs gold (measurement only). Protocols: **MCP** servers expose the source-retrieval/grounding tool, the benchmark loader, and the scorer (primary, real usage); **A2A** for agent hand-offs *only if wired by mid-afternoon — claim only what you built*. Control: a Python loop; the four conditions are the same loop with the trigger off / forced-on / random-at-budget / risk-driven.

**4. Sponsor tools used and how.** **W&B Inference** — all model calls via the OpenAI-compatible endpoint, open models, traced by default. **W&B Weave** — (1) traces every turn/step/LM call; (2) `weave.Evaluation` + exact-match scorer over the four conditions, published as a fidelity/cost **leaderboard**; (3) per-hop drift, answer-instability, intervention decision, and cost logged as **custom signals/scores** — a domain-specific *handoff-fidelity guardrail* built on Weave's observe→score→intervene model, applied to *intermediate inter-agent state* rather than final output. *(Best-Use-of-Weave: tracing + evaluation + leaderboard + custom signals + guardrail-style control.)*

---

## PART B — PITCH & PREP (don't submit)

**Spoken pitch (~45s core).** "Multi-agent systems lose information silently. Relay makes that loss observable. We pass a source through a chain of agents, and a Conductor measures — without the answer key — how far each handoff drifts from the *question-relevant* source and whether the answer the memo supports has started to change, then re-grounds only when the signal spikes. The point isn't that re-grounding helps — of course it does. The result is that *adaptive* re-grounding beats *random* re-grounding at the same budget, moving toward always-grounded fidelity at near-relay cost."
**Closer — keep this; it's the BD signal:** "This is the agentic analogue of what happens when a person passes along what an AI told them — the same drift, measurable in a controlled relay. That's the thing we're building next." *(Analogy, not a replication claim.)*

**The demo = show the failure, not the evals.** One item: source has a critical detail → Agent 1 keeps it → Agent 2 drops it → drift spikes *and* the shadow answer flips C→B → Conductor injects the source sentence → memo repaired → answer back to C. Then the Weave trace, then the four-bar leaderboard.

**Q&A answers.**
- *Why not always include the source?* Cost, latency, context-window, privacy, attention budget — Relay is selective re-grounding.
- *Why Sentence-BERT, isn't it weak?* Intentionally — a cheap runtime feature, not the judge; the shadow-answerer makes it operational, and we validate the whole risk model against gold accuracy.
- *How do you know drift means factual loss?* We don't assume it; we measure whether it predicts downstream failure and whether acting on it beats random at the same budget.
- *Is this really multi-agent?* Yes — agents exchange compressed state and a Conductor observes, scores, and intervenes on intermediate state; the orchestration behavior is the point.
- *Isn't this just Weave?* Weave is the observability/eval/guardrail layer; Relay is a handoff-fidelity guardrail built on it — we score *intermediate* state and prove the trigger beats random.

**Sauce line.** Show: the drift/instability curves, the fidelity/cost frontier, gold accuracy. Withhold: typed cognitive-level divergence against a curated reference. "How would you diagnose *which* understanding broke, per learner, at *what* cognitive level?" → "That's the venture — let's talk offline."

**Framing honesty (say it).** Proof of concept over a small slice; descriptive results, no p-values on tiny n.
