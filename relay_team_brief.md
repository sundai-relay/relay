# Relay — Team Brief & Build Plan (v4, complete)
### Multi-Agent Orchestration Build Day · The Engine, 750 Main St, Cambridge · May 31, 2026
### Hosted by AGI House × W&B × TNT × SundAI × E14 · Build 11:30 → draft 7:00 → final 8:00 → awards 9:30

> Open this cold. Everything you need is here: what we're building, why, the rules, who does what, the hour-by-hour plan, the exact prompts, the pitch, and the things that can break.

---

## 0. The 30-second version
Multi-agent pipelines lose information at every hand-off, and in production you can't see it happen because you have no answer key. **Relay** is a runtime observability harness — a flight recorder + circuit breaker for lossy hand-offs. A chain of agents passes a compressed memo hop to hop; a **Conductor** estimates per-hop degradation risk from key-free signals and selectively **re-grounds** only when risk spikes. We validate the signals against gold answers and show, in Weave, whether adaptive re-grounding beats naive, always-reground, **and a budget-matched random control** on a fidelity/cost frontier.

**Locked claim (honest, post-spike):** strong single models barely lose info, but under realistic **cost pressure** (short memos, cheap relay models, several hops) information degrades, and **answer-instability is the live signal that catches it; embedding drift is near-inert.** We are NOT claiming a truth detector.

---

## 1. Why this matters (the pitch's backbone — three real sources)
- **Anthropic, "How we built our multi-agent research system":** multi-agent works because *the essence of search is compression* — subagents compress findings back to a lead. Their appendix names the exact failure mode the **"game of telephone"** and routes around it by writing to a filesystem. Token usage explains ~80% of performance variance; multi-agent burns ~15× the tokens of chat. → our cost/fidelity frontier is their economic constraint, and their telephone problem is our thesis.
- **arXiv 2604.10290, "AI Organizations are More Effective but Less Aligned":** multi-agent orgs are more capable but less aligned, via **task decomposition** and **miscoordination**. → Relay instruments one slice of that: whether task-critical info survives the hand-off.
- **W&B (this event's hosts), Emmanuel + Nico's talk:** offline evals can't cover the behavior space, so score things in flight; trace, benchmark, evaluate the outcome, then optimize. → Relay's drift/instability signal is exactly an in-flight (online) signal, validated by an offline benchmark.

**The seam, in one line:** *"Compression is how multi-agent systems scale; decomposition and lossy hand-offs are where they fail. Relay watches that seam — it measures when a hand-off is losing the answer, and re-grounds only when it's worth the tokens."*

---

## 2. Goals & intent (what we are and are not doing)
**Primary goal:** win/place — strongest on **Best Use of Weave** (near-floor for us) and a real shot at **Most Sophisticated Harness** if the adaptive result lands.
**Honest status:** the *story* and *harness* are strong; the *winning result* (adaptive beats random on the frontier) is **unproven** and depends on today's gate. The fallback below is respectable, not a face-saver.
**Secondary goal (BD / quiet):** signal the Externalis venture so the right people lean in — **without** leaking the sauce.
**What this is NOT:** not the research, not a truth detector, not a new method. It's a clever, well-instrumented application + honest validation.

---

## 3. Rules & constraints (do not violate)
- **Built entirely at the event. No extending prior work. Primarily your own.** Last night's spike is *private prior research* — carry forward the **findings, demo cases, and design**, NOT the code. Rebuild fresh in the public submission repo today.
- Public GitHub repo · <2-min demo video · 3-min live pitch (+2-min Q&A) · AGI House submission with the 4 fields.
- Team ≤ 5. Effective build window ≈ 5 hrs.
- No GPU needed (models via W&B Inference API; embeddings on CPU).
- Judging axes: **Agent Orchestration, Utility, Technical Execution, Creativity, Sponsor Usage.**
- Submission must list orchestration protocols (A2A, MCP) and every sponsor tool used.

**Sauce line (memorize):** Show the curves, the fidelity/cost frontier, gold accuracy. Withhold the typed, cognitive-level divergence against a curated reference. If asked "how would you diagnose *which* understanding broke, per learner, at *what* level?" → *"That's the venture — let's talk offline."*

---

## 4. Architecture
```
Source + Question
      │
   Explainer  (sees source + question → writes handoff memo)
      │
   Relay 1    (sees only prior memo → rewrites)         ← weak model; loss happens here
      │
   Relay 2    (sees only prior memo → rewrites)
      │
   Answerer   (sees only final memo → A/B/C/D)
      ▲
   Conductor  (holds source; after each hop: score risk; if high, inject question-relevant
               source chunk + repair the memo before it passes on; log everything to Weave)
```
**Optional stretch (more Anthropic-like, only if the chain works first):** Lead → parallel Researchers A/B/C → Synthesizer, with the Conductor watching each researcher→synthesizer hand-off. Defuses "is this a toy chain?" Describe the base chain honestly as a **minimal model organism of hand-off degradation**.

---

## 5. Signals (key-free; gold answers are for scoring ONLY)
1. **Question-conditioned drift (secondary, expect near-inert):** chunk the source, retrieve top-k chunks by the *question* embedding (all-MiniLM-L6-v2, local), `drift = 1 − cos(memo, those chunks)`. Compare to question-relevant chunks, NOT the whole source.
2. **Answer instability — the live signal (primary):** at each hop, a shadow-answerer answers the MCQ from the current memo (no gold) → choice; `flip = 1` if it changed from the prior hop.
3. **Continuous tunability:** a binary flip can't be tuned to a 30% budget (last night: 69%→47%, never hit target). Make risk **continuous** so it's tunable: `risk = flip + margin_drop`, where `margin_drop` is the **relay model's** top-2 logprob margin drop (NOT the answerer's — the answerer is saturated/overconfident; loss happens at the relay).
   - ⚠️ **logprobs may not be exposed by W&B Inference (unverified).** FIRST THING: test a call with `logprobs`/`top_logprobs`. If unavailable, fall back to the shadow-answerer's self-reported confidence drop, or run on flip + drift alone. Do NOT architect around logprobs until that test passes.

---

## 6. The four conditions (the rigor — random-at-budget is non-negotiable)
| Condition | What it proves | Cost | Fidelity |
|---|---|---|---|
| Naive relay | loss exists | low | low |
| Always-reground | upper bound (confirms something is recoverable) | high | high |
| **Random-at-budget** | re-grounding *itself* helps | = adaptive | modest |
| **Adaptive (Relay)** | the *signal* has decision value | = random | best per token |
**The whole project = adaptive vs random-at-budget.** Match random's intervention rate to adaptive's *observed* rate. Without it you only prove "re-grounding helps," which is trivial.

---

## 7. Sponsor / tool stack (settled; all verified against the W&B docs)
- **W&B Inference** — all model calls. Endpoint `https://api.inference.wandb.ai/v1` (OpenAI-compatible). Auth: `Authorization: Bearer <WANDB_API_KEY>` + header/param `project="<team>/<project>"`. Verified model IDs: `meta-llama/Llama-3.3-70B-Instruct` (explainer/answerer/conductor), `meta-llama/Llama-3.1-8B-Instruct` or `microsoft/Phi-4-mini-instruct` (weak relay), also `deepseek-ai/DeepSeek-R1-0528`, `deepseek-ai/DeepSeek-V3-0324`, `meta-llama/Llama-4-Scout-17B-16E-Instruct`. **429 = concurrency limit** → sequential calls + backoff.
- **W&B Weave** — `@weave.op()` traces every agent call; `weave.Evaluation` + an `exact_match` scorer; `weave.publish(leaderboard.Leaderboard(...))` for the 4-condition leaderboard; drift/flip/margin/intervention logged as custom scores (Signals). This is the **Best-Use-of-Weave** entry. *(The docs show this exact Evaluation+Leaderboard pattern — copy it.)*
- **W&B MCP (~20 tools)** — connect it so Claude Code `/goal` can read Weave traces/metrics during tuning.
- **Claude Code `/goal`** — the morning's tuning loop (Section 9).
- **MCP** — primary orchestration protocol: serve the source-grounding tool / benchmark loader / scorer as MCP servers (do this only if the core runs). **A2A** — claim ONLY if actually wired; otherwise the Conductor coordinates directly. Don't claim hypothetical protocols.
- **Dynamic workflows** — *optional build accelerator only*, for one bounded task if your plan supports it and the token burn is contained. **They CAN be traced in Weave** (correction to an earlier note). NOT Relay's architecture — they're parallel/breadth-first; Relay is sequential degradation. Default: skip.
- **Do NOT claim** Weave "guardrails that affect control flow" — unverified.

---

## 8. Data
- **Primary: QuALITY** (long passages → real ceiling, so there's a gap to recover). Sample N≈40. Verify exact HF path at the event.
- **Confirmed-working fallback: a hard RACE subset** (we ran RACE last night; it loads). Pick items with negations, numbers, ordering, exceptions, "which is NOT true."
- **Ceiling rule:** you need `a_full − a_naive ≥ ~0.20`. Last night's RACE gap was ~1 item at n=15 — too small to demonstrate anything. That's the whole reason for the QuALITY swap.
- Demo cases from the spike (race-1429, race-1508) stay local as your demo source — not in the public repo.

---

## 9. Claude Code `/goal` — four prompts, used in order (after the benchmark runs)
**1 — green the harness:**
> Improve the Relay experiment until `python run_all_conditions.py --n 20` completes and produces results.jsonl, leaderboard.md, frontier.png, and ≥1 Weave trace per condition. Do not add features or change scope. Only fix bugs and reliability. Stop when it passes twice in a row.

**2 — find a recoverable gap (if naive≈always):**
> Find a 15–25 item slice where always-reground beats naive by ≥15 points. Allowed: adjust dataset filtering, up to 4 hops, cap memo length, weaker relay model, pick questions with negations/quantities/ordering/exceptions. Not allowed: use gold answers in the runtime policy; hand-pick on final correctness without a logged rule; new architecture. Stop when the selection rule is in the README, the slice is saved, naive vs always are logged, and it reproduces from one command.

**3 — tune adaptive vs random (after four conditions run):**
> Tune the adaptive policy to a 25–35% intervention rate and beat random-at-budget on the current slice. Allowed: tune the risk threshold, adjust the risk formula over already-logged signals, improve relay/repair/shadow prompts, read Weave metrics to diagnose. Not allowed: inspect gold at runtime; remove random-at-budget; balloon N; add new signals if the pipeline is unstable. Stop when adaptive's rate is 25–35%, random uses the same observed rate, the leaderboard is regenerated, and the README states honestly whether adaptive won.

**4 — find the demo case:**
> From logged runs, find the clearest case where naive is wrong and adaptive/always is right, with a visible degradation point + risk spike + repair. Produce demo_case.md, demo_case_trace.json, and a <90s narration. Don't modify core logic.

Guardrail: `/goal` is a strengthener, not the MVP. Cap iterations, watch the credit burn.

---

## 10. Roles
| Role | Owns |
|---|---|
| **Lead / integrator** | loop architecture, the gate call, scope discipline, final assembly |
| **Data + eval** | QuALITY/RACE load + normalize, exact-match scorer, selection rule |
| **Signal** | the two signals, the logprob check, the risk trigger, `/goal` tuning |
| **Weave + infra** | env, client wrapper, tracing, Evaluation/leaderboard, viz, rate-limits, video |
| **Pitch + demo** | slide, 3-min pitch, Q&A, README, submission, the clock |
**Collapse:** 3 → (Lead+Data)(Signal+Weave)(Pitch+infra). 2 → split C/F/G/H/B vs E/I/J/L/M. **Never collapse the Lead.**

---

## 11. Hour-by-hour
| Time | Do | Target |
|---|---|---|
| **before 11:30** | rotate the exposed key; W&B account + credits form; **W&B MCP connected**; confirm QuALITY/RACE loads; create empty public repo; assign roles; **env smoke test (one traced W&B Inference call appears in Weave)**; **logprob test call** | stack verified |
| **11:30–1:00** | env skeleton; load data; relay chain (relay on a W&B Inference model); exact-match scorer; drift + shadow-answer instability; run the sanity gate | sanity result |
| **~1:00 GATE** | GREEN if `a_full − a_naive ≥ ~0.20` and `a_none` low → build Conductor. RED → QuALITY/harder slice, weaker relay, more hops, cap memo. Still red ~1:30 → fall back to RACE-with-levers + reframe | GO/NO-GO |
| **1:00–4:00** | Conductor + risk trigger; four conditions (incl. random-at-budget); Weave Evaluation + leaderboard + MCP; `/goal` tuning | 4-condition result |
| **4:30 CHECK-IN** | must have ≥ naive-vs-adaptive; if behind, **lock MVP, stop adding** | — |
| **4:00–6:00** | frontier plot, failure→repair demo case, <2-min video, 1 slide, rehearse 3-min + Q&A | demo ready |
| **6:00 / 7:00** | **STOP CODING.** README, 4 fields, sponsor list, submit draft | draft in |
| **7:00–8:00** | final polish + submit | final |

---

## 12. The pitch (3 min) + demo + Q&A
**Spoken core (~45s):** "Multi-agent systems lose information silently — and in production you can't see it, because you have no answer key. Relay makes it visible. A source passes through a chain of agents; a Conductor measures, *without the answer key*, how far each hand-off drifts from the question-relevant source and whether the answer the memo supports has started to change — and re-grounds only when the signal spikes. The point isn't that re-grounding helps — of course it does. The result is that *adaptive* re-grounding beats *random* at the same budget, moving toward always-grounded fidelity at near-relay cost."
**Closer (the BD signal — keep it, flag as analogy):** "This is the agentic analogue of what happens when a person passes along what an AI told them — same drift, measurable in a controlled relay. That's what we're building next."

**Demo = show the failure, not the evals:** source has a critical detail → Agent 1 keeps it → Agent 2 drops it → drift/instability spikes (shadow answer flips C→B) → Conductor injects the source sentence → memo repaired → answer back to C. Then the Weave trace, then the 4-bar leaderboard / frontier.

**Q&A:**
- *Why not always include the source?* Cost/latency/context/privacy — Anthropic's own data: multi-agent burns ~15× tokens. Relay is selective.
- *Why Sentence-BERT, isn't it weak?* Intentional — a cheap feature, not the judge; answer-instability makes it operational; we validate the whole risk model against gold.
- *How do you know drift means loss?* We don't assume it — we measure whether it predicts failure and whether acting on it beats random at the same budget.
- *Is this really multi-agent?* Yes — agents exchange compressed state and a Conductor observes, scores, and intervenes on intermediate state. Orchestration behavior is the point, not persona count.
- *Isn't this just Weave?* Weave is the observability/eval layer; Relay is a hand-off-fidelity control policy built on it — we score *intermediate inter-agent state* and prove the trigger beats random.
- *Dynamic workflows?* Complementary — they verify *parallel* subagent outputs at merge; we watch *sequential* hand-offs degrade in flight.

---

## 13. Fallback (if adaptive doesn't beat random — still respectable)
> "The harness made the failure visible. Without these traces and policy comparisons you'd never know which hand-off lost the answer. Relay's finding: simple drift is insufficient — answer-instability is the live signal — and the harness turns hand-off reliability into something measurable and optimizable." Lead with the failure→repair trace and the honest leaderboard.

---

## 14. Risks
| Risk | Mitigation |
|---|---|
| logprobs unavailable | test first; fall back to confidence / flip+drift |
| ceiling still too low on QuALITY | n≥40, hard items; else honest "loss exists, recovery undemonstrable at this n" |
| confounded result | random-at-budget is non-optional |
| `/goal` burns credits / won't converge | cap iterations; it's a strengthener, hand-tuned threshold still works |
| scope creep | cut list binding; stretch only after the four conditions run |
| over-claiming | "cheap feature, not a truth detector"; PoC framing; claim only protocols you wired |
| losing the BD signal | keep the Externalis closer + the named displacement signal |
| key exposed | rotate it; submission runs on W&B Inference anyway |

---

## 15. Cut list (do NOT build)
NLI/entailment signal · expert-vs-AI source experiment · Learning Commons · typed cognitive-level divergence (the withheld sauce) · any training · custom UI · >4 hops · >~30–40 items · learned risk weights beyond /goal's knobs · custom MCP servers unless core is done · A2A unless wired · LAB-Bench unless everything else is done · OpenAI dependence in the submission.

---

## 16. Definition of Done
- [ ] Public repo built fresh at the event (README with run command + the 4 conditions)
- [ ] logprob availability tested; signal path chosen
- [ ] Sanity gate GREEN (real ceiling); relay on a W&B Inference model
- [ ] Four-condition run: accuracy + cost + interventions + drift + instability
- [ ] Weave: traces + Evaluation + leaderboard + custom signals + MCP connected
- [ ] `/goal`-tuned trigger (or honest best-config report)
- [ ] Fidelity-vs-cost frontier (adaptive vs random at same budget)
- [ ] Demo video < 2:00 (failure→repair) + 1 slide + rehearsed pitch + Q&A
- [ ] AGI House submission: team, members+socials, repo, video, 4-field description, sponsor-tool list
- [ ] Externalis closer + Anthropic-telephone + AI-Organizations citations in the pitch
