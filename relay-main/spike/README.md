# Relay — feasibility spike (THROWAWAY)

Disposable feasibility test. **Not the hackathon submission.** The submission is
rebuilt fresh at the event; only the *findings + design notes* from this spike
carry forward, never this code.

## The one question
Does **adaptive** re-grounding (driven by cheap key-free signals) beat **random**
re-grounding at the **same intervention budget**?

## Pipeline
`Explainer (sees source) → Relay1 → Relay2 → Answerer`. Each relay sees only the
previous memo + question + choices. Gold answers are used **only** for scoring.

Two key-free signals per hop:
- `question_conditioned_drift` — `1 - cos(memo, top-3 question-relevant source chunks)` (MiniLM)
- `answer_instability` — shadow answerer's choice flips vs previous hop
- `risk = drift_delta + answer_instability`

## Run order
```bash
pip install -r requirements.txt
export WANDB_API_KEY=...          # W&B Inference (OpenAI-compatible)
# optional: export WANDB_PROJECT=entity/project

python llm.py            # 0. smoke test: confirms model calls work
python relay_sanity.py   # 2. the GATE — does naive relay lose accuracy?  (stop if RED)
python probe.py          # 3. four-condition probe (only if gate is GREEN)
```

Models default to `meta-llama/Llama-3.3-70B-Instruct` (answering/explainer) and
`meta-llama/Llama-3.1-8B-Instruct` (relay, to induce more loss). Override via
`RELAY_BIG_MODEL` / `RELAY_SMALL_MODEL`. Endpoint via `WANDB_BASE_URL`.

## Decision rule
**GREENLIGHT** the full build if: `naive < always`, `adaptive > random`,
`adaptive` uses fewer interventions than `always`, and ≥1 clean demo case exists.
Noisy numbers fine at this n.

**REFRAME** if: no degradation, or `adaptive ≯ random`, or no vivid case → honest
pitch becomes "an evaluation harness showing drift alone is insufficient without
an answer-instability/coverage check." If adaptive only ties random, lean on
answer-instability over drift, or add a coverage signal.

## What was verified offline (no model access in the build container)
- Signal pipeline runs on real embeddings: faithful memo drift << lossy memo drift.
- RACE loads 18 well-formed passage-MCQ items (~275-word passages).
- The W&B Inference endpoint was **blocked** from the build container (Cloudflare
  1010); the LLM half (`relay_sanity.py`, `probe.py`) must be run where the
  endpoint is reachable.

## Files
- `data.py` — RACE loader (+ tiny offline fallback)
- `signals.py` — chunking, retrieval, MiniLM drift, risk
- `llm.py` — W&B Inference client + smoke test
- `agents.py` — explainer / relay / answerer / shadow / reground prompts
- `relay_sanity.py` — the degradation gate (step 2)
- `probe.py` — the four-condition probe (step 3) → `relay_log.jsonl`, `demo_cases.jsonl`
