# Relay — selective intervention for lossy multi-agent hand-offs

> Built for the **Multi-Agent Orchestration Build Day** (The Engine, Cambridge ·
> AGI House × W&B × TNT × SundAI × E14). Benchmark: **Microsoft Research
> [DELEGATE52](https://huggingface.co/datasets/microsoft/delegate52)** — round-trip
> document editing under delegation.

Relay is a runtime observability harness — a flight recorder + circuit breaker —
for **lossy multi-agent hand-offs**. A workflow relays a structured document
through a chain of reversible edits. A **Conductor** estimates each step's
*handoff-degradation risk* from cheap, **key-free** signals and selectively
performs a **targeted repair** only when risk spikes.

**The pitch.** Current practice for keeping an agent chain faithful is to dump
the full source context into every hop — accurate but expensive, and it scales
its cost with the chain. Relay's claim is that **selective intervention** —
re-grounding *only* the hops where a cheap signal says corruption spiked —
captures most of the fidelity of always-regrounding at a fraction of the cost.
We prove it by comparing four conditions over the same tasks:

| Condition | What it proves | Cost | Fidelity |
|---|---|---|---|
| `naive` | loss exists | low | low |
| `always_reground` | recovery is possible (upper bound — today's SOTA "dump full context") | high | high |
| `random_at_budget` | re-grounding *itself* helps | = adaptive | modest |
| `adaptive` | **the signal has decision value** | = random | best per token |

**The whole project is `adaptive` vs `random_at_budget`** — at the *same* number
of interventions, does the cheap risk signal pick *better moments* than chance?
Adaptive **must beat random** for the signal to count as a legitimate
improvement (beating naive alone would just prove that re-grounding helps, which
is obvious). We don't claim a factuality detector; the scientific claim is
operational — *does this risk model pick better re-grounding moments than random
at the same budget?* Gold answers are used **only** for final scoring, so
nothing is circular.

---

## What's in this repo

Two harnesses share the same four-condition experiment and the same
adaptive-vs-random thesis:

- **`relay/` + `run_all_conditions.py`** — the portable, **domain-agnostic**
  harness. It generates a synthetic structured JSON document and relays it
  through reversible edits. Pure-stdlib mock editor, **no API key required** —
  this is the fastest way to see the whole pipeline and the leaderboard.
- **`delegate_relay/`** — the harness **specialized to DELEGATE52**. It relays a
  *real* document (an accounting ledger, or a chess PGN) through structurally
  reversible edit pairs, scores round-trip fidelity ∈ [0,1] against the seed,
  and runs the same four conditions on W&B Inference + Weave. See
  [`delegate_relay/README.md`](delegate_relay/README.md).

> **See also the [`amar`](../../tree/amar) branch — per-domain customized evals.**
> The default harness runs the *same* generic eval content across every domain.
> The `amar` branch instead **customizes the eval per domain** (accounting
> ledgers vs. chess move sequences) — a ledger-aware scorer/signal set and a
> PGN-aware one — and gives the intervening **Conductor agent its own
> domain-specific eval** rather than one shared notion of "corruption." Adding
> those individual per-domain evals **measurably improves** the selective
> intervention: on the chess domain, adaptive reaches **0.896** fidelity vs
> random's **0.761** at the same budget (**+0.135**), while always-reground
> costs ~60% more tokens for a 1.000 ceiling. That result is the evidence for
> our broader vision (below): tailoring evals per domain compounds the gains of
> selective intervention.

### Vision (where this goes)

Much like Google's AlphaGo/AlphaProof line selects the appropriate solver per
domain, we expect even larger gains if the system **selects domain-appropriate
eval methods** — the mapping of solve/repair tactics to the problem at hand —
rather than applying one generic notion of fidelity everywhere. The `amar`
branch is the first datapoint: tailoring the eval to the two test domains we
chose (Chess and Auditing/accounting) produced dramatic improvements, supporting
that a customized per-domain eval suite further advances selective intervention
and leads to coordinated agent swarms with better grounding faithfulness at
lower cost.

---

## The task: round-trip reconstruction

Generate a structured document (the portable harness: 15–20 JSON records, stable
IDs, nested fields, numeric values, known totals; the DELEGATE52 harness: a real
ledger or PGN), transform it with **reversible edit pairs** over several round
trips, then reverse them — a faithful workflow reconstructs the seed. The
Conductor watches a **key-free checksum** after every edit and performs a
**targeted repair** (restore the flagged invariants, *preserve* the legitimate
edit — never just reset to the seed) only when risk fires. All four conditions
run on the same tasks / edits / round-trips; only the policy changes.

---

## Quickstart (NO API key)

The mock editor is pure-stdlib — no key, no `pip install`, no network. Pass
`--mock` to force it (this is the acceptance run):

```bash
python run_all_conditions.py --mock --n 5
```

It prints the 4-row leaderboard, writes a per-step trace to
`outputs/results.jsonl`, and emits the naive-vs-always gate. Expected ordering
(degradation is real, recovery works, and the signal beats random at equal
budget):

```
naive  <  random_at_budget  <  adaptive  <≈  always_reground
```

> Without `--mock` or `--live`, the editor mode is **auto-detected from
> `WANDB_API_KEY`**: no key → mock; key present → LIVE (which needs
> `pip install openai`). Use `--mock` when a key is in your environment but you
> want the offline path.

The naive-vs-always gate on its own (build adaptive only once there's a gap to
recover):

```bash
python run_all_conditions.py --mock --n 5 --conditions naive always_reground
#   GREEN  always - naive >= 0.15   (build the Conductor)
#   YELLOW 0.05 .. 0.15
#   RED    < 0.05                    (more round trips / higher --slip-p / harder edits)
```

With real models (W&B Inference) + Weave tracing:

```bash
export WANDB_API_KEY=...  WANDB_PROJECT=entity/project
python run_all_conditions.py --n 20 --live
```

### DELEGATE52 harness (`delegate_relay/`)

```bash
cd delegate_relay
pip install -r requirements.txt
export WANDB_API_KEY=...                       # W&B Inference + Weave
export WANDB_PROJECT=entity/relay-delegate52

# offline, no inference API — validates the whole pipeline in seconds
RELAY_MOCK=1 python run_conditions.py --n 10 --depth 4 --target-rate 0.30
# pick the domain (default accounting):
RELAY_DOMAIN=chess RELAY_MOCK=1 python run_conditions.py --n 10 --depth 4
```

---

## CLI (`run_all_conditions.py`)

```
python run_all_conditions.py \
    [--n 20] [--conditions naive always_reground adaptive random_at_budget] \
    [--num-round-trips 4] [--threshold 0.008] [--slip-p 0.6] \
    [--rng-seed 42] [--random-seed 123] [--mock | --live] \
    [--provider wandb|openai] [--workers N] [--no-weave] [--out-dir outputs]
```

| Flag | Meaning |
|---|---|
| `--n` | number of tasks |
| `--conditions` | subset of `naive always_reground adaptive random_at_budget` |
| `--num-round-trips` | forward+backward edit pairs per task |
| `--threshold` | adaptive re-grounds when `risk > threshold` (see note below) |
| `--slip-p` | (mock) probability an edit loses something |
| `--rng-seed` / `--random-seed` | task generation / random-at-budget decisions |
| `--mock` / `--live` | force the mock editor / real inference (else auto-detect) |
| `--provider` | LIVE backend: `wandb` (default) or `openai` (`$OPENAI_API_KEY`, `gpt-4o-mini`) |
| `--workers` | parallel tasks per condition (forced to 1 for `wandb`, which 429s on concurrency) |
| `--no-weave` | skip Weave tracing (faster; recommended with `--workers`) |

> Adaptive runs before `random_at_budget` so random can match adaptive's
> *observed* per-task intervention rate.

> **Threshold note:** the checksum risk is *small-magnitude* (one dropped record
> among ~17 is `id_loss ≈ 0.06`, so `risk ≈ 0.02`), not a 0–1 confidence. The
> default `--threshold 0.008` is tuned for ~25–35% interventions. This is the
> knob to tune; **the final structural score never drives the runtime policy.**

---

## Outputs (`outputs/`)

- `results.jsonl` — one row per edit step:
  `{task_id, condition, round_trip_index, step_type, runtime_risk, risk_after,
  intervened, cost_proxy, final_score, ...}` (`final_score` is the task's final
  structural score, stamped on every row of that task; `0.0` means the final doc
  did not parse).
- `task_summary.jsonl` — one row per (condition, task) carrying the full
  `final_doc` (which the lean `results.jsonl` strips), so the leaderboard
  publisher can re-score sub-metrics from disk without re-running the workflow.
- `leaderboard.md` — the 4-row comparison:
  `condition | avg_score | avg_interventions | intervention_rate | cost_proxy | score_per_cost`.
- `frontier.png` — cost vs fidelity (`frontier.txt` if matplotlib is absent).
- `demo_case.md` — the clearest naive-loses / adaptive-recovers task.

`gate_read.py` reads `outputs/results.jsonl` (never runs a model, never writes)
and prints the always−naive delta, the per-task spread, adaptive-vs-random, and
a GREEN / YELLOW / RED verdict:

```bash
python gate_read.py
```

---

## Modules (`relay/`)

- `roundtrip/tasks.py` — procedural seed-doc + edit-pair generation (fixed seeds).
- `roundtrip/agents.py` — `apply_edit` / `repair_doc` (real W&B Inference, or a
  deterministic mock with no key).
- `roundtrip/checksum.py` — `runtime_risk`, the Conductor's key-free warning light.
- `roundtrip/scorer.py` — `final_structural_score` (post-run only; never drives
  policy): parse validity + id F1 + key-path F1 + scalar value fidelity +
  aggregate invariants.
- `roundtrip/runner.py` — run one task under one policy.
- `conductor.py` — `should_intervene(...)`, the functional re-grounding rule.
- `weave_compat.py` — `@op()` works with or without `weave`; `weave.init` only
  fires with real creds.
- `weave_leaderboard.py` — publishes a Weave **Evaluation + Leaderboard** over the
  four policies (`structural_score` + `intervention_rate` + `cost` columns) by
  **pure lookup** of already-computed results — no workflow re-run, no credits.
  Also a standalone CLI: `python -m relay.weave_leaderboard --results outputs/results.jsonl`.
  The `outputs/leaderboard.md` table is always written, with or without `weave`.
- `run_all_conditions.py` — the entry point: runs the four conditions, writes the
  outputs above, applies the gate, and (creds-only) auto-publishes via
  `weave_leaderboard`.

For the DELEGATE52-specific modules (the per-domain `domain.py` dispatcher,
`ledger.py`, `chess_domain.py`, `signals.py`, `conductor.py`, `run_conditions.py`,
`run_gate.py`) see [`delegate_relay/README.md`](delegate_relay/README.md).

---

## Environment variables (real path only)

```bash
export WANDB_API_KEY=...            # from wandb.ai/settings  (NEVER commit it)
export WANDB_PROJECT=entity/project # your W&B entity/project
```

- **`--mock` ignores both** — it always runs offline.
- With both set (and `--live` or auto-detect), Weave initializes and traces every
  `apply_edit` / `repair_doc` / `runtime_risk` / `final_structural_score`, and
  publishes a `weave.Evaluation` + a 4-condition `Leaderboard`.
- Without them (or without `weave` installed) tracing is a transparent no-op and
  the local leaderboard + JSONL are still produced.

W&B Inference is OpenAI-compatible:
`base_url=https://api.inference.wandb.ai/v1`, auth `Bearer $WANDB_API_KEY`,
project sent as the `OpenAI-Project` header. Models:
`meta-llama/Llama-3.3-70B-Instruct` (strong) /
`meta-llama/Llama-3.1-8B-Instruct` (weak). HTTP 429 = concurrency limit →
sequential calls with exponential backoff (handled in `relay/wandb_client.py`).

---

## Notes / honest scope

The signals are **cheap runtime features, not a factuality detector**. The
scientific claim is operational: *does this risk model pick better re-grounding
moments than random at the same budget?* Gold answers are used **only** for final
scoring — no signal touches them, so nothing is circular. Results are descriptive
over a small slice (no p-values on tiny n) — a proof of concept that selective
intervention beats budget-matched random, and that per-domain evals (the `amar`
branch) push it further.

## Install (only for the real path)

```bash
pip install -r requirements.txt   # openai + weave + wandb
```
