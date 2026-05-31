# Relay — a multi-agent handoff-degradation harness

Relay is a runtime observability harness — a flight recorder + circuit breaker —
for **lossy multi-agent hand-offs**. A workflow relays a structured document
through a chain of reversible edits. A **Conductor** estimates each step's
*handoff-degradation risk* from cheap, **key-free** signals and selectively
performs a **targeted repair** only when risk spikes. We then validate whether
that signal actually has decision value by comparing four conditions over the
same tasks:

| Condition | What it proves | Cost | Fidelity |
|---|---|---|---|
| `naive` | loss exists | low | low |
| `always_reground` | recovery is possible (upper bound) | high | high |
| `random_at_budget` | re-grounding *itself* helps | = adaptive | modest |
| `adaptive` | **the signal has decision value** | = random | best per token |

**The whole project is `adaptive` vs `random_at_budget`** — at the *same* number
of interventions, does the cheap risk signal pick *better moments* than chance?

---

## The task: round-trip reconstruction

Generate a structured JSON document (15–20 records, stable IDs, nested fields,
numeric values, known totals), transform it with **three reversible edit pairs**
over several round trips, then reverse them — a faithful workflow reconstructs
the seed. The Conductor watches a **key-free checksum** after every edit and
performs a **targeted repair** (restore the flagged invariants, *preserve* the
legitimate edit — never just reset to the seed) only when risk fires. All four
conditions run on the same tasks / edits / round-trips; only the policy changes.

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

---

## CLI

```
python run_all_conditions.py \
    [--n 20] [--conditions naive always_reground adaptive random_at_budget] \
    [--num-round-trips 4] [--threshold 0.008] [--slip-p 0.6] \
    [--rng-seed 42] [--random-seed 123] [--mock | --live] [--out-dir outputs]
```

| Flag | Meaning |
|---|---|
| `--n` | number of tasks |
| `--conditions` | subset of `naive always_reground adaptive random_at_budget` |
| `--num-round-trips` | forward+backward edit pairs per task |
| `--threshold` | adaptive re-grounds when `risk > threshold` (see note below) |
| `--slip-p` | (mock) probability an edit loses something |
| `--rng-seed` / `--random-seed` | task generation / random-at-budget decisions |
| `--mock` / `--live` | force the mock editor / real W&B Inference (else auto-detect) |

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
- `run_all_conditions.py` — the entry point: runs the four conditions, writes the
  outputs above, applies the gate, and (creds-only) publishes a `weave.Evaluation`
  + 4-condition `Leaderboard` by **looking up already-computed scores** (no re-run).

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
scoring — no signal touches them, so nothing is circular.

## Install (only for the real path)

```bash
pip install -r requirements.txt   # openai + weave + wandb
```
