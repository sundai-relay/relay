# Relay — a multi-agent handoff-degradation harness

Relay is a runtime observability harness — a flight recorder + circuit breaker —
for **lossy multi-agent hand-offs**. A chain of agents relays a piece of state
hop to hop. A **Conductor** estimates each hop's *handoff-degradation risk* from
cheap, **key-free** signals and selectively **re-grounds** the chain only when
risk spikes. We then validate whether that signal actually has decision value by
comparing four conditions over the same episodes:

| Condition | What it proves | Cost | Fidelity |
|---|---|---|---|
| `naive` | loss exists | low | low |
| `always` | recovery is possible (upper bound) | high | high |
| `random_at_budget` | re-grounding *itself* helps | = adaptive | modest |
| `adaptive` | **the signal has decision value** | = random | best per token |

**The whole project is `adaptive` vs `random_at_budget`** — at the *same* number
of interventions, does the cheap risk signal pick *better moments* than chance?

---

## Quickstart (NO API key)

The default `mock` substrate is pure-stdlib — no key, no `pip install`, no
network. This is the acceptance run:

```bash
python run.py --substrate mock --condition all --n 5
```

It prints a 4-row leaderboard and writes per-hop rows to `outputs/results.jsonl`.
Expected ordering (degradation is real, recovery works, and the signal beats
random at equal budget):

```
naive  <  random_at_budget  <  adaptive  <≈  always
```

A single condition:

```bash
python run.py --substrate mock --condition adaptive --n 10
python run.py --substrate mock --condition naive    --n 10
```

> `--condition random` (and `all`) run `adaptive` first to learn the per-episode
> intervention budget that `random_at_budget` must match.

---

## The round-trip build (`run_all_conditions.py`)

The headline task. Generate a structured JSON document (15–20 records, stable
IDs, nested fields, numeric values, known totals), transform it with **three
reversible edit pairs** over several round trips, then reverse them — a faithful
workflow reconstructs the seed. The Conductor watches a **key-free checksum**
after every edit and performs a **targeted repair** (restore the flagged
invariants, *preserve* the legitimate edit — never just reset to the seed) only
when risk fires. Runs the four conditions on the same tasks/edits/round-trips;
only the policy changes.

```bash
# Offline, NO key — deterministic mock editor (lossy) + competent repairer:
python run_all_conditions.py --n 20

# The naive-vs-always gate (build adaptive only once there's a gap to recover):
python run_all_conditions.py --n 5 --conditions naive always_reground
#   GREEN  always - naive >= 0.15   (build the Conductor)
#   YELLOW 0.05 .. 0.15
#   RED    < 0.05                    (more round trips / higher --slip-p / harder edits)

# With real models (W&B Inference) + Weave tracing:
export WANDB_API_KEY=...  WANDB_PROJECT=entity/project
python run_all_conditions.py --n 20 --live
```

Outputs to `outputs/`: `results.jsonl` (per-step trace), `leaderboard.md`,
`frontier.png` (cost vs fidelity; `frontier.txt` if matplotlib is absent),
`demo_case.md` (the clearest naive-loses / adaptive-recovers task).

Modules (`relay/roundtrip/`): `tasks.py` (procedural docs + edit pairs),
`agents.py` (`apply_edit` / `repair_doc` — real W&B Inference or the mock),
`checksum.py` (`runtime_risk`), `scorer.py` (`final_structural_score`),
`runner.py` (one task under one policy). The intervention rule is
`conductor.should_intervene(...)`.

> **Threshold note:** the checksum risk is *small-magnitude* (one dropped record
> among ~17 is `id_loss ≈ 0.06`, so `risk ≈ 0.02`), not a 0–1 confidence. The
> round-trip default `--threshold 0.008` is tuned for ~25–35% interventions.
> This is the `/goal` knob to tune; **the final structural score never drives
> the runtime policy.**

---

## CLI

```
python run.py --substrate {mock,roundtrip,mcq} \
              --condition {naive,always,adaptive,random,all} \
              --n 5 \
              [--threshold 0.4] [--hops N] [--corruption P] [--seed 0] \
              [--out outputs/results.jsonl]
```

| Flag | Meaning |
|---|---|
| `--substrate` | which task (see below) |
| `--condition` | which re-grounding policy/policies to run |
| `--n` | number of episodes |
| `--threshold` | adaptive re-grounds when `risk > threshold` |
| `--hops` / `--corruption` | (mock only) hops per episode / per-hop corruption prob |
| `--seed` | reproducibility |

---

## Environment variables (real substrates only)

```bash
export WANDB_API_KEY=...            # from wandb.ai/settings  (NEVER commit it)
export WANDB_PROJECT=entity/project # your W&B entity/project
```

- **Mock mode ignores both** — it always runs.
- With both set, Weave initializes and traces every `apply_hop` / `risk` /
  `reground` / `score` / `run_condition`, and publishes a `weave.Evaluation`
  + a 4-condition `Leaderboard`.
- Without them (or without `weave` installed) tracing is a transparent no-op and
  the local leaderboard + JSONL are still produced.

W&B Inference is OpenAI-compatible:
`base_url=https://api.inference.wandb.ai/v1`, auth `Bearer $WANDB_API_KEY`,
project sent as the `OpenAI-Project` header. Models:
`meta-llama/Llama-3.3-70B-Instruct` (strong) /
`meta-llama/Llama-3.1-8B-Instruct` (weak). HTTP 429 = concurrency limit →
sequential calls with exponential backoff (handled in `relay/wandb_client.py`).

---

## Outputs

- `outputs/results.jsonl` — one row per hop:
  `{episode_id, condition, hop, risk, intervened, score, token_proxy}`
  (`score` is the running fidelity of the state after that hop; the last hop's
  score is the episode's final fidelity).
- A printed 4-row leaderboard: `condition | mean_score | avg_interventions | avg_token_proxy`.

---

## Architecture (substrate-agnostic)

Swap the task with one `--substrate` flag. Everything goes through two interfaces
(`relay/core.py`):

```python
class Episode:    id; initial_state(); hops() -> list; reference(); score(final_state) -> float
class Substrate:  load_episodes(n) -> [Episode]
                  apply_hop(state, hop, grounding=None) -> new_state   # may degrade; repairs if grounded
                  risk(state_before, state_after, episode) -> float    # key-free, in [0,1]
                  reground(episode) -> grounding                       # the source slice
```

- `relay/conductor.py` — the `Conductor` + the four policies.
- `relay/conditions.py` — the per-hop run loop + `run_conditions` (derives
  random's budget from adaptive's *observed* interventions).
- `relay/leaderboard.py` — aggregation, the printed table, JSONL.
- `relay/weave_compat.py` — `@op()` works with or without `weave`; `weave.init`
  only fires with real creds.
- `relay/weave_leaderboard.py` — the creds-only `Evaluation` + `Leaderboard`.

---

## Substrates

### `mock` — built first, no API (fully implemented)
State is a dict of N integer "facts". Each hop has a **deterministic** corruption
schedule (seeded by episode + hop), so the corruption pattern is **identical
across all four conditions** — they differ only in *when* they re-ground, which
is what makes adaptive-vs-random a fair test. `risk()` is a noisy proxy of the
before/after change; re-grounding repairs the current hop's damage (local fix);
`score()` is the fraction of facts still matching the reference. Tuned so the
expected ordering above is visible. Use this to get the whole pipeline + four
conditions + Weave + leaderboard GREEN before any real API call.

### `roundtrip` — fully implemented (see `relay/roundtrip/` + `run_all_conditions.py`)
State is a JSON doc; hops are (forward, backward) **reversible** edit pairs over
several round trips; a faithful round trip reconstructs the seed. `risk()` =
checksum drift (`runtime_risk`), `score()` = `final_structural_score` (parse
validity + id F1 + key-path F1 + scalar value fidelity + aggregate invariants),
`reground()` = the seed, and `apply_hop` calls `apply_edit` / targeted
`repair_doc` (real W&B Inference, or a deterministic mock with no key). The
substrate adapter (`relay/substrates/roundtrip.py`) lets `run.py --substrate
roundtrip` drive it; `run_all_conditions.py` is the dedicated entry with the
gate, frontier, and demo case (see the section above).

### `mcq` — thin stub, fallback (`relay/substrates/mcq.py`)
State is a handoff memo; hops are relay rewrites; `score` is exact-match of a
shadow-answer to the gold MCQ letter. Deterministic scoring scaffold + a
drift-based `risk` proxy are in place; the model calls (Explainer / relay /
shadow-answerer) and the QuALITY/RACE loader are `TODO(teammate)`.

---

## Notes / honest scope

The signals are **cheap runtime features, not a factuality detector**. The
scientific claim is operational: *does this risk model pick better re-grounding
moments than random at the same budget?* Gold answers are used **only** for final
scoring — no signal touches them, so nothing is circular.

## Install (only for the real path)

```bash
pip install -r requirements.txt   # openai + weave
```
