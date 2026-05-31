# Relay × DELEGATE52 (accounting)

A runtime handoff-fidelity harness for **delegated document editing**. We relay a real
accounting ledger through a chain of LLM edits (a "round-trip" telephone game), measure how
much content silently corrupts, and let a **Conductor** selectively **re-ground** the
document only when a cheap key-free signal says corruption spiked. We then compare adaptive
re-grounding against naive / always-reground / a budget-matched random control on a
fidelity-vs-cost frontier — logged in W&B Weave.

This is the [`relay-main`](../relay-main) QuALITY/MCQ harness ported to documents, with a
**new deterministic scoring technique**: instead of grading a multiple-choice answer, we
parse the ledger and score round-trip fidelity ∈ [0,1] against the seed document (gold).

## The idea
- **Dataset:** [microsoft/DELEGATE52](https://huggingface.co/datasets/microsoft/delegate52),
  `accounting` domain — hledger plain-text ledgers (~64–98 transactions, ~3–4k tokens).
- **Round-trip = one relay hop:** a *forward* edit (e.g. "split this ledger by person") then
  its *inverse* (e.g. "merge these files into one ledger sorted by date"). A faithful
  round-trip returns the seed exactly, so `score(boundary, seed)` has a true 1.0 ceiling.
- **Chain:** several round-trips applied in sequence, each fed the previous output, so
  corruption **compounds** (the paper's "25% corruption after 20 interactions" regime).
- **Conductor:** holds the seed `D0`, measures key-free signals at each round-trip boundary
  (where the doc *should* equal `D0`), and re-grounds (re-injects the exact transactions that
  went missing + a repair pass) only when risk crosses a threshold.

## Signals (key-free; gold is used only by the scorer)
1. **`invariant_deviation`** (primary, live) — continuous structural deviation from `D0`:
   transaction count, payee/account/amount multisets. Tunable to an intervention budget.
   *(The document analog of relay-main's answer-instability — and not saturated.)*
2. **`parse_health`** (live backstop) — did the document stop parsing as a ledger?
3. **`embedding_drift`** (logged, ~inert) — `1 − cos(doc, D0)` via MiniLM. Weight ~0,
   reproducing relay-main's "embedding drift is near-inert" finding.

`risk = w_inv·invariant_deviation + w_health·(1−parse_health) + w_drift·embedding_drift`

## The four conditions (the experiment)
| Condition | Proves |
|---|---|
| naive | corruption exists & compounds |
| always_reground | upper bound — corruption is recoverable |
| random_at_budget | re-grounding itself helps |
| **adaptive** | the signal picks better re-grounding moments than chance, at equal budget |

The whole point is **adaptive vs random at the same intervention count**.

## Selection rule (what goes in a chain)
Only **structurally-lossless** edit families, so every boundary has a true 1.0 ceiling: a
forward edit is eligible iff its target state is not a `cpx_` (complex multi-file) state and
its inverse edit's `semantic_operations ⊆ {split_and_merge, classification, sorting,
format_knowledge}`. This keeps split-by-X→merge and csv/beancount round-trips; it excludes
FX conversion (rounding caps a perfect editor at ~0.94), website export, account-flattening,
and prose rewrites — which would confound agent corruption with format loss.

## Models (W&B Inference, OpenAI-compatible)
- **Editor** (`RELAY_EDITOR_MODEL`, default `meta-llama/Llama-3.1-8B-Instruct`): the relay
  agent where corruption happens — fast, non-reasoning, weak so loss is visible.
- **Conductor/repair** (`RELAY_BIG_MODEL`, default `meta-llama/Llama-3.3-70B-Instruct`).
- ⚠️ **`openai/gpt-oss-120b` was tested and rejected as the editor:** it is a *reasoning*
  model that spends the entire token budget on hidden reasoning (empty content unless
  `reasoning_effort=low`) and runs 60–140s/call — impractical for a many-call harness. It is
  still selectable via `RELAY_EDITOR_MODEL=openai/gpt-oss-120b` (low effort auto-applied).

## Run
```bash
pip install -r requirements.txt
export WANDB_API_KEY=...                      # W&B Inference + Weave
export WANDB_PROJECT=entity/relay-delegate52  # enables Weave tracing + leaderboard

# 1) sanity gate (naive vs always) — confirms a recoverable corruption gap
python run_gate.py --n 12 --depth 4

# 2) the four-condition experiment + Weave Evaluation/leaderboard + frontier.png
python run_conditions.py --n 12 --depth 4 --target-rate 0.30

# offline scorer self-test (no API): score(D0,D0)=1.0, corruption drops it
python ledger.py
```

### Offline mode (`RELAY_MOCK=1`) — validate the whole pipeline in seconds, no inference API
The only slow part is the LLM editor (~40s/call on W&B Inference). `RELAY_MOCK=1` swaps it for a
deterministic local editor that simulates lossy editing (drops/alters transaction blocks at a
per-edit rate, so corruption varies across boundaries and the risk signal has decision value).
The entire four-condition eval then runs in **~2s**, still publishing to Weave if `WANDB_PROJECT`
is set. Use it to validate the machinery; drop the flag for real LLM-corruption numbers.
```bash
# full four-condition eval, offline, pushed to W&B (runs in seconds)
RELAY_MOCK=1 WANDB_PROJECT=entity/relay-delegate52 python run_conditions.py --n 12 --depth 4
```
Mock runs are tagged `*-mock` on W&B and print a banner, so synthetic results are never confused
with real ones.

Outputs: `results.jsonl` (per-boundary log), `leaderboard.md`, `frontier.png`, `demo_case.md`,
and (with `WANDB_PROJECT` set) a Weave Evaluation + published Leaderboard.

## Domains (`RELAY_DOMAIN`)
The pipeline is domain-agnostic; `domain.py` dispatches the parser/scorer/prompts/mock by
`RELAY_DOMAIN` (set it in the env before launching — it's read at import):
- **`accounting`** (default) — hledger ledgers; the *transactions* are the content. Scorer in
  `ledger.py`; chains restricted to lossless split/merge round-trips.
- **`chess`** — PGN games; the *move sequence* is the content (annotations are surface form, so
  scoring on moves is robust to annotation-format round-trips). Scorer in `chess_domain.py`.
```bash
RELAY_DOMAIN=chess RELAY_MOCK=1 WANDB_PROJECT=entity/relay-delegate52 \
  python run_conditions.py --n 10 --depth 4        # chess, offline, pushed to W&B
```
Adding a domain = one module exposing `parse_doc, invariants, invariant_deviation, parse_health,
score, diff_blocks, editor_system_prompt, repair_system_prompt, repair_source_label, mock_edit,
mock_reground` + a branch in `domain.py`. Each domain's W&B runs are tagged `…-<domain>[-mock]`.

## Files
`domain.py` (domain dispatcher) · `ledger.py` (accounting scorer/invariants/diff) ·
`chess_domain.py` (chess move-sequence scorer) · `data.py` (round-trip chains, domain-aware) ·
`signals.py` (key-free risk) · `agents.py` (editor + reground + mock) · `conductor.py` (per-chain
policy + boundary scoring) · `llm.py` (W&B Inference client + Weave) · `run_gate.py` ·
`run_conditions.py`.

## Honest scope
Proof of concept over a small slice; descriptive results, no p-values on tiny n. The signals
are cheap runtime features, not a corruption oracle — the claim is operational: *does acting
on the signal beat random at the same budget?* We are **not** running a model-vs-model
leaderboard (that is the DELEGATE52 paper's job); the model is roughly fixed and we study the
handoff degradation and the Conductor's decision value.

## Result

**Offline pipeline validation** (`RELAY_MOCK=1`, synthetic corruption, n=12 chains · depth 4) —
proves the full machinery and the four-condition shape end-to-end:

| Condition | Fidelity | Avg interventions | Avg tokens |
|---|---|---|---|
| naive | 0.127 | 0.00 | 5252 |
| always_reground | 0.934 | 4.00 | 11394 |
| **adaptive** | **0.711** | 0.83 | 6498 |
| random_at_budget | 0.452 | 0.83 | 6628 |

**adaptive − random = +0.259** at equal intervention budget (10 clean adaptive-fixes-naive chains).
Published to Weave: `…/weave/leaderboards/Relay-DELEGATE52-accounting-mock`.

> These are MOCK numbers — they validate the pipeline, signals, and four-condition comparison,
> not real LLM corruption. The real result comes from the same command **without** `RELAY_MOCK`
> (editor = Llama-3.3-70B on W&B Inference; ~40s/call). The single-round-trip real check already
> confirmed genuine corruption (one 70B round-trip → fidelity 0.821).
