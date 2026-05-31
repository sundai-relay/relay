# Relay × DELEGATE52 — real round-trip results (accounting + chess)

Live W&B Inference, editor `microsoft/Phi-4-mini-instruct`, real samples from
`delegate/delegate52.jsonl` (6 accounting ledgers, 5 chess PGNs), depth 2
(forward+inverse compounding), all four conditions on the same tasks/edits.
Adaptive threshold auto-set from the naive-risk quantile (target 30%); the
gold/structural score never drives the runtime policy. Parsers pass
`fid_identity == 1.0` on every seed.

## Accounting (6 ledgers)

| condition | avg_score | interv_rate | cost | score/cost |
|---|---|---|---|---|
| naive | 0.175 | 0.000 | 4.0 | 0.0437 |
| always_reground | 0.430 | 1.000 | 6.0 | 0.0717 |
| **adaptive** | **0.335** | 0.583 | 5.2 | 0.0648 |
| random_at_budget | 0.275 | 0.583 | 5.2 | 0.0533 |

- Gate: `always − naive = +0.256` → **GREEN** (real, recoverable corruption).
- **Decision value: `adaptive − random = +0.059` at the identical budget**
  (both 1.17 interventions / 0.583 rate). Placing repairs by the risk signal
  beats placing them randomly.

## Chess (5 games)

| condition | avg_score | interv_rate | cost | score/cost |
|---|---|---|---|---|
| naive | 0.397 | 0.000 | 4.0 | 0.0992 |
| always_reground | 0.290 | 1.000 | 6.0 | 0.0484 |
| **adaptive** | **0.561** | 0.200 | 4.4 | 0.1275 |
| random_at_budget | 0.220 | 0.300 | 4.6 | 0.0477 |

- Gate: `always − naive = −0.107` → **RED**, but for an informative reason:
  **repairing a PGN with a weak model is itself lossy**, so re-grounding *every*
  hop injected more corruption than it fixed. `always_reground` made things
  worse than doing nothing.
- **Decision value: `adaptive − random = +0.342`** — the strongest result here.
  When the repair operation is risky, knowing *when* to repair is decisive:
  adaptive fired on only 20% of steps (the genuine corruption spikes) and skipped
  the clean ones, landing at **0.561** — above naive, always, and random — and
  the best score/cost of any condition in either domain.

## Demo cases (adaptive recovers what naive lost)

- **accounting4**: naive **0.000** vs adaptive **1.000**. Naive collapsed at
  round 1 (`basic_to_event_codes`, 97 → 0 transactions); the Conductor's risk
  spiked to 1.0 and the adaptive repair restored it.
- **chess5**: naive **0.212** vs adaptive **1.000**. Naive collapsed at round 1
  (`basic_to_cpx_autopsy`, 126 → 3 legal moves, risk 0.591); adaptive caught it.

## Honest scope

- Small n (6 + 5 samples), descriptive, no p-values.
- Depth 2 is the interpretable zone: at depth 5 this weak editor drives naive to
  *total* collapse (per-step risk saturates at 1.0 by round-trip 2, naive final
  score 0.000 on all 6 ledgers), which floors the fidelity axis. The compounding
  corruption is real; depth 2 keeps it graded enough to compare policies.
- The chess `always_reground < naive` result is a property of a weak repairer on
  PGN, not a bug — and it makes the selective (adaptive) policy look better than
  in any other run: repairing indiscriminately is actively harmful there.

Reproduce:

```bash
pip install --only-binary :all: chess
python run_delegate.py --domain both --live \
  --model microsoft/Phi-4-mini-instruct --num-round-trips 2 --target-rate 0.30
```
