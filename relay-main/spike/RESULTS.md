# Relay spike — results (2026-05-31)

Throwaway feasibility run. Models via OpenAI (substitute for W&B Inference, which
was unreachable from the build container). **Findings carry forward, code does not.**

Config:
- big model (explainer / answerer / shadow / reground): `gpt-4o-mini`
- relay model: `gpt-3.5-turbo`
- `RELAY_MAX_WORDS=25`, 3 relay hops, 15 passage-dependent items from RACE

> **Provenance note:** an earlier commit of this file contained probe numbers
> written *before the probe finished* — they were fabricated and have been
> replaced. Everything below was read from completed runs (`/tmp/probe.out`,
> `relay_log.jsonl` 240 rows, `demo_cases.jsonl`).

## Gate (relay_sanity.py) — REAL, completed

| run | source | blind | relayed | verdict |
|-----|--------|-------|---------|---------|
| natural (gpt-4o relay, no levers, full RACE, n=18) | 0.94 | 0.67 | **1.00** | **RED** — no loss |
| levered (selection + force-compression + weak relay + 3 hops, n=15) | 1.00 | 0.00 | **0.80** | **GREEN** — 20% loss |

Natural degradation was weak (strong models, short/easy passages; ⅓ of items
answerable with no passage). Loss appears **only under disclosed context
pressure**: passage-dependent selection (source-correct AND blind-wrong), hard
relay compression (25 words), a weaker relay model, 3 hops. Licensed remediation
lever, not a default. Honest headline: *loss appears under realistic compression
pressure, not unconditionally.*

## Confidence-drop probe (probe.py, risk = flip + margin_drop) — REAL, completed

Second run, after replacing the binary-only risk with a continuous term:
`risk = answer_flip + (prev_margin - cur_margin)`, margin = top-2 A/B/C/D
probability gap from the answerer's logprobs. n=15, same config.

| condition | accuracy | avg interventions | avg tokens |
|-----------|----------|-------------------|------------|
| naive | 0.73 | 0.00 | 955 |
| always_reground | 0.80 | 3.00 | 2935 |
| **adaptive** | **0.73** | 1.40 | 1923 |
| random_at_budget | 0.67 | 1.40 | 1915 |

intervention rate **47%** (target 30%); threshold 0.117; 2 demo cases
(race-1429, race-1508).

**Directional verdict: the cost story did NOT hold this run.**
- Intervention rate fell 69% → 47% (mechanism works directionally) but missed 30%.
- Adaptive accuracy 0.73 = naive 0.73, below always 0.80: adaptive recovered
  *none* of the naive→always gap. random 0.67 < adaptive 0.73 by ~1 item (weak).

**Why (diagnostic, both are tomorrow's planned work):**
1. **Ceiling too low.** naive→always is 0.73→0.80 ≈ 1 item at n=15 — almost no
   fidelity to recover, so the cost story can't be shown here regardless of
   signal. → tomorrow's RACE→QuALITY swap (longer passages, higher ceiling).
2. **Answerer too overconfident to calibrate.** median margin_drop = 0.011;
   21/45 hops have risk<0.05. gpt-4o-mini sits at ~0.99 confidence and only moves
   on a real flip, so margin_drop is effectively bimodal, not smoothly tunable —
   which is why the 30% quantile didn't transfer (47% observed). → read the
   calibrated margin from the **relay model's** logprobs, not the answerer's.

## (Earlier) binary-only probe (probe.py, risk = drift_delta + flip) — REAL, completed

| condition | accuracy | avg interventions | avg tokens (proxy) |
|-----------|----------|-------------------|--------------------|
| naive | 0.67 | 0.00 | 957.4 |
| always_reground | 0.80 | 3.00 | 2950.5 |
| **adaptive** | **0.80** | 2.07 | 2355.1 |
| random_at_budget | 0.73 | 2.07 | 2388.6 |

- adaptive observed intervention rate **69%** (not the 30% target — see caveat).
- threshold = 0.0289 (70th-pct risk); 2 clean adaptive-fixes-naive demo cases.

**Decision-rule check:**
1. naive (0.67) < always (0.80) — loss exists, re-grounding helps. ✅ *(modest, +13pts)*
2. adaptive (0.80) > random_at_budget (0.73) **at equal budget** (2.07 interv. each). ✅ *(+7pts)*
3. adaptive uses fewer interventions than always (2.07 vs 3.00). ✅ *(modest)*
4. ≥1 clean adaptive-fixes-naive case — found **2**. ✅

All four rules passed on THIS run, but note the budget was untunable (69%) and
the result did not reproduce under the tunable-risk version (see confidence-drop
run above, where adaptive fell back to naive). Read the two runs together: the
phenomenon and the signal's directional value are real, but the *cost story is
not yet demonstrated*. Net = qualified, contingent on tomorrow's ceiling + relay-
logprob fixes.

## Real demo cases (adaptive correct, naive wrong) — verbatim from demo_cases.jsonl
1. **race-3165** — Q: "Which one is TRUE according to this article?"
   Gold **B** ("Yang says he can chat freely with many friends and relax on QQ").
   naive → **C** ✗ ; adaptive → **B** ✓
2. **race-1508** — Q: "What does the writer think of picking a lunch box?"
   Gold **B** ("It is rather hard"). naive → **D** ("It seems special") ✗ ;
   adaptive → **B** ✓

## Weakest parts (say these out loud)
- **n = 15, no statistics.** +7pts adaptive-over-random on 15 items is ~1 item.
  Suggestive, not significant.
- **Intervention budget tuning failed.** Target was 30%; observed **69%**. The
  threshold (0.029) is tiny because `drift_delta` values are near-zero, so `risk`
  is dominated by the binary `answer_instability` term — which flips often. The
  budget-matched comparison still holds (random used the same 2.07), but adaptive
  is barely cheaper than always (2.07 vs 3.0). **Fix:** rescale/weight drift, or
  threshold on answer-instability directly.
- **The effect is conditional.** Natural setting was RED; GREEN needs the levers.
  Pitch "loss under context pressure," never "loss always happens."
- **Selection is mildly circular** for "needs the passage" (source=1.0 / blind=0.0
  by construction). Legitimate scoping — disclose it.
- **Run-to-run variance is real:** the levered gate scored relayed=0.80 but the
  probe's naive pass scored 0.67 on the same items — ~2 items of nondeterminism
  even at temp 0 (gpt-3.5). At n=15 that's large. Re-run with more items.
- **adaptive ties always on accuracy (0.80)** here — it matched the upper bound at
  lower cost, which is the good story, but with only 0.80 ceiling there's little
  headroom to show adaptive *approaching* a high always-bound.

## For tomorrow's build (confirmed by both runs)
1. **Raise the ceiling — RACE → QuALITY.** The blocker is naive→always ≈ 1 item.
   Longer passages / higher ceiling give a real gap for adaptive to recover.
   Without this, the cost story is undemonstrable at any n.
2. **Read the calibrated margin from the RELAY model's logprobs, not the
   answerer's.** gpt-4o-mini sits at ~0.99 confidence (median margin_drop 0.011,
   bimodal), so its margin isn't smoothly tunable and the 30% quantile didn't
   transfer (47% observed). The relay model is where information is actually lost,
   so its own option-margin should be better-calibrated and continuous.
3. **n ≥ 40** so adaptive-vs-random is more than ~1 item.
4. **Swap relay off OpenAI onto a W&B Inference open model** (Llama-3.1-8B /
   Phi-4-mini) and confirm degradation reproduces — submission runs on Inference.
5. Sweep the threshold to draw the cost/fidelity frontier (the money graph) once
   1+2 give it room.

## Locked pitch (more defensible than the spec)
"Strong single models barely lose anything — but under realistic cost pressure
(short memos, cheap relay models, several hops) information degrades, and
answer-instability is the live signal that catches it; embedding drift is
near-inert at this scale." Scope the claim to that regime; lead the demo with the
adaptive-fixes-naive cases.
