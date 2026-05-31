# Relay — On-Site Quickstart (do these in order)
### Goal of the next 90 min: a traced call in Weave, logprobs answered, data loaded, and the sanity gate run.

═══════════════════════════════════════════
## STEP 0 — Before any code (everyone, ~10 min)
═══════════════════════════════════════════
- [ ] **Rotate the exposed OpenAI key** (it's in last night's transcript — treat as burned).
- [ ] Submit the **W&B API / Inference credits form** (link in the event doc). Confirm credits are live.
- [ ] Create the **public GitHub repo** (empty) + AGI House team entry.
- [ ] Assign roles: Lead / Data / Signal / Weave-infra / Pitch.
- [ ] Set env vars on every machine:
  ```
  export WANDB_API_KEY=...           # from wandb.ai/settings
  export WANDB_PROJECT=team/project  # your entity/project
  ```
- [ ] `pip install openai weave datasets sentence-transformers matplotlib`

═══════════════════════════════════════════
## STEP 1 — Smoke test: one traced call (Weave-infra, 10 min)
═══════════════════════════════════════════
```python
import os, weave, openai
weave.init(os.environ["WANDB_PROJECT"])
client = openai.OpenAI(
    base_url="https://api.inference.wandb.ai/v1",
    api_key=os.environ["WANDB_API_KEY"],
    project=os.environ["WANDB_PROJECT"],
)
@weave.op()
def agent(model, system, user):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}],
    )
    return r.choices[0].message.content
print(agent("meta-llama/Llama-3.3-70B-Instruct", "You are concise.", "Say hi."))
```
PASS = reply prints AND a trace appears in the Weave **Traces** tab.

═══════════════════════════════════════════
## STEP 2 — ⚠️ THE LOGPROB TEST (Signal, 10 min) — decides your signal path
═══════════════════════════════════════════
Run this BEFORE building the Conductor. It determines whether your primary signal is logprob-margin or a fallback.
```python
r = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role":"user","content":"Answer with one letter only: A, B, C, or D. Q: 2+2=? A:3 B:4 C:5 D:6"}],
    logprobs=True, top_logprobs=5, max_tokens=1,
)
print(r.choices[0].logprobs)   # do we get token logprobs back?
```
- **Logprobs returned** → primary signal = `risk = flip + relay_margin_drop` (relay model's top-2 logprob margin).
- **None / error** → fallback. Either ask the shadow-answerer for a 0–1 confidence in the prompt, or run on `flip + drift` alone. **Do not block the build on this** — pick the path and move.

═══════════════════════════════════════════
## STEP 3 — Data (Data, 30 min)
═══════════════════════════════════════════
Primary = **QuALITY** (long passages = real ceiling). Confirmed fallback = **hard RACE subset** (you ran RACE last night). Inspect schema first, normalize to:
```python
# {"id":..., "passage":str, "question":str, "options":[..], "gold":"A"/"B"/"C"/"D"}
from datasets import load_dataset
# try QuALITY; if the path/loader fights you, fall back to RACE:
#   ds = load_dataset("race", "all")
# PRINT columns + one full example BEFORE mapping. Sample N≈40.
```
**Selection rule (write it into README):** keep items where the answer depends on a precise detail in the passage — negations, numbers, ordering, exceptions, "which is NOT true." These are where hand-off loss is visible.

═══════════════════════════════════════════
## STEP 4 — Sanity gate (Lead + Data, 30 min) — GO/NO-GO at ~1:00
═══════════════════════════════════════════
Relay chain on ~20 items, NO re-grounding yet. Relay agent on **Llama-3.1-8B** (weak, so loss shows). Force short memos ("rewrite in ≤40 words, preserve only what's needed to answer").
Compute and print:
- `a_none`  = answer with NO passage (contamination check — must be low)
- `a_full`  = answer with full passage (the ceiling)
- `a_1..a_N` = answer from each relayed memo (closed-book), N=3
- per-hop drift + shadow-answer flip

**VERDICT:**
- **GREEN** if `a_full − a_N ≥ 0.20` AND `a_none ≤ a_full − 0.2` → build the Conductor.
- **RED (no ceiling, a_full−a_N small)** → switch to QuALITY / harder items, more hops, cap memo length.
- **RED (a_none high)** → pick more passage-dependent items.
- Still RED ~1:30 → run the RACE-with-levers result and pitch the **fallback** (§13 of the brief). Don't sink the afternoon.

═══════════════════════════════════════════
## AFTER THE GATE (reference — full detail in relay_team_brief.md)
═══════════════════════════════════════════
- Conductor: `risk = flip + margin_drop`; threshold → ~30% intervention rate; on fire, inject question-relevant source chunk + repair memo.
- Four conditions: naive / always-reground / **random-at-budget** / adaptive. Random matches adaptive's observed rate. THIS COMPARISON IS THE PROJECT.
- Weave: Evaluation + exact_match scorer + leaderboard (copy the pattern from the W&B Inference docs); log drift/flip/margin/intervention as signals.
- `/goal` prompts (4, in order) are in the brief §9. Connect the W&B MCP first.
- Freeze architecture 4:30 · stop coding 6:00 · draft 7:00 · final 8:00.

## Today's first ask to the W&B staff on the floor
"We're building Relay — a multi-agent handoff harness, all on Weave + Inference. Two quick things: does your Inference endpoint return token logprobs, and what's the cleanest way to log a custom per-turn score / use Signals and publish a 4-run leaderboard?"
