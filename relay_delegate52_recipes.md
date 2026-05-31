# Relay × DELEGATE52 — Eval Recipes (proposal)
### Porting the Relay handoff-fidelity harness from QuALITY MCQ onto DELEGATE52 round-trip document editing.

> Read this alongside `relay_team_brief.md`, `relay_quickstart.md`, and `relay_submission_package.md`.
> This file proposes **what to build and how to evaluate**; it does not change the locked Relay thesis,
> it gives that same thesis a stronger dataset.

---

## 0. The one-paragraph version
`relay-main` relays an **explanation** (a compressed memo) down a chain of agents and measures whether
the **MCQ answer** survives. DELEGATE52 relays a **document** down a chain of **edits** and measures
whether the **document content** survives a round trip. It is the *same telephone game* — but on real
professional documents (ledgers, recipes, chess games, code, 3D objects), with a **built-in, exact
ceiling** and **guaranteed degradation** (the paper: frontier models corrupt ~25% of content after 20
interactions). Everything the Relay brief asks for — Conductor, key-free signals, the four conditions,
the fidelity/cost frontier, Weave eval+leaderboard — ports over directly. The Conductor (the
"coordinating bot") watches cheap key-free metrics as content passes through the stages and **interjects /
re-grounds only when the corruption signal spikes.** We are **not** running a model leaderboard ("which
clone edits best"); we hold the model roughly fixed and study **how content degrades across the four
conditions**, and whether the Conductor's signal has decision value.

---

## 1. Why DELEGATE52 is a *better* fit than QuALITY (this is the headline)
The spike's two biggest risks (see `relay-main/spike/RESULTS.md`) were **(a) no ceiling** —
`a_full − a_naive ≈ 1 item`, nothing to recover — and **(b) the answerer too saturated to calibrate**
(median margin-drop 0.011, untunable to a 30% budget). DELEGATE52 removes both:

| Spike pain (RACE/QuALITY) | DELEGATE52 fix |
|---|---|
| Had to *hunt* for a recoverable gap; `/goal` prompt #2 existed only for this | **Ceiling is structural and free.** Round-trip gold = the original seed document. The paper guarantees a large naive-vs-always gap (~25% corruption @ 20 hops). |
| Gold = a 1-letter MCQ; coarse, 0/1 per item | **Gold = a continuous domain similarity score ∈ [0,1]** (paper Fig. 5). Smooth fidelity axis, no n=15 cliff. |
| "Answer-instability" signal saturated (70B at 0.99 conf) | **Structural-invariant breakage** (a dropped ledger row, an illegal chess move) is *not* saturated and is naturally continuous → directly tunable to a 25–35% budget. |
| Loss only appeared under disclosed levers (weak relay, 25-word memos) | Loss is the dataset's whole point; **no levers needed** — the documents corrupt on their own. The "is this a contrived setup?" question disappears. |
| Demo was an abstract C→B answer flip | **Demo is visual and visceral** — a 3D palm tree / textile pattern / graph diagram visibly disintegrates over hops, then snaps back when the Conductor re-grounds (paper Fig. 1). |

This is the same project with the gate de-risked. Lead the pitch with: *"We took our handoff-fidelity
harness and pointed it at Microsoft Research's brand-new DELEGATE52 — where the degradation is real,
measurable, and visual — and showed the Conductor catches it cheaper than chance."*

---

## 2. What I verified in `delegate/delegate52.jsonl` (grounding)
- **234 samples · 48 domains · avg 8 states/sample · 7 forward edits/sample · 1,629 backward (inverse) edits total.**
- Seed docs: avg **3,685 tokens** (1,348–9,569) — in the 2–5k "real complexity" band, similar to QuALITY passages.
- Each sample is a **state graph**: `basic_state` (the seed) has N forward edit prompts, each pointing to a
  `target_state`; every `target_state` has exactly one **backward prompt that points back to `basic_state`**
  (the inverse edit). → every (forward, backward) pair is a ready-made **2-hop round trip**.
- **Critical for eval design:** the public release ships **only the seed (`basic_state/*`) + distractor files** —
  target-state reference solutions are *not* included (confirmed: only 3/234 ship any). **Therefore the gold is the
  seed document itself, and the evaluation must be round-trip** (`apply forward → apply inverse → compare to seed`).
  This is exactly what the dataset calls a "round-trip relay simulation," and it is what makes the gold key-free
  for the agents: they never see the seed during the inverse edit, only the Conductor/scorer holds it.
- Semantic operations available as free metadata (use for slicing/selection):
  `sorting(646) · split_and_merge(456) · classification(455) · format_knowledge(397) · numerical_reasoning(323) ·
  string_manipulation(285) · referencing(279) · context_expansion(234) · domain_knowledge(107) · topic_modeling(61) ·
  constraint_satisfaction(57)`.
- Distractor files exist per sample ("topically relevant but task-irrelevant") — an *optional* extra lever:
  feed them into the editor's context to test whether re-grounding also suppresses distractor contamination.

---

## 3. Architecture (port of `relay-main`, edits replace summaries)
```
Seed document D0  (basic_state, held by the Conductor as the source/ceiling)
      │
   Editor 1   (sees Dk + the NL edit instruction → writes Dk+1)   ← corruption happens here
      │  edit_1  (a forward edit prompt from the state graph)
   Editor 2
      │  edit_2
      …  (k hops; for a round trip the back half are the inverse instructions)
      ▼
   Final document Dn
      ▲
   Conductor  (holds D0 + cheap domain invariants of D0; after each edit:
               score corruption-risk from key-free signals; if risk > threshold,
               RE-GROUND — re-inject the original records the edit touched + repair
               Dk+1 before it passes on; log every hop to Weave)
```
- **The relayed unit is the document content, not a memo.** This is what your note meant by *"the content
  that it ingested going through the multiple stages."*
- **Editor model:** one fixed model (e.g. `meta-llama/Llama-3.1-8B-Instruct` weak, or `Llama-3.3-70B` to mirror
  the paper's "frontier models still corrupt"). We are **not** comparing models — we compare the four
  **conditions** on the same content. (Your: *"not comparing the clone that each does what they have best."*)
- **Two chain shapes** (pick per time budget):
  - **(A) Single round-trip (MVP, fast):** `D0 →forward→ D1 →inverse→ D0'`. 2 edit hops, intervention possible at each.
    Score `sim(D0', D0)`. Cheapest; proves the mechanism.
  - **(B) Compounding chain (the paper's regime, the money demo):** chain several round-trips, feeding `D0'` back
    in: `D0 →f1→ →b1→ D0' →f2→ →b2→ D0'' → …`. Corruption **compounds across hops** (the paper's "25% after 20
    interactions"). This is where naive collapses and the gap for adaptive to recover is widest.

---

## 4. The eval recipe (straight from the "recipe for evals" PDF = the DELEGATE52 paper, Fig. 5)
Every domain implements the same four-stage pipeline; **this is the deterministic, gold-only scorer** (agents never touch it):

```
Raw text (Dn)  ──parse──▶  structured representation  ──count──▶  domain statistics
                                                                        │
reference (D0) ──parse──▶  structured representation  ────────────────▶ │
                                                                        ▼
                                                   Semantic Equivalence: weighted similarity ∈ [0,1]
```
Worked example from the PDF (recipe domain):
`parse_recipe(file) → {ingredients:11, steps:36, tips:8}`, then
`score = 0.4·Ingredient + 0.4·Step + 0.2·Tip` where
*Ingredient = Hungarian matching on names · Step = sequential text similarity · Tip = bipartite matching.*

**Build action:** the real DELEGATE52 GitHub repo (`microsoft/delegate52`, `run_relay.py`) ships these
parsers/evaluators per domain. **Do not re-implement 48 domain scorers.** Two options, in order of preference:
1. **Vendor the official evaluators for the 3–4 domains we pick** (read their licenses; CDLA-permissive data,
   check code license) and wrap them behind one **MCP "scorer" tool** + one **MCP "source-grounding" tool** —
   satisfying the brief's MCP-as-primary-protocol requirement *with real usage*.
2. If their code fights us, **hand-write a thin parser+scorer for our chosen domains only** (recipe/accounting/
   chess/json are all <50 lines each — see §6). Cap at what we can finish before the 1:00 gate.

`fidelity = score(Dn, D0) ∈ [0,1]` is the single accuracy number that replaces the spike's `exact_match`.

---

## 5. Key-free signals — the Conductor's eyes (the part that earns "Most Sophisticated Harness")
Gold (the §4 score) is used **only** for final scoring. The Conductor holds `D0` (legitimate, like the spike's
Conductor holds the passage) and watches these, none of which touch the reference solution:

| # | Signal | Analog in spike | Live or inert? |
|---|---|---|---|
| 1 | **Structural-invariant deviation** — Conductor precomputes cheap conserved quantities on `D0` (e.g. #transactions & debit=credit balance for ledgers; #ingredients/#steps for recipes; #moves & legal final position for chess; row-count/column-set for spreadsheets; set of function names / AST node count for python). At each hop recompute on `Dk`; `inv_dev = fraction of invariants that broke or drifted`. **Continuous → threshold-tunable to a 25–35% budget.** | **answer-instability / margin-drop** | **LIVE (primary).** Detects when an edit changed *what matters*. Not saturated (unlike the 70B answerer). |
| 2 | **Parse-health** — run the domain parser on `Dk`; `parse_fail ∈ {0,1}` or graded (# parse errors). A file that no longer opens = corruption. | a hard answer flip | LIVE, cheap, binary backstop. |
| 3 | **Embedding drift** — `1 − cos(emb(Dk), emb(D0))` via all-MiniLM-L6-v2 (local, CPU). Coarse semantic displacement. | **question-conditioned drift** | **Expect near-inert** (matches the spike finding — keep it logged for the honest "drift alone is insufficient" story). |
| 4 | **Edit-confidence / shadow-reverter** (secondary) — either the editor's token-logprob margin in the rewritten regions (⚠️ **logprob availability on W&B Inference is unverified — run the §2 logprob test from the quickstart FIRST**), *or* a cheap shadow agent that applies the inverse instruction to `Dk` and measures how far it lands from `Dk-1` (self-consistency of the edit). | **shadow-answerer flip** | LIVE if logprobs exist; else use the shadow-reverter fallback. |

```
risk = w1·inv_dev + w2·parse_fail + w3·drift (+ w4·edit_uncertainty)
```
Start `w3≈0` (drift is the inert control), tune `w1,w2` with Claude Code `/goal` prompt #3. Intervene when
`risk > threshold`; threshold set to a target **25–35% intervention rate** (quantile of observed risk, same as
the spike). **Re-ground** = re-inject the specific original records the edit touched (Conductor diffs `Dk` vs
`D0` structurally, pulls the affected ledger rows / recipe steps / chess moves, hands them back) + a repair pass.

> Honest framing to keep (from the brief): these are **cheap runtime features, not a corruption oracle.** The
> scientific claim is operational — *does this risk model pick better re-grounding moments than random at the
> same budget?* The structural-invariant signal is far harder to dismiss than an embedding number, which is
> exactly the spike's "answer-instability is the live signal" lesson, now on documents.

---

## 6. The four conditions = the **four lenses** of degradation (the project)
Same content, same edit chain, four policies — this is the locked comparison from the brief, unchanged. This is
"how it degrades across four different lenses, with the coordinating bot interjecting":

| Lens / Condition | What it proves | Cost | Fidelity |
|---|---|---|---|
| **Naive relay** (Conductor silent) | corruption exists & **compounds** over hops | low | low |
| **Always-reground** (re-ground every hop) | upper bound — the corruption *is* recoverable | high | high |
| **Random-at-budget** (re-ground at random hops, count matched to adaptive) | re-grounding *itself* helps | = adaptive | modest |
| **Adaptive (Relay)** (re-ground when risk spikes) | the **signal has decision value** | = random | best per token |

**The whole project is adaptive vs random-at-budget**, at equal intervention count (the brief calls this
non-negotiable). The money graph: **x = cost (tokens or # interventions), y = fidelity (round-trip score ∈ [0,1]),
four labelled points.** With DELEGATE52's real ceiling, the naive→always gap should be wide enough that adaptive
can visibly sit on the efficient frontier — the thing the spike couldn't show on RACE.

> Secondary "measurement lenses" (optional, for the trace view, *not* the experimental axis): the same final
> document can be scored through **parse-validity → structural-count fidelity → semantic-equivalence →
> round-trip-to-origin**, a degradation stack. Show this in the Weave trace if time allows; the four *conditions*
> above remain the headline comparison.

---

## 7. Concrete domain recipes (pick 2–3; ranked for this build)
Counts are samples available in the file. Pick for: easy parser, visible corruption, demo punch.

1. **`accounting` (6 samples) — RECOMMENDED MVP.** `.ledger` text.
   - *Parser:* split on blank lines → transactions; each = date + payee + postings (account, amount).
   - *Invariants (signal 1):* #transactions, set of payees, Σ debits = Σ credits (double-entry must balance), set of accounts.
   - *Round trip:* `basic_to_category_split` → `merge back to one ledger sorted by date`. Corruption = a dropped/renamed transaction or an unbalanced posting → invariant breaks immediately and visibly.
   - *Score (§4 style):* `0.5·transaction-match (Hungarian on (date,payee,amount)) + 0.3·balance-preserved + 0.2·account-set`.
2. **`recipe` (1 sample) — BEST FOR THE DEMO** (it's the PDF's worked example; scorer is spelled out for you).
   - Round trip: `split into 4 files` → `merge back`, or `scale 12→30 eclairs` → `scale back to 12 + metric`. Numeric drift on the scale-back is a clean, visible corruption.
   - Score: the exact `0.4·Ingredient + 0.4·Step + 0.2·Tip` from Fig. 5. Only 1 sample though → demo, not stats.
3. **`chess` (5 samples) — strongest "is it really corrupted?" signal.** `.pgn`.
   - *Invariants:* moves are legal from the start position; #moves; final FEN. An illegal move = unambiguous corruption (no embedding hand-waving). Use `python-chess` to validate → parse-health signal is essentially free and decisive.
4. **`python` (7), `json` (6), `spreadsheet` (6) — most samples → best for n≥15 stats.**
   - python: AST-parse (`ast.parse`) for parse-health; set of def/class names + node count for invariants.
   - json: `json.loads` for parse-health; recursive key-set / leaf count for invariants.
   - spreadsheet: row count, column set, per-column type, totals.
5. **`obj3d` (6), `musicsheet` (6), `subtitles` (5) — VISUAL DEMO GOLD** (the paper's teaser uses 3D objects /
   textile patterns / graphs). Harder to score precisely under time pressure, but a rendered before/after is the
   <2-min video's strongest 10 seconds. Use one as the *demo case*, score it loosely; do the *stats* on accounting/json/python.

**Selection rule (write into README):** prefer round trips whose `semantic_operations` include
`numerical_reasoning`, `split_and_merge`, `referencing`, or `sorting` — these are where silent drops/reorderings
happen, i.e. where the invariant signal lights up. (Analog of the spike's "negations/numbers/ordering" rule.)

---

## 8. Sanity gate (port of `relay_quickstart.md` STEP 4) — GO/NO-GO at ~1:00
Run the chain on ~15–20 round trips, **no re-grounding yet**, on the chosen domain(s). Compute & print:
- `fid_identity` = score(D0, D0) — must be **1.0** (parser/scorer sanity; if not, the scorer is buggy).
- `fid_naive`    = score(round-trip output, D0) under naive relay (the corruption).
- `fid_always`   = score under re-ground-every-hop (the ceiling, should approach 1.0).
- per-hop `inv_dev`, `parse_fail`, `drift`.

**VERDICT (mirrors the brief):**
- **GREEN** if `fid_always − fid_naive ≥ ~0.20` (real, recoverable corruption) → build/finish the Conductor.
- **RED (gap too small)** → switch to the **compounding chain (shape B)**, add hops, use the weaker editor model,
  or pick a more fragile domain (accounting/chess over fiction).
- **RED (scorer noisy / `fid_identity` < 1.0)** → fix the parser before anything else.
- Because DELEGATE52 is *designed* to corrupt, GREEN is the expected outcome — this gate should pass where the
  spike's RACE gate failed.

---

## 9. Weave + sponsor mapping (unchanged from the brief — this is the Best-Use-of-Weave entry)
- `@weave.op()` on `editor`, `conductor.score`, `reground`, `score(Dn,D0)` → full per-hop trace.
- `weave.Evaluation` + the §4 round-trip scorer over the four conditions → published as a **fidelity/cost leaderboard**.
- Log per hop as **custom signals**: `inv_dev`, `parse_fail`, `drift`, `edit_uncertainty`, `risk`, `intervened`,
  `tokens`, `fidelity` — a domain-specific **handoff-fidelity guardrail** on *intermediate document state*.
- **MCP (primary protocol, real usage):** wrap the **source-grounding tool** (Conductor's re-inject), the
  **benchmark loader** (delegate52.jsonl → states/round-trips), and the **domain scorer** as MCP servers. This is
  more genuinely "MCP-as-orchestration" than the spike had, because the scorer/grounder are real tools the loop calls.
- **W&B Inference** for all editor/conductor/shadow calls (open models, traced by default). 429 → sequential + backoff (the spike's `llm.py` backoff client ports directly).
- Claude Code `/goal` prompts #1–#4 from the brief apply verbatim, s/`run_all_conditions.py`/our runner/ and
  s/MCQ accuracy/round-trip fidelity/.

---

## 10. Demo (port of "show the failure, not the evals")
One round trip, compounding chain, on a **visual** domain (obj3d palm tree or graphviz diagram):
seed renders correctly → editor hop drops/renames groups → `inv_dev` spikes & parse-health flags it → Conductor
re-injects the original group records → document repaired → final render matches the seed. Then cut to the Weave
trace, then the four-point fidelity/cost frontier. Closer (the BD line, unchanged): *"This is the agentic analogue
of handing a document down a chain of assistants — the same silent corruption, now measurable and repairable."*

---

## 11. Cut list / honesty (from the brief, still binding)
- Don't implement all 48 domain scorers — **2–3 domains only.** Don't claim A2A unless wired. Don't claim
  logprobs until the §2 test passes. No model-vs-model leaderboard (that's the paper's job, not ours).
- Say out loud: small n, descriptive results, no p-values; signals are cheap features, not a corruption oracle;
  the claim is **adaptive beats random at equal budget**, nothing more.

---

## 12. First three actions
1. `git clone https://github.com/microsoft/delegate52` and read `run_relay.py` + the evaluators for
   accounting/recipe/chess — decide vendor-vs-reimplement (§4).
2. Run the **logprob test** (quickstart STEP 2) on W&B Inference → fixes signal #4's path.
3. Write the loader (jsonl → list of round-trip tasks) + the accounting parser/scorer + `fid_identity`==1.0 check,
   then run the §8 gate on n≈15 accounting round trips. That's the GO/NO-GO.
