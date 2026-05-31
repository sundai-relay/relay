# Relay × DELEGATE52 (chess · OFFLINE MOCK) — four-condition leaderboard

n=10 chains · depth=4 · editor=meta-llama/Llama-3.3-70B-Instruct · threshold=0.3965 · target_rate=0.3

| Condition | Fidelity | Avg interventions | Avg tokens |
|---|---|---|---|
| naive | 0.594 | 0.00 | 3054 |
| always_reground | 1.000 | 4.00 | 5684 |
| adaptive | 0.896 | 0.80 | 3518 |
| random_at_budget | 0.761 | 0.80 | 3597 |

adaptive intervention rate: 20%  (random matched to 8 interventions)

**adaptive − random fidelity = +0.135** (adaptive beats random at equal budget).
