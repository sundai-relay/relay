# Relay × DELEGATE52 — demo cases (adaptive recovers what naive lost)

## accounting — `accounting4`  (adaptive − naive = +1.000)

naive final score 0.000 | adaptive final score 1.000

Per-round-trip runtime risk under naive (no repair):

| round | edit | risk | n_seed→n_cur |
|---|---|---|---|
| 0 | basic_to_endowment | 0.143 | 97→91 |
| 1 | basic_to_event_codes | 1.000 | 97→0 |

## chess — `chess5`  (adaptive − naive = +0.788)

naive final score 0.212 | adaptive final score 1.000

Per-round-trip runtime risk under naive (no repair):

| round | edit | risk | n_seed→n_cur |
|---|---|---|---|
| 0 | basic_to_activity | 0.000 | 126→126 |
| 1 | basic_to_cpx_autopsy | 0.591 | 126→3 |

