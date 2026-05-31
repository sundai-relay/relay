# Demo case — adaptive recovers what naive lost

Chain `chess2#0`: naive fidelity **0.540** → adaptive **1.000** (+0.460).

Per-boundary trace (naive vs adaptive) is in results.jsonl; filter `sample_id == "chess2#0"`. Look for the boundary where naive's invariant_deviation spikes and adaptive's `intervened == 1`.
