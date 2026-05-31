"""Real DELEGATE52 round-trip substrate (accounting + chess).

Unlike relay.roundtrip (a synthetic JSON inventory), this package loads the
actual DELEGATE52 samples from delegate/delegate52.jsonl and uses the dataset's
own forward/backward NL prompt pairs as the reversible edits. Gold is the seed
document itself; the evaluation is a round trip (apply forward -> apply inverse
-> compare to seed), exactly as the dataset intends.
"""
