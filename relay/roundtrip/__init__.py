"""relay.roundtrip — the locked round-trip build.

Generate a structured JSON doc, transform it with reversible edits, then
reverse them. A faithful workflow reconstructs the seed. The Conductor watches
a key-free checksum after every edit and performs *targeted repair* only when
risk fires. Four policies are compared on a fidelity/cost frontier.

Modules:
    tasks     - procedural seed-doc + edit-pair generation (fixed seeds)
    agents    - apply_edit / repair_doc (W&B Inference, or a deterministic mock)
    checksum  - runtime_risk (the Conductor's warning light)
    scorer    - final_structural_score (post-run only; never drives policy)
    runner    - run one task under one policy
"""
