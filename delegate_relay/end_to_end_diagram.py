"""Render an end-to-end diagram of the Relay x DELEGATE52 (accounting) harness with
current status badges. Pure matplotlib, no API calls. -> end_to_end.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C = dict(seed="#1f6feb", edit="#d29922", boundary="#8957e5", cond="#238636",
         score="#cf222e", weave="#bf3989", grey="#57606a", bg="#ffffff")

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, text, color, fc=None, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                 linewidth=2, edgecolor=color, facecolor=fc or "white", zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="#0b0f14", zorder=3, fontweight="bold" if bold else "normal")


def arrow(x1, y1, x2, y2, color="#57606a", text=None, style="-|>", lw=2, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=18,
                 linewidth=lw, color=color, zorder=1,
                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.5, text, ha="center", va="bottom",
                fontsize=8, color=color, style="italic")


ax.text(50, 96, "Relay × DELEGATE52 (accounting) — End-to-End", ha="center",
        fontsize=18, fontweight="bold", color="#0b0f14")
ax.text(50, 92.3, "A document is relayed through a chain of LLM edits; a Conductor watches "
        "cheap key-free signals and re-grounds only when corruption spikes.",
        ha="center", fontsize=10, color=C["grey"])

# ---- Row 1: the relay chain (one round-trip = one hop) ----
ax.text(3, 86, "① THE RELAY CHAIN  (telephone game on a real ledger)", fontsize=11,
        fontweight="bold", color="#0b0f14")
box(2, 73, 15, 9, "SEED  D0\n(gold, held by\nConductor only)\n64 txns · $18,565", C["seed"],
    fc="#ddf4ff", bold=True)
box(22, 75, 13, 7, "EDITOR\nforward edit\n“split by person”", C["edit"], fc="#fff8e6")
box(40, 75, 13, 7, "mid state\nperson_a.ledger\nother.ledger …", C["grey"], fc="#f6f8fa")
box(58, 75, 13, 7, "EDITOR\ninverse edit\n“merge by date”", C["edit"], fc="#fff8e6")
box(76, 73, 14, 9, "BOUNDARY B1\nshould == D0\n(here we measure)", C["boundary"], fc="#f3eefc",
    bold=True)

arrow(17, 77.5, 22, 78.5)
arrow(35, 78.5, 40, 78.5)
arrow(53, 78.5, 58, 78.5)
arrow(71, 78.5, 76, 77.5)
arrow(90, 77.5, 94, 77.5)
ax.text(95.5, 77.5, "B2 …\nBk", fontsize=8, color=C["boundary"], va="center")
ax.text(46.5, 72.5, "corruption happens here (compounds each hop)", ha="center",
        fontsize=8, color=C["score"], style="italic")

# ---- Row 2: the Conductor ----
ax.text(3, 66, "② THE CONDUCTOR  (coordinating bot — interjects on metrics)", fontsize=11,
        fontweight="bold", color="#0b0f14")
box(20, 47, 60, 15,
    "CONDUCTOR  (holds D0)\n\n"
    "at each boundary, compute KEY-FREE signals vs D0:\n"
    "•  invariant_deviation  (txn/payee/amount sets)  ← primary, continuous\n"
    "•  parse_health   •  embedding_drift (≈inert)\n"
    "risk = w·invariant_deviation + …      if risk > θ  →  RE-GROUND",
    C["cond"], fc="#eaffea")
arrow(83, 73, 83, 62, color=C["boundary"], style="-|>", rad=-0.2)
ax.text(86, 67, "signals", fontsize=8, color=C["boundary"])
arrow(70, 62, 78, 72, color=C["cond"], text=None, style="-|>", rad=-0.25)
ax.text(73.5, 66.5, "re-ground:\ndiff vs D0 →\nre-inject missing\ntxns → repair",
        fontsize=7.5, color=C["cond"])

# ---- Row 3: four conditions + scorer + weave ----
ax.text(3, 43, "③ FOUR CONDITIONS  (same chain, different re-grounding policy)",
        fontsize=11, fontweight="bold", color="#0b0f14")

# mini frontier
fx, fy, fw, fh = 6, 14, 40, 25
ax.add_patch(FancyBboxPatch((fx, fy), fw, fh, boxstyle="round,pad=0.5", linewidth=1.5,
             edgecolor=C["grey"], facecolor="#fbfcfd", zorder=1))
ax.text(fx + fw / 2, fy + fh - 2, "fidelity  ↑   vs   cost (tokens) →", ha="center",
        fontsize=9, color=C["grey"], fontweight="bold")
# axes
ax.plot([fx + 5, fx + 5], [fy + 3, fy + fh - 4], color=C["grey"], lw=1)
ax.plot([fx + 5, fx + fw - 3], [fy + 3, fy + 3], color=C["grey"], lw=1)
pts = [("naive", 0.20, 0.18, C["score"]), ("random", 0.55, 0.45, "#8250df"),
       ("adaptive", 0.55, 0.80, C["cond"]), ("always", 0.95, 0.95, C["seed"])]
for name, cx, cy, col in pts:
    px = fx + 6 + cx * (fw - 11)
    py = fy + 4 + cy * (fh - 9)
    ax.scatter(px, py, s=70, color=col, zorder=4)
    ax.text(px + 0.6, py + 0.6, name, fontsize=8, color=col, fontweight="bold")
ax.text(fx + fw / 2, fy + 0.5, "the project = adaptive vs random at equal budget",
        ha="center", fontsize=7.5, color=C["grey"], style="italic")

box(52, 26, 18, 9, "SCORER (gold)\nparse → invariants →\nround-trip fidelity 0–1\nscore(D0,D0)=1.000",
    C["score"], fc="#ffebe9")
box(52, 14, 18, 8, "W&B WEAVE\ntraces every call +\nEvaluation + Leaderboard\n(+ custom signals)",
    C["weave"], fc="#ffeff7")
box(76, 20, 20, 12,
    "ARTIFACTS\nleaderboard.md\nfrontier.png\ndemo_case.md\nresults.jsonl", C["grey"],
    fc="#f6f8fa")
arrow(70, 30.5, 76, 28)
arrow(70, 18, 76, 24)

# ---- status panel ----
ax.text(3, 9.5, "STATUS", fontsize=11, fontweight="bold", color="#0b0f14")
status = [
    ("✓", "Deterministic scorer + invariants  (fid(D0,D0)=1.0, dup/date fixed)", C["cond"]),
    ("✓", "Surgical re-ground proven  (0.885 → 1.000 on re-inject)", C["cond"]),
    ("✓", "Real corruption confirmed  (70B one round-trip → fidelity 0.821)", C["cond"]),
    ("✓", "Weave Evaluation + Leaderboard API  (tested, published OK)", C["cond"]),
    ("»", "Live 4-condition run  (editor=Llama-3.3-70B, ~40s/call, in progress)", C["edit"]),
    ("·", "Then scale to n≈6 depth 3 → frontier + demo", C["grey"]),
]
for i, (mark, txt, col) in enumerate(status):
    y = 7.2 - i * 1.25
    ax.text(4, y, mark, fontsize=10, color=col, fontweight="bold")
    ax.text(6.5, y, txt, fontsize=8.5, color="#0b0f14", va="center")

fig.savefig("end_to_end.png", dpi=140, bbox_inches="tight", facecolor="white")
print("wrote end_to_end.png")
