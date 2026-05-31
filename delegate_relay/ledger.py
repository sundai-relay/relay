"""Deterministic ledger parser + invariants + round-trip scorer (the GOLD).

Domain: `accounting` (hledger/ledger plain-text). This is the §4 "recipe for evals"
pipeline for one domain:  parse -> domain statistics -> weighted semantic equivalence.

Nothing here calls an LLM. Gold (the seed document D0) is used ONLY by score() and
by the Conductor's invariants; the editor agents never see it during a relay hop.

A faithful round trip (split-by-person  ->  merge-sorted-by-date) must return EXACTLY
the seed, so score(D0, D0) == 1.0 and any drop below 1.0 is real corruption.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- parsing ----
_AMOUNT_RE = re.compile(r"-?\$?-?[\d,]+(?:\.\d+)?")
_DATE_RE = re.compile(r"^\d{4}[/-]\d{1,2}[/-]\d{1,2}\b")


@dataclass
class Posting:
    account: str
    amount: float | None  # None = implicit balancer


@dataclass
class Txn:
    date: str
    payee: str
    postings: list[Posting] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)

    def amounts(self) -> list[float]:
        return [p.amount for p in self.postings if p.amount is not None]

    def accounts(self) -> set[str]:
        return {p.account for p in self.postings}


def _norm_date(date: str) -> str:
    """Zero-pad a YYYY/M/D date so 2016/6/1 == 2016/06/01 (avoids false round-trip
    mismatches when an editor emits unpadded dates on merge)."""
    parts = re.split(r"[/-]", date.strip())
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        y, m, d = parts
        return f"{int(y):04d}/{int(m):02d}/{int(d):02d}"
    return date.strip()


def _money(tok: str) -> float | None:
    if not tok:
        return None
    neg = tok.count("-") % 2 == 1
    digits = tok.replace("$", "").replace(",", "").replace("-", "")
    if not digits:
        return None
    try:
        v = float(digits)
    except ValueError:
        return None
    return -v if neg else v


def parse_ledger(text: str) -> list[Txn]:
    """Parse ledger text into transactions. Robust to blank-line-separated blocks."""
    txns: list[Txn] = []
    cur: Txn | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            if cur:
                txns.append(cur)
                cur = None
            continue
        # comment line
        if stripped.startswith(";"):
            if cur:
                cur.comments.append(stripped.lstrip("; ").strip())
            continue
        # transaction header: starts at col 0 with a date
        if _DATE_RE.match(line) and not line.startswith((" ", "\t")):
            if cur:
                txns.append(cur)
            date = _norm_date(_DATE_RE.match(line).group(0))
            payee = line[_DATE_RE.match(line).end():].strip()
            # trailing inline comment on header
            if ";" in payee:
                payee = payee.split(";", 1)[0].strip()
            cur = Txn(date=date, payee=payee)
            continue
        # posting line (indented account + optional amount)
        if cur is not None and (raw.startswith((" ", "\t"))):
            body = stripped
            if ";" in body:
                inline = body.split(";", 1)[1].strip()
                body = body.split(";", 1)[0].strip()
                if inline:
                    cur.comments.append(inline)
            m = _AMOUNT_RE.search(body)
            if m and "$" in m.group(0):
                account = body[: m.start()].strip()
                amount = _money(m.group(0))
            else:
                account, amount = body, None
            if account:
                cur.postings.append(Posting(account=account, amount=amount))
    if cur:
        txns.append(cur)
    return txns


# ------------------------------------------------------------ invariants ----
# Key-free: computed from the document itself. The Conductor computes these on D0
# (which it legitimately holds) and re-checks them on each relayed Dk. None of them
# touch the reference SCORE, so the signal stays honest.

def invariants(txns: list[Txn]) -> dict:
    amts = sorted(round(a, 2) for t in txns for a in t.amounts())
    return {
        "n_txns": len(txns),
        "payees": _multiset(t.payee for t in txns),
        "accounts": _multiset(a for t in txns for a in t.accounts()),
        "amount_multiset": tuple(amts),
        "amount_sum": round(sum(amts), 2),
        "n_comments": sum(len(t.comments) for t in txns),
    }


def _multiset(it) -> tuple:
    from collections import Counter
    return tuple(sorted(Counter(it).items()))


def invariant_deviation(cur: list[Txn], ref: list[Txn]) -> float:
    """Continuous key-free risk feature in [0,1]: fraction of invariant 'mass' that
    deviates from the reference. 0 = identical structure, ->1 = badly corrupted.
    Tunable, so it can be thresholded to a target intervention budget."""
    ic, ir = invariants(cur), invariants(ref)
    terms = []
    # count deviations
    terms.append(_rel(ic["n_txns"], ir["n_txns"]))
    terms.append(_rel(ic["n_comments"], ir["n_comments"]))
    terms.append(_rel(ic["amount_sum"], ir["amount_sum"]))
    # set/multiset jaccard distances
    terms.append(1 - _jacc(ic["payees"], ir["payees"]))
    terms.append(1 - _jacc(ic["accounts"], ir["accounts"]))
    terms.append(1 - _jacc(ic["amount_multiset"], ir["amount_multiset"]))
    return float(sum(terms) / len(terms))


def _rel(a: float, b: float) -> float:
    if a == b:
        return 0.0
    denom = max(abs(a), abs(b), 1e-9)
    return min(1.0, abs(a - b) / denom)


def _jacc(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def parse_health(text: str) -> float:
    """1.0 = fully parseable; ->0 = malformed. Cheap binary-ish backstop signal."""
    txns = parse_ledger(text)
    if not txns:
        return 0.0
    bad = sum(1 for t in txns if not t.postings or not t.payee)
    return 1.0 - bad / len(txns)


# --------------------------------------------------------------- scorer ------
# The GOLD score (paper Fig. 5 style). Round-trip: compare final doc to seed D0.

def score(final_text: str, seed_text: str, fx_tol: float = 0.02) -> float:
    """Weighted round-trip fidelity in [0,1].

      0.5 * transaction match  (greedy on (date, payee, sorted amounts))
    + 0.2 * amount fidelity     (1 - normalized abs error on the grand total)
    + 0.2 * account-set jaccard
    + 0.1 * receipt/comment match
    """
    fin = parse_ledger(final_text)
    ref = parse_ledger(seed_text)
    if not ref:
        return 0.0

    txn_match = _txn_match(fin, ref)
    amount_fid = _amount_fidelity(fin, ref, fx_tol)
    acct_j = _jacc(
        _multiset(a for t in fin for a in t.accounts()),
        _multiset(a for t in ref for a in t.accounts()),
    )
    comment_j = _jacc(
        _multiset(c for t in fin for c in t.comments),
        _multiset(c for t in ref for c in t.comments),
    )
    return round(0.5 * txn_match + 0.2 * amount_fid + 0.2 * acct_j + 0.1 * comment_j, 4)


def _txn_key(t: Txn) -> tuple:
    return (t.date, t.payee.lower(), tuple(sorted(round(a, 2) for a in t.amounts())))


txn_key = _txn_key  # public alias


def diff_blocks(cur_text: str, seed_text: str) -> list[str]:
    """Return the raw transaction blocks from the seed (D0) that are MISSING or CHANGED
    in cur_text — the exact records the Conductor re-injects when re-grounding. Uses raw
    block text (not reconstructed) so original formatting/amounts are preserved."""
    from collections import Counter

    def blocks(text: str) -> list[tuple[tuple, str]]:
        out = []
        for raw in re.split(r"\n\s*\n", text):
            if not raw.strip():
                continue
            ts = parse_ledger(raw)
            if ts:
                out.append((_txn_key(ts[0]), raw.strip()))
        return out

    cur_keys = Counter(k for k, _ in blocks(cur_text))
    missing: list[str] = []
    for key, raw in blocks(seed_text):
        if cur_keys[key] > 0:
            cur_keys[key] -= 1  # accounted for
        else:
            missing.append(raw)  # dropped or altered beyond recognition
    return missing


def _txn_match(fin: list[Txn], ref: list[Txn]) -> float:
    from collections import Counter
    cf, cr = Counter(_txn_key(t) for t in fin), Counter(_txn_key(t) for t in ref)
    matched = sum((cf & cr).values())
    return matched / max(len(ref), 1)


def _amount_fidelity(fin: list[Txn], ref: list[Txn], tol: float) -> float:
    sf = sum(a for t in fin for a in t.amounts())
    sr = sum(a for t in ref for a in t.amounts())
    if abs(sr) < 1e-9:
        return 1.0 if abs(sf) < 1e-9 else 0.0
    err = abs(sf - sr) / abs(sr)
    return 1.0 if err <= tol else max(0.0, 1.0 - err)


if __name__ == "__main__":
    import json

    rows = [json.loads(l) for l in open("../delegate/delegate52.jsonl")]
    acc = [r for r in rows if r["sample_type"] == "accounting"]
    print(f"{'sample':14}{'txns':>6}{'payees':>8}{'accts':>7}{'sum':>12}{'fid(D0,D0)':>12}{'health':>8}")
    for r in acc:
        seed_key = [k for k in r["files"] if k.startswith("basic_state/")][0]
        text = r["files"][seed_key]
        txns = parse_ledger(text)
        inv = invariants(txns)
        s = score(text, text)            # must be 1.0
        h = parse_health(text)
        print(f"{r['sample_id']:14}{inv['n_txns']:>6}{len(set(p for p,_ in inv['payees'])):>8}"
              f"{len(set(a for a,_ in inv['accounts'])):>7}{inv['amount_sum']:>12.2f}{s:>12}{h:>8.2f}")

    # corruption sanity: drop one transaction -> fidelity must fall, inv_dev must rise
    text = acc[0]["files"][[k for k in acc[0]["files"] if k.startswith("basic_state/")][0]]
    txns = parse_ledger(text)
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    corrupted = "\n\n".join(blocks[:-1])  # drop last transaction
    print(f"\ncorruption check (drop 1 of {len(txns)} txns):")
    print(f"  fidelity        {score(text, text):.3f} -> {score(corrupted, text):.3f}")
    print(f"  invariant_dev   0.000 -> {invariant_deviation(parse_ledger(corrupted), txns):.3f}")

    # date-padding robustness: unpadded dates on merge must NOT count as corruption
    unpadded = re.sub(r"^(\d{4})/0?(\d{1,2})/0?(\d{1,2})",
                      lambda m: f"{m.group(1)}/{int(m.group(2))}/{int(m.group(3))}",
                      text, flags=re.MULTILINE)
    print(f"\ndate-padding check (re-emit dates as YYYY/M/D):")
    print(f"  fidelity vs seed {score(unpadded, text):.3f}   (must be 1.000)")

    # duplicate-transaction handling: a duplicated txn is a real corruption, not a no-op
    dup = text + "\n\n" + blocks[0]
    print(f"\nduplicate check (append a copy of txn #1):")
    print(f"  fidelity        {score(dup, text):.3f}   (should be < 1.000)")
    print(f"  invariant_dev   {invariant_deviation(parse_ledger(dup), txns):.3f}   (should be > 0)")
