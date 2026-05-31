"""Accounting domain: a thin `.ledger` parser + key-free invariants + scorer.

Recipe (relay_delegate52_recipes.md §7):
  parser    : split on blank lines -> transactions; each = date + payee +
              postings (account, amount).
  invariants: #transactions, set of payees, set of accounts, double-entry
              balance (per transaction).
  score     : 0.5*transaction-match + 0.3*balance-preserved + 0.2*account-set.

Everything here is key-free w.r.t. the gold: runtime_risk compares the current
doc to the SEED only; score() is the post-run gold metric.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

name = "accounting"

_DATE = re.compile(r"^\d{4}[/-]\d{1,2}[/-]\d{1,2}\b")
_AMOUNT_CHARS = re.compile(r"[^\d.\-]")  # strip currency symbols, commas, spaces


def _to_amount(s: str) -> Optional[float]:
    s = s.strip()
    neg = "-" in s or s.startswith("(")
    cleaned = _AMOUNT_CHARS.sub("", s.replace(",", ""))
    cleaned = cleaned.replace("-", "")
    if not cleaned or cleaned == ".":
        return None
    try:
        v = float(cleaned)
    except ValueError:
        return None
    return -v if neg else v


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def parse(text: str) -> Dict:
    """Parse ledger text -> {transactions:[{date,payee,postings:[{account,amount}]}]}.

    A transaction block is delimited by blank lines; its first non-comment line
    is the `DATE Payee` header; indented non-comment lines are postings. Lines
    starting with ';' are comments (e.g. receipt refs) and ignored.
    """
    transactions: List[Dict] = []
    if not isinstance(text, str):
        return {"transactions": []}
    blocks = re.split(r"\n\s*\n", text)
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        # find the header line (first non-comment line that looks like a date)
        header = None
        body_start = 0
        for i, ln in enumerate(lines):
            if ln.lstrip().startswith(";"):
                continue
            header = ln.strip()
            body_start = i + 1
            break
        if header is None or not _DATE.match(header):
            continue
        parts = header.split(None, 1)
        date = parts[0]
        payee = parts[1].strip() if len(parts) > 1 else ""
        postings = []
        for ln in lines[body_start:]:
            s = ln.strip()
            if not s or s.startswith(";"):
                continue
            cols = re.split(r"\s{2,}|\t+", s)
            account = cols[0].strip()
            amount = _to_amount(cols[1]) if len(cols) > 1 else None
            if account:
                postings.append({"account": account, "amount": amount})
        if postings:
            transactions.append({"date": date, "payee": payee,
                                 "postings": postings})
    return {"transactions": transactions}


def _balanced(txn: Dict) -> bool:
    """Double-entry sanity: >=2 postings and at most one elided amount; if none
    elided, the explicit amounts must (signed) net to ~0."""
    ps = txn["postings"]
    if len(ps) < 2:
        return False
    elided = [p for p in ps if p["amount"] is None]
    if len(elided) > 1:
        return False
    if len(elided) == 1:
        return True  # the blank posting absorbs the balance
    total = sum(p["amount"] for p in ps)
    scale = max((abs(p["amount"]) for p in ps), default=1.0)
    return abs(total) <= 0.01 * max(1.0, scale)


def _payees(doc: Dict):
    return {_norm(t["payee"]) for t in doc["transactions"]}


def _accounts(doc: Dict):
    return {_norm(p["account"]) for t in doc["transactions"]
            for p in t["postings"]}


def _txn_key(t: Dict):
    return (t["date"].replace("-", "/"), _norm(t["payee"]))


def _txn_total(t: Dict) -> float:
    return round(sum(abs(p["amount"]) for p in t["postings"]
                     if p["amount"] is not None), 2)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _match_count(seed: Dict, cur: Dict, with_amount: bool) -> int:
    """Greedy multiset match of seed transactions against current ones."""
    pool: Dict = {}
    for t in cur["transactions"]:
        key = _txn_key(t) + ((_txn_total(t),) if with_amount else ())
        pool[key] = pool.get(key, 0) + 1
    matched = 0
    for t in seed["transactions"]:
        key = _txn_key(t) + ((_txn_total(t),) if with_amount else ())
        if pool.get(key, 0) > 0:
            pool[key] -= 1
            matched += 1
    return matched


def runtime_risk(seed: Dict, cur: Dict) -> Dict:
    """Key-free corruption signal: current vs SEED on conserved quantities."""
    n_seed = len(seed["transactions"])
    n_cur = len(cur["transactions"])
    if n_cur == 0:
        return {"risk": 1.0, "parse_fail": 1.0, "txn_loss_rate": 1.0,
                "payee_dist": 1.0, "account_dist": 1.0, "balance_viol": 1.0,
                "missing_txns": [list(_txn_key(t)) for t in seed["transactions"]],
                "n_seed": n_seed, "n_cur": 0}

    matched = _match_count(seed, cur, with_amount=False)
    txn_loss_rate = (n_seed - matched) / max(1, n_seed)

    payee_dist = 1.0 - _jaccard(_payees(seed), _payees(cur))
    account_dist = 1.0 - _jaccard(_accounts(seed), _accounts(cur))

    unbalanced = sum(0 if _balanced(t) else 1 for t in cur["transactions"])
    balance_viol = unbalanced / max(1, n_cur)

    # which seed transactions went missing (for targeted repair)
    cur_keys = {}
    for t in cur["transactions"]:
        cur_keys[_txn_key(t)] = cur_keys.get(_txn_key(t), 0) + 1
    missing = []
    for t in seed["transactions"]:
        k = _txn_key(t)
        if cur_keys.get(k, 0) > 0:
            cur_keys[k] -= 1
        else:
            missing.append(list(k))

    risk = (0.35 * 0.0 + 0.30 * txn_loss_rate + 0.15 * payee_dist
            + 0.10 * account_dist + 0.15 * balance_viol)
    return {
        "risk": round(max(0.0, min(1.0, risk)), 4),
        "parse_fail": 0.0,
        "txn_loss_rate": round(txn_loss_rate, 4),
        "payee_dist": round(payee_dist, 4),
        "account_dist": round(account_dist, 4),
        "balance_viol": round(balance_viol, 4),
        "missing_txns": missing[:25],
        "n_seed": n_seed, "n_cur": n_cur,
    }


def score(seed: Dict, cur: Dict) -> Dict:
    """Gold metric (§7): 0.5*txn-match + 0.3*balance-preserved + 0.2*account-set."""
    n_seed = len(seed["transactions"])
    n_cur = len(cur["transactions"])
    if n_seed == 0:
        return {"score": 0.0, "txn_match": 0.0, "balance_preserved": 0.0,
                "account_set": 0.0}
    matched = _match_count(seed, cur, with_amount=True)
    txn_match = matched / max(n_seed, n_cur, 1)
    balance_preserved = (sum(1 for t in cur["transactions"] if _balanced(t))
                         / n_cur) if n_cur else 0.0
    account_set = _jaccard(_accounts(seed), _accounts(cur))
    s = 0.5 * txn_match + 0.3 * balance_preserved + 0.2 * account_set
    return {"score": round(s, 4), "txn_match": round(txn_match, 4),
            "balance_preserved": round(balance_preserved, 4),
            "account_set": round(account_set, 4)}


def repair_view(report: Dict) -> Dict:
    return {"missing_transactions": report.get("missing_txns", []),
            "seed_transaction_count": report.get("n_seed"),
            "current_transaction_count": report.get("n_cur"),
            "unbalanced_fraction": report.get("balance_viol")}
