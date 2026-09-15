"""Busca lexical sobre protocolos."""
from __future__ import annotations

import re
import sqlite3

_TOKEN = re.compile(r"[a-z0-9]{3,}")
_STOP = frozenset(
    """
    the and for with this that does what should hospital protocol protocols
    patient patients pending exam exams assistant physician validation requires
    suggested conduct before after about there are wait labs lab say not none
    only use using question evidence summary did can could would may must next
    team review flag start here from based related clinical internal
    """.split()
)


def _tokens(text: str) -> set[str]:
    found = set(_TOKEN.findall((text or "").lower()))
    extra = {tok[:-1] for tok in found if tok.endswith("s") and len(tok) > 4}
    return (found | extra) - _STOP


def search_protocols(
    conn: sqlite3.Connection,
    query: str,
    k: int = 3,
    context: str = "",
) -> list[dict]:
    q = _tokens(query)
    ctx = _tokens(context)
    if not q and not ctx:
        return []
    scored = []
    for row in conn.execute("SELECT * FROM protocols"):
        title_kw = _tokens(f"{row['title']} {row['keywords']}")
        body = _tokens(row["body"])
        q_title = len(q & title_kw)
        q_body = len(q & body)
        ctx_title = len(ctx & title_kw)
        score = 4 * q_title + 2 * q_body + ctx_title
        if score <= 0:
            continue
        scored.append((score, q_title, dict(row)))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    docs = []
    best = scored[0][0] if scored else 0
    for score, _title_hits, row in scored[:k]:
        if score < max(2, best * 0.4) and docs:
            break
        docs.append({
            "protocol_id": row["id"],
            "title": row["title"],
            "pmid": row["pmid"],
            "snippet": row["body"][:500],
            "score": score,
        })
    return docs


def expected_hit(docs: list[dict], pmid: str, k: int = 3) -> bool:
    return any(d["pmid"] == str(pmid) for d in docs[:k])
