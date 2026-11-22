"""查询侧法条直达锚点解析与别名扩展。"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from .cn2num import cn2num
from .normalize import normalize_law_name

_NUM = "零〇一二三四五六七八九十百千万0-9"
_SEP = r"[\s_\-·~～]*"
_LAW = r"《[^《》\n]{1,40}》|[^\s第，。；、,;:：！？!?\-·~～\n]{0,60}?"
_REF = re.compile(
    rf"(?P<law>{_LAW}){_SEP}第\s*(?P<no>[{_NUM}]+)\s*条"
    rf"(?:\s*之\s*(?P<sfx>[{_NUM}]+))?"
    rf"(?=$|[\s，。？！、,.?!_\-·~～的是规定内容怎么什么之款项第])"
)

_GENERIC_LAWS = frozenset({
    "", "法", "条例", "办法", "规定", "规则", "细则", "解释", "意见", "决定",
    "本法", "本条例", "本规定", "本办法", "本细则", "其", "他", "之",
})

_SELECT = """SELECT c.id AS chunk_id, c.document_id, c.heading, c.content, c.lead_context,
    c.law_name, c.article_no, c.clause_path, c.seq,
    d.title, d.doc_no, d.issuing_org, d.category, d.authority_level,
    d.status, d.status_raw, d.topic, d.publish_date, d.effective_date
    FROM chunks c JOIN documents d ON d.id = c.document_id"""


@dataclass(frozen=True)
class ArticleRef:
    law_raw: str
    article_no: int
    suffix: str | None = None


def _clean_law(raw: str) -> str:
    return (raw or "").strip().strip("《》").strip()


def parse_article_query(query: str) -> tuple[list[ArticleRef], str]:
    refs: list[ArticleRef] = []
    seen: set[ArticleRef] = set()

    def _sub(m: re.Match) -> str:
        no = cn2num(m.group("no"))
        if no is None:
            return m.group(0)
        law = _clean_law(m.group("law"))
        if law in _GENERIC_LAWS:
            ref = ArticleRef("", no, m.group("sfx"))
            if ref not in seen:
                seen.add(ref)
                refs.append(ref)
            return m.group(0)
        ref = ArticleRef(law, no, m.group("sfx"))
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
        return " "

    rest = _REF.sub(_sub, query or "")
    return refs, re.sub(r"\s+", " ", rest).strip()


def _law_candidates(law: str, limit: int = 12) -> list[str]:
    out: list[str] = []
    for i in range(len(law)):
        cand = normalize_law_name(law[i:])
        if len(cand) >= 2 and cand not in out:
            out.append(cand)
            if len(out) >= limit:
                break
    return out


def _pick_suffix(rows: list[dict], sfx: str | None) -> list[dict]:
    if not rows:
        return rows

    def has_zhi(r: dict) -> bool:
        return bool(r.get("clause_path")) and "之" in r["clause_path"]

    want = [r for r in rows if has_zhi(r) == bool(sfx)]
    return want or rows


def _status_rank(status: str | None) -> int:
    return {"effective": 0, "not_yet_effective": 1, "amended": 2, "repealed": 3}.get(status or "", 4)


def resolve_alias_targets(conn: sqlite3.Connection, law_raw: str) -> list[str]:
    """通过 legal_aliases 把简称（如「民诉解释」「公司法」）映射到完整规范法名。"""
    law = normalize_law_name(law_raw)
    if not law:
        return []
    out: list[str] = [law]
    try:
        rows = conn.execute(
            "SELECT canonical FROM legal_aliases WHERE alias = ? LIMIT 8", (law,)
        ).fetchall()
        for r in rows:
            c = r["canonical"]
            if c and c not in out:
                out.append(c)
    except sqlite3.Error:
        pass
    return out


def resolve_anchors(
    conn: sqlite3.Connection,
    refs: list[ArticleRef],
    *,
    status_filter: str | None = "effective",
    category_filter: str | None = None,
    limit_per_ref: int = 12,
) -> list[dict]:
    out: list[dict] = []
    seen: set[int] = set()

    for ref in refs:
        no = ref.article_no
        law = normalize_law_name(ref.law_raw)
        ranked: dict[int, tuple[int, int, int, int, dict]] = {}

        def add(rows, match_rank: int) -> None:
            for r in rows:
                d = dict(r)
                if status_filter and status_filter != "all" and d["status"] != status_filter:
                    continue
                if category_filter and category_filter != "all" and d["category"] != category_filter:
                    continue
                key = (match_rank, _status_rank(d["status"]), int(d["authority_level"]), len(d["title"] or ""))
                cur = ranked.get(d["chunk_id"])
                if cur is None or key < cur[:4]:
                    ranked[d["chunk_id"]] = (*key, d)

        if not law:
            add(
                conn.execute(
                    _SELECT + " WHERE c.article_no = ? ORDER BY d.authority_level ASC LIMIT 200",
                    (no,),
                ).fetchall(),
                0,
            )
        else:
            targets = resolve_alias_targets(conn, law)
            for t in targets:
                for i, cand in enumerate(_law_candidates(t)):
                    rows = conn.execute(
                        _SELECT + " WHERE c.article_no = ? AND c.law_name = ? LIMIT 200",
                        (no, cand),
                    ).fetchall()
                    if rows:
                        add(rows, 0 if i == 0 else 1)
                        break
            if len(law) >= 2:
                add(
                    conn.execute(
                        _SELECT + """ WHERE c.article_no = ? AND length(c.law_name) >= 2
                            AND (instr(?, c.law_name) > 0 OR instr(c.law_name, ?) > 0) LIMIT 200""",
                        (no, law, law),
                    ).fetchall(),
                    2,
                )

        ordered = sorted(ranked.values(), key=lambda x: (x[0], x[1], x[2], x[3], x[4]["chunk_id"]))
        picked = _pick_suffix([x[4] for x in ordered], ref.suffix)[:limit_per_ref]
        for r in picked:
            if r["chunk_id"] not in seen:
                seen.add(r["chunk_id"])
                out.append(r)
    return out


def alias_expand(conn: sqlite3.Connection, query: str, limit: int = 8) -> list[str]:
    out: list[str] = []
    try:
        for row in conn.execute("SELECT alias, canonical FROM legal_aliases LIMIT 5000"):
            alias, canon = row["alias"], row["canonical"]
            if alias and alias in query and canon and canon not in query:
                out.append(canon)
                if len(out) >= limit:
                    break
    except sqlite3.Error:
        pass
    return out
