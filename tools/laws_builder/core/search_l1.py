"""L1 极速检索层（< 50ms，0 外部 API 成本）：
法条直达 → 法名/文号匹配 → 低频核心词 AND 层 → OR 层 → 位阶与时效性综合排序。
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from .legal.anchor import alias_expand, parse_article_query, resolve_anchors
from .legal.categories import CATEGORY_BY_CODE, STATUS_LABELS
from .store import open_ro
from .tokenizer import Tokenizer, match_and, match_or, normalize

PER_PAGE = 20
MAX_PAGE = 5  # 公网防遍历：最多翻前 5 页（100 条）
POOL_PER_TIER = 200
AND_MAX_DF = 0.08
ANCHOR_LIMIT = 20

_NOISE_RE = re.compile(r"[\s《》〈〉「」『』\"'_\-+·.,，。、()（）\[\]【】:：;；!！?？~～&]+")


@dataclass
class LawHit:
    chunk_id: int
    document_id: int
    scope: str
    title: str
    law_name: str
    doc_no: str
    issuing_org: str
    category: str
    category_label: str
    authority_level: int
    status: str
    status_label: str
    topic: str
    publish_date: str
    effective_date: str
    heading: str
    clause_path: str | None
    article_no: int | None
    content: str
    lead_context: str = ""
    citation_text: str = ""
    match: str = "or"     # anchor | title | and | or | deep
    score: float = 0.0
    bm25: float | None = None
    reason: str = ""
    anchor: bool = False
    rank: float = 0.0


@dataclass
class L1Result:
    query: str
    terms: list[str] = field(default_factory=list)
    anchors: list = field(default_factory=list)
    page: int = 1
    per_page: int = PER_PAGE
    total_documents: int = 0
    total_pages: int = 0
    elapsed_ms: int = 0
    results: list[LawHit] = field(default_factory=list)


def format_citation(title: str, clause_path: str | None, content: str) -> str:
    clean_title = title.strip().strip("《》")
    body = (content or "").strip()
    if clause_path:
        if body.startswith(clause_path):
            return f"《{clean_title}》{body}"
        return f"《{clean_title}》{clause_path}：{body}"
    return f"《{clean_title}》：{body}"


def _title_key(s: str) -> str:
    return _NOISE_RE.sub("", normalize(s or ""))


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", normalize(s or ""))


def _bigrams(s: str) -> set[str]:
    sq = _squash(s)
    return {sq[i : i + 2] for i in range(len(sq) - 1)} if len(sq) >= 2 else ({sq} if sq else set())


def _coverage(terms: list[str], text: str) -> float:
    if not terms:
        return 0.0
    norm = normalize(text or "")
    return sum(1 for t in terms if t in norm) / len(terms)


def _char_coverage(qbi: set[str], text: str) -> float:
    if not qbi:
        return 0.0
    sq = _squash(text)
    return sum(1 for b in qbi if b in sq) / len(qbi)


def _row_to_hit(r, scope: str, match: str, bm25: float | None = None, anchor: bool = False) -> LawHit:
    cat_code = r["category"]
    cat_def = CATEGORY_BY_CODE.get(cat_code)
    st = r["status"]
    title = r["title"]
    clause = r["clause_path"]
    content = r["content"]
    return LawHit(
        chunk_id=int(r["chunk_id"]),
        document_id=int(r["document_id"]),
        scope=scope,
        title=title,
        law_name=r["law_name"],
        doc_no=r["doc_no"] or "",
        issuing_org=r["issuing_org"] or "",
        category=cat_code,
        category_label=cat_def.label if cat_def else cat_code,
        authority_level=int(r["authority_level"]),
        status=st,
        status_label=r["status_raw"] or STATUS_LABELS.get(st, st),
        topic=r["topic"] or "",
        publish_date=r["publish_date"] or "",
        effective_date=r["effective_date"] or "",
        heading=r["heading"] or "",
        clause_path=clause,
        article_no=r["article_no"],
        content=content,
        lead_context=r["lead_context"] or "",
        citation_text=format_citation(title, clause, content),
        match=match,
        bm25=bm25,
        anchor=anchor,
    )


def _df_ratio(conn: sqlite3.Connection, term: str, total: int) -> float:
    row = conn.execute(
        "SELECT doc FROM chunk_vocab WHERE term = ? AND col = 'body'", (term,)
    ).fetchone()
    return (int(row["doc"]) if row else 0) / max(1, total)


def _bm25_query(
    conn: sqlite3.Connection,
    expr: str,
    status: str | None,
    category: str | None,
    year: str | None,
    limit: int,
):
    if not expr:
        return []
    wheres = []
    args: list = [expr, limit * 4]
    if status and status != "all":
        wheres.append("d.status = ?")
        args.append(status)
    if category and category != "all":
        wheres.append("d.category = ?")
        args.append(category)
    if year and year != "all":
        wheres.append("d.publish_date LIKE ?")
        args.append(f"{year}%")
    where_sql = (" AND " + " AND ".join(wheres)) if wheres else ""
    args.append(limit)
    return conn.execute(
        f"""WITH hits AS (
              SELECT rowid AS chunk_id, bm25(chunk_fts, 2.5, 2.0, 1.0) AS score
              FROM chunk_fts WHERE chunk_fts MATCH ?
              ORDER BY score ASC, rowid ASC LIMIT ?
            )
            SELECT h.chunk_id, h.score, c.document_id, c.heading, c.content, c.lead_context,
                   c.law_name, c.article_no, c.clause_path, c.seq,
                   d.title, d.doc_no, d.issuing_org, d.category, d.authority_level,
                   d.status, d.status_raw, d.topic, d.publish_date, d.effective_date
            FROM hits h
            JOIN chunks c ON c.id = h.chunk_id
            JOIN documents d ON d.id = c.document_id
            WHERE 1 = 1 {where_sql}
            ORDER BY h.score ASC, d.authority_level ASC, h.chunk_id ASC
            LIMIT ?""",
        args,
    ).fetchall()


def _search_single_db(
    db_path: Path,
    scope: str,
    tok: Tokenizer,
    query: str,
    *,
    status: str = "effective",
    category: str | None = None,
    year: str | None = None,
) -> tuple[list[LawHit], list[LawHit], list[str], list]:
    refs, core = parse_article_query(query)
    named = [r for r in refs if r.law_raw]
    anonymous = [r for r in refs if not r.law_raw]
    anchors_meta = [[r.law_raw, r.article_no] for r in refs]

    with open_ro(db_path) as conn:
        # 1) 法条直达
        anchor_hits: list[LawHit] = []
        anchor_rows = list(resolve_anchors(conn, named, status_filter=status, category_filter=category)) if named else []
        core_terms = tok.query_terms(core) if core else []
        if anonymous and not core_terms:
            anchor_rows += resolve_anchors(conn, anonymous, status_filter=status, category_filter=category)
        for r in anchor_rows[:ANCHOR_LIMIT]:
            if year and year != "all" and not (r["publish_date"] or "").startswith(year):
                continue
            h = _row_to_hit(r, scope, "anchor", anchor=True)
            h.score = 1.0
            h.rank = 300.0 - h.authority_level * 3.0
            anchor_hits.append(h)

        # 2) 正文检索词 + 别名扩展
        text_src = query if (named and not anchor_hits) else core
        terms = tok.query_terms(text_src) if text_src else []
        for canon in alias_expand(conn, query):
            for t in tok.query_terms(canon):
                if t not in terms:
                    terms.append(t)

        pool: dict[int, LawHit] = {}
        if terms:
            total = int(conn.execute("SELECT COALESCE(MAX(id), 0) FROM chunks").fetchone()[0]) or 1
            and_terms = [t for t in terms if _df_ratio(conn, t, total) <= AND_MAX_DF]
            if and_terms:
                for r in _bm25_query(conn, match_and(and_terms), status, category, year, POOL_PER_TIER):
                    pool.setdefault(int(r["chunk_id"]), _row_to_hit(r, scope, "and", bm25=float(r["score"])))
            for r in _bm25_query(conn, match_or(terms), status, category, year, POOL_PER_TIER):
                pool.setdefault(int(r["chunk_id"]), _row_to_hit(r, scope, "or", bm25=float(r["score"])))

        # 3) 标题 / 文号精确与包含匹配
        q_key = _title_key(core if core else query)
        title_boost: dict[int, int] = {}
        if len(q_key) >= 2:
            wheres = ["(instr(law_canon, ?) > 0 OR instr(title, ?) > 0 OR instr(doc_no, ?) > 0)"]
            args: list = [q_key, (core or query).strip(), (core or query).strip()]
            if status and status != "all":
                wheres.append("status = ?")
                args.append(status)
            if category and category != "all":
                wheres.append("category = ?")
                args.append(category)
            if year and year != "all":
                wheres.append("publish_date LIKE ?")
                args.append(f"{year}%")
            doc_rows = conn.execute(
                f"SELECT id, title, law_canon, doc_no FROM documents WHERE {' AND '.join(wheres)} ORDER BY authority_level ASC LIMIT 30",
                args,
            ).fetchall()
            in_pool_docs = {h.document_id for h in pool.values()}
            for dr in doc_rows:
                did = int(dr["id"])
                exact = _title_key(dr["law_canon"]) == q_key or _title_key(dr["title"]) == q_key or (dr["doc_no"] and q_key == _title_key(dr["doc_no"]))
                title_boost[did] = 2 if exact else 1
                if did not in in_pool_docs:
                    c_row = conn.execute(
                        """SELECT c.id AS chunk_id, c.document_id, c.heading, c.content, c.lead_context,
                                  c.law_name, c.article_no, c.clause_path, c.seq,
                                  d.title, d.doc_no, d.issuing_org, d.category, d.authority_level,
                                  d.status, d.status_raw, d.topic, d.publish_date, d.effective_date
                           FROM chunks c JOIN documents d ON d.id = c.document_id
                           WHERE c.document_id = ? ORDER BY c.seq ASC LIMIT 1""",
                        (did,),
                    ).fetchone()
                    if c_row:
                        pool.setdefault(int(c_row["chunk_id"]), _row_to_hit(c_row, scope, "title"))

        # 4) 综合重排：AND层 + 标题匹配 + 词/二元组覆盖率 + BM25 + 效力位阶 + 时效性加权
        anchor_ids = {h.chunk_id for h in anchor_hits}
        neg_max = max((-h.bm25 for h in pool.values() if h.bm25 is not None), default=0.0) or 1.0
        qbi = _bigrams(core if core else query)
        best_by_doc: dict[int, LawHit] = {}

        for h in pool.values():
            if h.chunk_id in anchor_ids:
                continue
            t_lvl = title_boost.get(h.document_id, 0)
            rel = (-h.bm25 / neg_max) if h.bm25 is not None else 0.0
            auth_bonus = max(0.0, (8 - h.authority_level) * 3.0)
            status_bonus = 18.0 if h.status == "effective" else (6.0 if h.status == "not_yet_effective" else -20.0)
            h.rank = (
                (45.0 if h.match == "and" else 0.0)
                + (90.0 if t_lvl == 2 else 15.0 if t_lvl == 1 else 0.0)
                + 30.0 * _coverage(terms, h.content)
                + 20.0 * _char_coverage(qbi, h.content)
                + 12.0 * max(0.0, min(1.0, rel))
                + auth_bonus
                + status_bonus
            )
            cur = best_by_doc.get(h.document_id)
            if cur is None or (h.rank, -h.chunk_id) > (cur.rank, -cur.chunk_id):
                best_by_doc[h.document_id] = h

        docs = sorted(best_by_doc.values(), key=lambda x: (-x.rank, x.authority_level, x.chunk_id))
        for h in docs:
            h.score = round(max(0.05, min(0.99, h.rank / 210.0)), 3)

    return anchor_hits, docs, terms, anchors_meta


def search_l1(
    core_db: Path,
    ext_db: Path | None,
    tok: Tokenizer,
    query: str,
    *,
    page: int = 1,
    per_page: int = PER_PAGE,
    status: str = "effective",
    category: str | None = None,
    year: str | None = None,
    include_ext: bool = False,
) -> L1Result:
    t0 = time.time()
    page = max(1, min(page, MAX_PAGE))
    out = L1Result(query=query, page=page, per_page=per_page)

    dbs: list[tuple[Path, str]] = []
    # 如果用户明确选了扩展库分类，自动查 ext_db
    cat_def = CATEGORY_BY_CODE.get(category or "")
    if cat_def and cat_def.scope == "ext":
        if ext_db and ext_db.is_file():
            dbs.append((ext_db, "ext"))
    else:
        if core_db.is_file():
            dbs.append((core_db, "core"))
        if include_ext and ext_db and ext_db.is_file():
            dbs.append((ext_db, "ext"))

    if not dbs or not (query or "").strip():
        out.elapsed_ms = int((time.time() - t0) * 1000)
        return out

    all_anchors: list[LawHit] = []
    all_docs: list[LawHit] = []
    terms_out: list[str] = []
    anchors_meta: list = []

    for db_p, sc in dbs:
        ah, ds, ts, am = _search_single_db(
            db_p, sc, tok, query, status=status, category=category, year=year
        )
        all_anchors.extend(ah)
        all_docs.extend(ds)
        if not terms_out:
            terms_out = ts
        if not anchors_meta:
            anchors_meta = am

    all_anchors.sort(key=lambda x: (-x.rank, x.authority_level, x.chunk_id))
    all_docs.sort(key=lambda x: (-x.rank, x.authority_level, x.chunk_id))

    out.terms = terms_out
    out.anchors = anchors_meta
    out.total_documents = len(all_docs)
    out.total_pages = min(MAX_PAGE, (len(all_docs) + per_page - 1) // per_page) if all_docs else 0
    start = (page - 1) * per_page
    page_docs = all_docs[start : start + per_page]
    out.results = (all_anchors[:ANCHOR_LIMIT] if page == 1 else []) + page_docs
    out.elapsed_ms = int((time.time() - t0) * 1000)
    return out
