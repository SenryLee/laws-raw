"""SQLite 存储层：laws_core.db / laws_ext.db（只读数据面）与 cache.db（独立可写运行面）。"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 1

LAWS_DDL: list[str] = [
    """CREATE TABLE IF NOT EXISTS metadata(
        key TEXT PRIMARY KEY,
        value TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS documents(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rel_path TEXT UNIQUE,
        title TEXT NOT NULL,
        law_canon TEXT NOT NULL,
        doc_no TEXT DEFAULT '',
        issuing_org TEXT DEFAULT '',
        raw_level TEXT DEFAULT '',
        category TEXT NOT NULL,
        scope TEXT NOT NULL,
        authority_level INTEGER NOT NULL DEFAULT 5,
        status TEXT NOT NULL DEFAULT 'effective',
        status_raw TEXT DEFAULT '',
        topic TEXT DEFAULT '',
        publish_date TEXT DEFAULT '',
        effective_date TEXT DEFAULT '',
        article_count INTEGER DEFAULT 0,
        completeness TEXT DEFAULT '',
        content_hash TEXT DEFAULT ''
    )""",
    """CREATE TABLE IF NOT EXISTS chunks(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        seq INTEGER NOT NULL,
        heading TEXT NOT NULL,
        content TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        law_name TEXT NOT NULL,
        article_no INTEGER,
        clause_path TEXT,
        lead_context TEXT DEFAULT ''
    )""",
    """CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
        title, heading, body, content='', contentless_delete=1
    )""",
    """CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vocab USING fts5vocab(chunk_fts, 'col')""",
    """CREATE TABLE IF NOT EXISTS legal_entities(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        canonical TEXT UNIQUE NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS entity_aliases(
        entity_id INTEGER NOT NULL REFERENCES legal_entities(id) ON DELETE CASCADE,
        alias TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'seed',
        UNIQUE(entity_id, alias)
    )""",
    """CREATE VIEW IF NOT EXISTS legal_aliases AS
        SELECT a.alias AS alias, e.canonical AS canonical, a.source AS source
        FROM entity_aliases a JOIN legal_entities e ON e.id = a.entity_id""",
    """CREATE TABLE IF NOT EXISTS chunk_citations(
        from_chunk INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
        to_law TEXT NOT NULL,
        to_article INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS doc_parent_links(
        child_doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        parent_law TEXT NOT NULL,
        UNIQUE(child_doc_id, parent_law)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_docs_cat_status ON documents(category, status, authority_level)",
    "CREATE INDEX IF NOT EXISTS idx_docs_canon ON documents(law_canon)",
    "CREATE INDEX IF NOT EXISTS idx_docs_docno ON documents(doc_no)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc_seq ON chunks(document_id, seq)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_article ON chunks(article_no, law_name)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_law ON chunks(law_name)",
    "CREATE INDEX IF NOT EXISTS idx_cit_from ON chunk_citations(from_chunk)",
    "CREATE INDEX IF NOT EXISTS idx_cit_to ON chunk_citations(to_law, to_article)",
    "CREATE INDEX IF NOT EXISTS idx_parent_law ON doc_parent_links(parent_law)",
]

CACHE_DDL: list[str] = [
    """CREATE TABLE IF NOT EXISTS rewrite_cache(
        query_hash TEXT PRIMARY KEY,
        result_json TEXT NOT NULL,
        model TEXT NOT NULL,
        created_at INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS jev_cache(
        query_hash TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        model TEXT NOT NULL,
        score REAL NOT NULL,
        reason TEXT DEFAULT '',
        created_at INTEGER NOT NULL,
        PRIMARY KEY(query_hash, content_hash, model)
    )""",
    """CREATE TABLE IF NOT EXISTS query_stats(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        day TEXT NOT NULL,
        mode TEXT NOT NULL,
        latency_ms INTEGER NOT NULL,
        hit_count INTEGER NOT NULL,
        cache_hit INTEGER NOT NULL DEFAULT 0,
        llm_calls INTEGER NOT NULL DEFAULT 0
    )""",
    "CREATE INDEX IF NOT EXISTS idx_stats_day ON query_stats(day)",
]


def connect_rw(db_path: Path, *, bulk: bool = False) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if bulk:
        conn.execute("PRAGMA journal_mode = MEMORY")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA cache_size = -64000")
    else:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -8000")
    return conn


@contextmanager
def open_ro(db_path: Path):
    """以只读 URI 打开法规库，支持线上零锁并发与原子替换热重载。"""
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA cache_size = -8000")
    try:
        yield conn
    finally:
        conn.close()


def init_laws_db(conn: sqlite3.Connection) -> None:
    for stmt in LAWS_DDL:
        conn.execute(stmt)
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def init_cache_db(db_path: Path) -> None:
    conn = connect_rw(db_path)
    try:
        for stmt in CACHE_DDL:
            conn.execute(stmt)
        conn.commit()
    finally:
        conn.close()


def cleanup_old_cache(db_path: Path, max_age_days: int = 90) -> None:
    if not db_path.exists():
        return
    cutoff = int(time.time()) - max_age_days * 86400
    conn = connect_rw(db_path)
    try:
        with conn:
            conn.execute("DELETE FROM rewrite_cache WHERE created_at < ?", (cutoff,))
            conn.execute("DELETE FROM jev_cache WHERE created_at < ?", (cutoff,))
    finally:
        conn.close()
