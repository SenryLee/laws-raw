"""离线双库构建管线：将本地法律法规库构建为只读 laws_core.db 与 laws_ext.db。

用法：
  # 1. 构建内置演示样例库（用于无全量数据时的端到端验证）
  python3 builder/build_db.py --demo

  # 2. 从本地完整法规库构建核心库（约 3.4 万份，推荐首期上线）
  python3 builder/build_db.py --source "/Users/senrylee/个人资料/法律法规数据库/laws" --scope core

  # 3. 同时构建核心库 + 扩展库（全量 35.9 万份）
  python3 builder/build_db.py --source "/Users/senrylee/个人资料/法律法规数据库/laws" --scope both
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

LAWS_ROOT = Path(__file__).resolve().parent.parent
if str(LAWS_ROOT) not in sys.path:
    sys.path.insert(0, str(LAWS_ROOT))

from builder.parse_laws import ParsedLaw, parse_law_file, parse_law_markdown  # noqa: E402
from core.config import load_config  # noqa: E402
from core.legal.chunk_law import chunk_law_text  # noqa: E402
from core.legal.normalize import extract_citations, extract_parent_laws, normalize_law_name  # noqa: E402
from core.store import connect_rw, init_laws_db  # noqa: E402
from core.tokenizer import Tokenizer  # noqa: E402


def _seed_entities_and_aliases(conn: sqlite3.Connection, aliases_path: Path) -> tuple[int, int]:
    """构建全量法名实体注册表（无 800 上限）与别名映射（标题变体 + 人工种子表）。"""
    entity_ids: dict[str, int] = {}

    def get_or_add(canon: str) -> int:
        if canon not in entity_ids:
            conn.execute("INSERT OR IGNORE INTO legal_entities(canonical) VALUES(?)", (canon,))
            row = conn.execute("SELECT id FROM legal_entities WHERE canonical=?", (canon,)).fetchone()
            entity_ids[canon] = int(row["id"])
        return entity_ids[canon]

    # 1. 所有已入库法规的全称与规范名
    for row in conn.execute("SELECT DISTINCT title, law_canon FROM documents"):
        canon = row["law_canon"]
        title = row["title"]
        if not canon:
            continue
        eid = get_or_add(canon)
        if title and title != canon:
            conn.execute(
                "INSERT OR IGNORE INTO entity_aliases(entity_id, alias, source) VALUES(?,?,?)",
                (eid, title, "title"),
            )

    # 2. 种子别名表（支持简称 → 完整法名）
    if aliases_path.is_file():
        with aliases_path.open(encoding="utf-8") as fh:
            for row_ in csv.reader(fh):
                if len(row_) >= 2:
                    alias = row_[0].strip()
                    canon = normalize_law_name(row_[1].strip())
                    if alias and canon:
                        eid = get_or_add(canon)
                        conn.execute(
                            "INSERT OR IGNORE INTO entity_aliases(entity_id, alias, source) VALUES(?,?,?)",
                            (eid, alias, "seed"),
                        )

    n_ent = conn.execute("SELECT COUNT(*) FROM legal_entities").fetchone()[0]
    n_ali = conn.execute("SELECT COUNT(*) FROM entity_aliases").fetchone()[0]
    return int(n_ent), int(n_ali)


def _ingest_one(
    conn: sqlite3.Connection,
    tok: Tokenizer,
    law: ParsedLaw,
) -> int:
    cur = conn.execute(
        """INSERT INTO documents(
            rel_path, title, law_canon, doc_no, issuing_org, raw_level,
            category, scope, authority_level, status, status_raw, topic,
            publish_date, effective_date, article_count, completeness, content_hash
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            law.rel_path, law.title, law.law_canon, law.doc_no, law.issuing_org, law.raw_level,
            law.category.code, law.category.scope, law.category.authority_level,
            law.status, law.status_raw, law.topic,
            law.publish_date, law.effective_date, law.article_count, law.completeness, law.content_hash,
        ),
    )
    doc_id = int(cur.lastrowid)

    # 母法链接（司法解释/实施条例 → 母法）
    for parent in extract_parent_laws(law.title, law.body):
        if parent != law.law_canon:
            conn.execute(
                "INSERT OR IGNORE INTO doc_parent_links(child_doc_id, parent_law) VALUES(?,?)",
                (doc_id, parent),
            )

    chunks = chunk_law_text(law.body, law.title)
    fts_title = tok.fts_body(f"{law.title} {law.doc_no}")
    for ch in chunks:
        c_cur = conn.execute(
            """INSERT INTO chunks(
                document_id, seq, heading, content, content_hash,
                law_name, article_no, clause_path, lead_context
            ) VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                doc_id, ch.seq, ch.heading, ch.content, ch.content_hash,
                ch.law_name, ch.article_no, ch.clause_path, ch.lead_context,
            ),
        )
        cid = int(c_cur.lastrowid)
        conn.execute(
            "INSERT INTO chunk_fts(rowid, title, heading, body) VALUES(?,?,?,?)",
            (cid, fts_title, tok.fts_body(ch.heading), tok.fts_body(ch.content)),
        )
        for to_law, to_art in extract_citations(ch.content):
            if to_law != ch.law_name or to_art != ch.article_no:
                conn.execute(
                    "INSERT INTO chunk_citations(from_chunk, to_law, to_article) VALUES(?,?,?)",
                    (cid, to_law, to_art),
                )
    return len(chunks)


def _finalize_db(conn: sqlite3.Connection, tmp_path: Path, final_path: Path, aliases_path: Path, scope: str) -> dict:
    n_ent, n_ali = _seed_entities_and_aliases(conn, aliases_path)
    n_docs = int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
    n_chunks = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    n_cits = int(conn.execute("SELECT COUNT(*) FROM chunk_citations").fetchone()[0])
    now_iso = time.strftime("%Y-%m-%d %H:%M:%S")
    conn.executemany(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES(?,?)",
        [
            ("scope", scope),
            ("built_at", now_iso),
            ("documents", str(n_docs)),
            ("chunks", str(n_chunks)),
            ("entities", str(n_ent)),
            ("aliases", str(n_ali)),
            ("citations", str(n_cits)),
        ],
    )
    conn.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('optimize')")
    conn.commit()
    check = conn.execute("PRAGMA quick_check").fetchone()[0]
    conn.close()
    if check != "ok":
        raise RuntimeError(f"SQLite 完整性校验失败：{check}")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_path, final_path)
    return {
        "scope": scope,
        "db": str(final_path),
        "size_mb": round(final_path.stat().st_size / (1024 * 1024), 2),
        "documents": n_docs,
        "chunks": n_chunks,
        "entities": n_ent,
        "aliases": n_ali,
        "citations": n_cits,
        "built_at": now_iso,
    }


def build_from_source(source_dir: Path, scope: str, limit: int | None = None) -> dict:
    cfg = load_config(LAWS_ROOT)
    tok = Tokenizer(cfg.legal_dict_path)
    targets = ["core", "ext"] if scope == "both" else [scope]

    conns: dict[str, tuple[sqlite3.Connection, Path, Path]] = {}
    for sc in targets:
        final_p = cfg.core_db if sc == "core" else cfg.ext_db
        tmp_p = final_p.with_suffix(".db.new")
        if tmp_p.exists():
            tmp_p.unlink()
        conn = connect_rw(tmp_p, bulk=True)
        init_laws_db(conn)
        conns[sc] = (conn, tmp_p, final_p)

    t0 = time.time()
    scanned = 0
    errors: list[dict] = []
    counts = {sc: 0 for sc in targets}

    for md_path in sorted(source_dir.rglob("*.md")):
        if limit and scanned >= limit:
            break
        scanned += 1
        try:
            law = parse_law_file(md_path, source_dir)
            if law is None:
                continue
            sc = law.category.scope
            if sc not in conns:
                continue
            _ingest_one(conns[sc][0], tok, law)
            counts[sc] += 1
            if scanned % 2000 == 0:
                for c, _, _ in conns.values():
                    c.commit()
                print(f"[build] 已扫描 {scanned} 份，入库 {counts} ({time.time() - t0:.1f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(md_path), "error": f"{type(exc).__name__}: {exc}"})

    reports = {}
    for sc, (conn, tmp_p, final_p) in conns.items():
        reports[sc] = _finalize_db(conn, tmp_p, final_p, cfg.aliases_path, sc)

    return {
        "scanned": scanned,
        "elapsed_s": round(time.time() - t0, 1),
        "errors_count": len(errors),
        "errors_sample": errors[:20],
        "databases": reports,
    }


def _demo_samples() -> list[tuple[str, str, str]]:
    """内置全真格式演示语料（含现行有效、已废止旧法、母法与司法解释联动）。"""
    return [
        ("法律/2020/民法典.md", "法律", """---
标题: 中华人民共和国民法典
文号: 中华人民共和国主席令第四十五号
制定机关: 全国人民代表大会
效力等级: 法律
时效性: 现行有效
主题分类: 民法商法总类
公布日期: 2020.05.28
施行日期: 2021.01.01
条数: 1260
完整性: 全文
---

# 中华人民共和国民法典

第一编 总则

第九章 诉讼时效

第一百八十八条 向人民法院请求保护民事权利的诉讼时效期间为三年。法律另有规定的，依照其规定。
诉讼时效期间自权利人知道或者应当知道权利受到损害以及义务人之日起计算。法律另有规定的，依照其规定。但是，自权利受到损害之日起超过二十年的，人民法院不予保护，有特殊情况的，人民法院可以根据权利人的申请决定延长。

第一百九十二条 诉讼时效期间届满的，义务人可以提出不履行义务的抗辩。诉讼时效期间届满后，义务人同意履行的，不得以诉讼时效期间届满为由抗辩；义务人已经自愿履行的，不得请求返还。

第二编 物权

第十七章 抵押权

第四百零二条 以本法第三百九十五条第一款第一项至第三项规定的财产或者第五项规定的正在建造的建筑物抵押的，应当办理抵押登记。抵押权自登记时设立。

第三编 合同

第八章 违约责任

第五百七十七条 当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。

第五百八十五条 当事人可以约定一方违约时应当根据违约情况向对方支付一定数额的违约金，也可以约定因违约产生的损失赔偿额的计算方法。约定的违约金低于造成的损失的，人民法院或者仲裁机构可以根据当事人的请求予以增加；约定的违约金过分高于造成的损失的，人民法院或者仲裁机构可以根据当事人的请求予以适当减少。

第十四章 租赁合同

第七百零八条 出租人应当按照约定将租赁物交付承租人，并在租赁期限内保持租赁物符合约定的用途。

第七百二十四条 有下列情形之一，非因承租人原因致使租赁物无法使用的，承租人可以解除合同：（一）租赁物被司法机关或者行政机关依法查封、扣押；（二）租赁物权属有争议；（三）租赁物具有违反法律、行政法规关于使用条件的强制性规定情形。
"""),
        ("司法解释/2020/民间借贷规定.md", "司法解释", """---
标题: 最高人民法院关于审理民间借贷案件适用法律若干问题的规定
文号: 法释〔2020〕17号
制定机关: 最高人民法院
效力等级: 司法解释
时效性: 现行有效
主题分类: 民事诉讼与民间借贷
公布日期: 2020.12.29
施行日期: 2021.01.01
条数: 32
完整性: 全文
---

# 最高人民法院关于审理民间借贷案件适用法律若干问题的规定

为正确审理民间借贷纠纷案件，根据《中华人民共和国民法典》《中华人民共和国民事诉讼法》等相关法律之规定，结合审判实践，制定本规定。

第二条 出借人向人民法院提起民间借贷诉讼时，应当提供借据、收据、欠条等债权凭证以及其他能够证明借贷法律关系存在的证据。

第十六条 原告仅依据金融机构的转账凭证提起民间借贷诉讼，被告抗辩转账系偿还双方之前借款或者其他债务的，被告应当对其主张提供证据证明。被告提供相应证据证明其主张后，原告仍应就借贷关系的成立承担举证责任。

第二十五条 出借人请求借款人按照合同约定利率支付利息的，人民法院应予支持，但是双方约定的利率超过合同成立时一年期贷款市场报价利率四倍的除外。
"""),
        ("法律/1999/合同法_已废止.md", "法律", """---
标题: 中华人民共和国合同法
文号: 中华人民共和国主席令第十五号
制定机关: 全国人民代表大会
效力等级: 法律
时效性: 已废止
主题分类: 民法商法总类
公布日期: 1999.03.15
施行日期: 1999.10.01
条数: 428
完整性: 全文
---

# 中华人民共和国合同法

第七章 违约责任

第一百零七条 当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。
"""),
        ("法律/2023/公司法.md", "法律", """---
标题: 中华人民共和国公司法
文号: 中华人民共和国主席令第十五号
制定机关: 全国人大常委会
效力等级: 法律
时效性: 现行有效
主题分类: 商法公司法
公布日期: 2023.12.29
施行日期: 2024.07.01
条数: 266
完整性: 全文
---

# 中华人民共和国公司法

第三章 有限责任公司的设立和组织机构

第四十七条 有限责任公司的注册资本为在公司登记机关登记的全体股东认缴的出资额。全体股东认缴的出资额由股东按照公司章程的规定自公司成立之日起五年内缴足。

第五十四条 公司不能清偿到期债务的，公司或者已到期债权的债权人有权要求已认缴出资但未届出资期限的股东提前缴纳出资。
"""),
        ("法律/2012/劳动合同法.md", "法律", """---
标题: 中华人民共和国劳动合同法
文号: 中华人民共和国主席令第七十三号
制定机关: 全国人大常委会
效力等级: 法律
时效性: 现行有效
主题分类: 劳动与社会保障
公布日期: 2012.12.28
施行日期: 2013.07.01
条数: 98
完整性: 全文
---

# 中华人民共和国劳动合同法

第四章 劳动合同的解除和终止

第四十七条 经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资的标准向劳动者支付。六个月以上不满一年的，按一年计算；不满六个月的，向劳动者支付半个月工资的经济补偿。

第八十七条 用人单位违反本法规定解除或者终止劳动合同的，应当依照本法第四十七条规定的经济补偿标准的二倍向劳动者支付赔偿金。
"""),
    ]


def build_demo() -> dict:
    cfg = load_config(LAWS_ROOT)
    tok = Tokenizer(cfg.legal_dict_path)
    tmp_p = cfg.core_db.with_suffix(".db.new")
    if tmp_p.exists():
        tmp_p.unlink()
    conn = connect_rw(tmp_p, bulk=True)
    init_laws_db(conn)
    for rel, top_dir, text in _demo_samples():
        law = parse_law_markdown(text, rel_path=rel, top_dir=top_dir)
        if law:
            _ingest_one(conn, tok, law)
    rep = _finalize_db(conn, tmp_p, cfg.core_db, cfg.aliases_path, "core")
    return {"mode": "demo", "databases": {"core": rep}}


def main() -> None:
    ap = argparse.ArgumentParser(description="构建法锤官网法律法规 SQLite 索引库")
    ap.add_argument("--source", help="本地 laws/ 根目录路径")
    ap.add_argument("--scope", choices=["core", "ext", "both"], default="core", help="构建范围（默认 core）")
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 个文件（调试用）")
    ap.add_argument("--demo", action="store_true", help="构建内置演示样例库")
    args = ap.parse_args()

    if args.demo:
        rep = build_demo()
    else:
        if not args.source:
            ap.error("请指定 --source <laws目录> 或使用 --demo")
        src = Path(args.source).expanduser().resolve()
        if not src.is_dir():
            ap.error(f"源目录不存在：{src}")
        rep = build_from_source(src, scope=args.scope, limit=args.limit)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
