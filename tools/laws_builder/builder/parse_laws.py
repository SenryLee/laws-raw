"""法律法规 YAML Markdown 解析器。

适配经 clean_laws.py 清洗后的标准格式：
---
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
# 标题
正文...
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from core.legal.categories import CategoryDef, map_category, normalize_status
from core.legal.normalize import normalize_law_name

_KEY_RE = re.compile(r"^([^\s:#][^:]*):(.*)$")


@dataclass
class ParsedLaw:
    rel_path: str
    title: str
    law_canon: str
    doc_no: str
    issuing_org: str
    raw_level: str
    category: CategoryDef
    status: str
    status_raw: str
    topic: str
    publish_date: str
    effective_date: str
    article_count: int
    completeness: str
    content_hash: str
    body: str


def _unquote(val: str) -> str:
    v = val.strip()
    if len(v) >= 2 and v[0] == v[-1] == "'":
        return v[1:-1].replace("''", "'")
    if len(v) >= 2 and v[0] == v[-1] == '"':
        return v[1:-1]
    return v


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """零外部依赖快速解析单层 YAML Front Matter（比 yaml.safe_load 快 10x+）。"""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end_idx = None
    for i in range(1, min(len(lines), 60)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return {}, text

    meta: dict[str, str] = {}
    for line in lines[1:end_idx]:
        m = _KEY_RE.match(line)
        if m:
            k = m.group(1).strip()
            v = _unquote(m.group(2))
            meta[k] = v
    body = "\n".join(lines[end_idx + 1 :]).strip()
    # 去掉与标题重复的首行 H1
    if body.startswith("# "):
        _, _, rest = body.partition("\n")
        body = rest.strip()
    return meta, body


def parse_law_markdown(text: str, rel_path: str, top_dir: str = "") -> ParsedLaw | None:
    meta, body = _parse_front_matter(text)
    if not body:
        return None
    title = (meta.get("标题") or Path(rel_path).stem).strip()
    if not title:
        return None

    raw_level = (meta.get("效力等级") or top_dir or "").strip()
    cat = map_category(raw_level, top_dir)
    status_raw = (meta.get("时效性") or "").strip()
    status = normalize_status(status_raw)
    try:
        art_cnt = int((meta.get("条数") or "0").strip())
    except ValueError:
        art_cnt = 0

    return ParsedLaw(
        rel_path=rel_path,
        title=title,
        law_canon=normalize_law_name(title) or title,
        doc_no=(meta.get("文号") or "").strip(),
        issuing_org=(meta.get("制定机关") or "").strip(),
        raw_level=raw_level,
        category=cat,
        status=status,
        status_raw=status_raw or "现行有效",
        topic=(meta.get("主题分类") or "").strip(),
        publish_date=(meta.get("公布日期") or "").strip(),
        effective_date=(meta.get("施行日期") or "").strip(),
        article_count=art_cnt,
        completeness=(meta.get("完整性") or "").strip(),
        content_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        body=body,
    )


def parse_law_file(path: Path, root: Path) -> ParsedLaw | None:
    rel = str(path.relative_to(root))
    parts = Path(rel).parts
    top_dir = parts[0] if len(parts) >= 2 else ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_law_markdown(text, rel_path=rel, top_dir=top_dir)
