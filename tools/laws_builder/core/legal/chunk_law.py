"""法条感知分块器：按条独立成块 + 超长法条按句切分并携带首款主语 lead_context。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ..tokenizer import utf16_len
from .cn2num import cn2num
from .normalize import normalize_law_name

_NUM = "零〇一二三四五六七八九十百千0-9"
_ART_LINE = re.compile(rf"^( {{0,8}})第([{_NUM}]+)条(之([{_NUM}]+))?(?=[\s　：:，。、]*)")
_STRUCT_LINE = re.compile(rf"^( {{0,8}})第([{_NUM}]+)(编|章|节)(?=[\s　：:，。、]*)")
_MD_HEAD = re.compile(r"^( {0,3})(#{1,6})\s+(.+?)\s*#*\s*$")
_SENT_BREAK = re.compile(r"(?<=[。！？!?；;])\s*|\n+")
_STRUCT_RANK = {"编": 1, "章": 2, "节": 3}


@dataclass
class LawChunk:
    heading: str
    content: str
    content_hash: str
    law_name: str
    article_no: int | None
    clause_path: str | None
    lead_context: str
    seq: int


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_lead(article_text: str, max_chars: int = 120) -> str:
    """提取条文第一句/第一款前导主语，供超长法条子块送审 JEV 时补全上下文。"""
    first_para = article_text.strip().split("\n", 1)[0].strip()
    if len(first_para) <= max_chars:
        return first_para
    cut = first_para[:max_chars]
    for sep in ("。", "；", "，", "："):
        idx = cut.rfind(sep)
        if idx >= 30:
            return cut[: idx + 1]
    return cut + "…"


def _split_long(text: str, max_chars: int = 480) -> list[str]:
    """按句/段切分超长文本，保证每块 <= max_chars（UTF-16 码元）。"""
    parts = [p.strip() for p in _SENT_BREAK.split(text) if p.strip()]
    if not parts:
        return [text.strip()] if text.strip() else []
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in parts:
        plen = utf16_len(p)
        if not cur:
            cur = [p]
            cur_len = plen
        elif cur_len + plen + 1 <= max_chars:
            cur.append(p)
            cur_len += plen + 1
        else:
            chunks.append("\n".join(cur))
            cur = [p]
            cur_len = plen
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _strip_toc(text: str) -> str:
    """跳过开头的「## 目录」列表，避免把目录里的章节点误当正文。"""
    m_toc = re.search(r"(?m)^##\s*目\s*录\s*$", text)
    if not m_toc:
        return text
    # 找到目录后第一个「第一条」或下一个 ## 标题
    after = text[m_toc.end():]
    m_first = re.search(r"(?m)^(?:#{1,3}\s+|第[一1]编|第[一1]章|第[一1]条)", after)
    if not m_first:
        return text
    # 若紧跟的是目录里的「第一编」，找到正文里真正出现「第一条」的位置前的最近编/章
    m_art1 = re.search(r"(?m)^第[一1]条[\s　：:]", after)
    if not m_art1:
        return text
    # 往前找 400 字内最后一个编/章标题作为正文起点
    prefix = after[: m_art1.start()]
    struct_matches = list(re.finditer(r"(?m)^第[一1](?:编|章)[^\n]*$", prefix))
    if struct_matches:
        start = struct_matches[-1].start()
        return text[: m_toc.start()] + after[start:]
    return text[: m_toc.start()] + after[m_art1.start():]


def chunk_law_text(
    body: str,
    title: str,
    max_chars: int = 480,
) -> list[LawChunk]:
    """将法规正文按「条」切分为结构化 LawChunk；无条文结构时退回按段切分。"""
    clean_body = _strip_toc(body or "").strip()
    if not clean_body:
        return []

    law_canon = normalize_law_name(title) or title.strip()
    articles: list[dict] = []
    structs: list[dict] = []

    pos = 0
    n = len(clean_body)
    while pos < n:
        nl = clean_body.find("\n", pos)
        line_end = n if nl == -1 else nl + 1
        line = clean_body[pos : line_end if nl == -1 else nl]
        stripped = line.lstrip()
        leading = len(line) - len(stripped)
        if leading <= 8 and not _MD_HEAD.match(line):
            am = _ART_LINE.match(stripped)
            if am:
                no = cn2num(am.group(2))
                if no is not None:
                    suffix = f"之{am.group(4)}" if am.group(4) else ""
                    label = stripped.split("条", 1)[0] + "条" + suffix
                    articles.append({"start": pos, "end": line_end, "no": no, "label": label})
                    pos = line_end
                    continue
            sm = _STRUCT_LINE.match(stripped)
            if sm:
                no = cn2num(sm.group(2))
                if no is not None:
                    rank = _STRUCT_RANK.get(sm.group(3), 9)
                    structs.append({
                        "start": pos, "end": line_end, "no": no,
                        "label": stripped.strip(), "rank": rank,
                    })
                    pos = line_end
                    continue
        pos = line_end

    chunks: list[LawChunk] = []
    seq = 0

    if len(articles) < 2:
        # 非条文体裁（如批复、通知、工作文件）：按段落/字数切块
        for sub in _split_long(clean_body, max_chars=max_chars):
            seq += 1
            chunks.append(LawChunk(
                heading=title,
                content=sub,
                content_hash=_sha256(sub),
                law_name=law_canon,
                article_no=None,
                clause_path=None,
                lead_context="",
                seq=seq,
            ))
        return chunks

    events = sorted(
        [(a["start"], "article", a) for a in articles]
        + [(s["start"], "struct", s) for s in structs],
        key=lambda x: x[0],
    )
    struct_stack: list[tuple[int, str]] = []

    for i, (start, kind, ev) in enumerate(events):
        if kind == "struct":
            rank = ev["rank"]
            while struct_stack and struct_stack[-1][0] >= rank:
                struct_stack.pop()
            struct_stack.append((rank, ev["label"]))
            continue

        next_start = events[i + 1][0] if i + 1 < len(events) else n
        seg = clean_body[start:next_start].strip()
        if not seg:
            continue
        chap = " / ".join(lbl for _, lbl in struct_stack)
        label = ev["label"]
        heading = f"{title} / {chap} / {label}" if chap else f"{title} / {label}"

        if utf16_len(seg) <= max_chars:
            seq += 1
            chunks.append(LawChunk(
                heading=heading,
                content=seg,
                content_hash=_sha256(seg),
                law_name=law_canon,
                article_no=ev["no"],
                clause_path=label,
                lead_context="",
                seq=seq,
            ))
        else:
            lead = _extract_lead(seg)
            subs = _split_long(seg, max_chars=max_chars)
            for idx, sub in enumerate(subs):
                seq += 1
                chunks.append(LawChunk(
                    heading=heading,
                    content=sub,
                    content_hash=_sha256(sub),
                    law_name=law_canon,
                    article_no=ev["no"],
                    clause_path=label,
                    lead_context="" if idx == 0 else lead,
                    seq=seq,
                ))
    return chunks
