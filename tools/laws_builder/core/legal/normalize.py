"""法律名称归一化与条文引用提取。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .cn2num import cn2num

_PREFIXES = ("中华人民共和国",)
_STRIP_CHARS = "《》〈〉「」\"'\u201c\u201d \t"
_NUM_CLASS = "零〇一二三四五六七八九十百千万0-9"


def normalize_law_name(raw: str) -> str:
    s = (raw or "").strip().strip(_STRIP_CHARS)
    for p in _PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
    return re.sub(r"\s+", "", s)


@dataclass
class CitationRef:
    law_raw: str
    law_name: str
    article_no: int | None


_REF_BOOK = re.compile(rf"《([^《》]+?)》第([{_NUM_CLASS}]+)条")
_REF_PLAIN = re.compile(
    rf"(?:依照|依据|根据|按照|违反|适用)\s*([\u4e00-\u9fff]{{2,24}})第([{_NUM_CLASS}]+)条"
)
_INTERP_BASE = re.compile(
    r"(?:为正确审理|为了正确适用|根据|依照)[^。；\n]{0,40}?《([^《》]{2,40})》"
)


def extract_citations(text: str, limit: int = 32) -> list[tuple[str, int]]:
    """提取高置信的 (归一法名, 条号) 引用边。"""
    out: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for pat in (_REF_BOOK, _REF_PLAIN):
        for m in pat.finditer(text or ""):
            no = cn2num(m.group(2))
            canon = normalize_law_name(m.group(1))
            if no is None or not canon:
                continue
            key = (canon, no)
            if key not in seen:
                seen.add(key)
                out.append(key)
                if len(out) >= limit:
                    return out
    return out


def extract_parent_laws(title: str, head_text: str) -> list[str]:
    """从司法解释/实施条例标题与首段提取其解释的母法规范名。"""
    out: list[str] = []
    seen: set[str] = set()
    for src in (title or "", (head_text or "")[:600]):
        for m in _INTERP_BASE.finditer(src):
            canon = normalize_law_name(m.group(1))
            if canon and len(canon) >= 2 and canon not in seen:
                seen.add(canon)
                out.append(canon)
        for m in re.finditer(r"《([^《》]{2,40})》", title or ""):
            canon = normalize_law_name(m.group(1))
            if canon and len(canon) >= 2 and canon not in seen:
                seen.add(canon)
                out.append(canon)
    return out
