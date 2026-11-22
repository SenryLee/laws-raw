"""本地查询意图路由器（< 5ms）：区分明确法条/法名/短术语查询与口语化模糊查询。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .legal.anchor import parse_article_query

_FUZZY_MARKERS = re.compile(
    r"(怎么办|怎么判|怎么算|怎么处理|能不能|能否|可以吗|算不算|有没有|要不要|赔多少|"
    r"犯法吗|违法吗|合法吗|有效吗|还有效|要回来|不还钱|不退|赶我走|辞退|开除|被骗|"
    r"没签合同|没写借条|转了微信|出了车祸|离婚|抚养权|遗产|拖欠工资|？|\?)"
)
_DOC_NO_RE = re.compile(r"(?:法释|国发|国办发|主席令|令第)[\s〔\[（(]*\d")


@dataclass(frozen=True)
class RouteDecision:
    mode: str        # "exact" | "fuzzy"
    reason: str
    has_anchor: bool


def route_query(query: str) -> RouteDecision:
    q = (query or "").strip()
    if not q:
        return RouteDecision("exact", "empty", False)

    refs, rest = parse_article_query(q)
    named_refs = [r for r in refs if r.law_raw]
    if named_refs and len(rest) <= 6:
        return RouteDecision("exact", "article_anchor", True)

    if _DOC_NO_RE.search(q) and len(q) <= 24:
        return RouteDecision("exact", "doc_no", False)

    if _FUZZY_MARKERS.search(q):
        return RouteDecision("fuzzy", "colloquial_question", bool(named_refs))

    if len(q) >= 14:
        return RouteDecision("fuzzy", "long_natural_query", bool(named_refs))

    return RouteDecision("exact", "short_term_or_title", bool(named_refs))
