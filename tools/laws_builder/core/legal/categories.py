"""法规分类归并、效力位阶与核心/扩展分库映射。

将原始 50 个细碎目录/效力等级规范化为 8 个官网展示大类，并区分：
- core（核心库，约 3.4 万份）：默认检索池，无行政许可批复/部门规范性文件噪声；
- ext （扩展库，约 32.5 万份）：前端按需勾选扩展检索。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CategoryDef:
    code: str
    label: str
    scope: str          # "core" | "ext"
    authority_level: int  # 1 最高 → 7 最低
    description: str


CATEGORIES: tuple[CategoryDef, ...] = (
    CategoryDef("law", "宪法与法律", "core", 1, "宪法、全国人大及其常委会制定的法律、法律解释与有关法律问题的决定"),
    CategoryDef("admin_reg", "行政法规", "core", 2, "国务院行政法规、行政法规解释、监察法规与军事法规"),
    CategoryDef("judicial", "司法解释与指导文件", "core", 3, "最高人民法院、最高人民检察院司法解释、指导性文件与司法政务文件"),
    CategoryDef("dept_rule", "部门规章", "core", 4, "国务院各部委发布的部门规章"),
    CategoryDef("state_norm", "国务院规范性文件", "core", 4, "国务院及国务院办公厅发布的规范性文件"),
    CategoryDef("party_mil", "党内与军事文件", "core", 5, "党章、党内规范性文件、军事规章与军事规范性文件"),
    CategoryDef("dept_norm", "部门规范性文件", "ext", 6, "各部委规范性文件、办法、规定、细则、工作文件等（扩展库）"),
    CategoryDef("industry_permit", "行业规定与许可批复", "ext", 7, "行业规定、团体规定、行政许可批复及其他文件（扩展库）"),
)

CATEGORY_BY_CODE: dict[str, CategoryDef] = {c.code: c for c in CATEGORIES}

# 原始效力等级/目录名 → 规范化大类 code
_RAW_TO_CODE: dict[str, str] = {
    "宪法": "law",
    "法律": "law",
    "法律解释": "law",
    "有关法律问题的决定": "law",
    "决议": "law",
    "行政法规": "admin_reg",
    "行政法规解释": "admin_reg",
    "监察法规": "admin_reg",
    "军事法规": "admin_reg",
    "条例": "admin_reg",
    "司法解释": "judicial",
    "司法指导性文件": "judicial",
    "司法政务文件": "judicial",
    "部门规章": "dept_rule",
    "国务院规范性文件": "state_norm",
    "党章": "party_mil",
    "党内规范性文件": "party_mil",
    "军事规章": "party_mil",
    "军事规范性文件": "party_mil",
    "部门规范性文件": "dept_norm",
    "部门工作文件": "dept_norm",
    "工作文件": "dept_norm",
    "工作答复": "dept_norm",
    "办法": "dept_norm",
    "规定": "dept_norm",
    "细则": "dept_norm",
    "规则": "dept_norm",
    "准则": "dept_norm",
    "行业规定": "industry_permit",
    "团体规定": "industry_permit",
    "行政许可批复": "industry_permit",
    "任免": "industry_permit",
    "其他文件": "industry_permit",
    "未分类": "industry_permit",
}

# 时效性规范化
STATUS_MAP: dict[str, str] = {
    "现行有效": "effective",
    "有效": "effective",
    "已被修改": "amended",
    "已修改": "amended",
    "部分失效": "amended",
    "已废止": "repealed",
    "失效": "repealed",
    "已失效": "repealed",
    "尚未生效": "not_yet_effective",
    "未生效": "not_yet_effective",
}

STATUS_LABELS: dict[str, str] = {
    "effective": "现行有效",
    "amended": "已修改",
    "repealed": "已废止",
    "not_yet_effective": "尚未生效",
    "unknown": "未知",
}


def map_category(raw_level: str | None, dir_name: str | None = None) -> CategoryDef:
    """支持复合目录名（如「行政法规、条例」「部门规范性文件、规定」），按最高位阶归类。"""
    candidates: list[str] = []
    for src in (raw_level or "", dir_name or ""):
        for part in src.replace(",", "、").split("、"):
            part = part.strip()
            if part:
                candidates.append(part)

    matched: list[CategoryDef] = []
    for token in candidates:
        code = _RAW_TO_CODE.get(token)
        if code:
            matched.append(CATEGORY_BY_CODE[code])
            continue
        for k, v in _RAW_TO_CODE.items():
            if k in token:
                matched.append(CATEGORY_BY_CODE[v])
                break
    if matched:
        matched.sort(key=lambda c: c.authority_level)
        return matched[0]
    return CATEGORY_BY_CODE["industry_permit"]


def normalize_status(raw: str | None) -> str:
    s = (raw or "").strip()
    if not s:
        return "effective"  # 缺省视为有效，避免误过滤
    if s in STATUS_MAP:
        return STATUS_MAP[s]
    if "废止" in s or "失效" in s:
        return "repealed"
    if "修改" in s or "修订" in s:
        return "amended"
    if "尚未" in s or "未生效" in s:
        return "not_yet_effective"
    if "有效" in s:
        return "effective"
    return "effective"
