"""中文数字 → 阿拉伯数字（万以内，含 零/〇 组合）。"""

from __future__ import annotations

_DIGITS = {
    "零": 0, "〇": 0, "O": 0,
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_UNITS = {"十": 10, "百": 100, "千": 1000}


def _parse_plain(s: str) -> int | None:
    total = 0
    cur = 0
    saw_digit = False
    for ch in s:
        if ch in _DIGITS:
            cur = _DIGITS[ch]
            saw_digit = True
        elif ch in _UNITS:
            unit = _UNITS[ch]
            if cur == 0:
                cur = 1
            total += cur * unit
            cur = 0
        elif ch in ("零", "〇"):
            continue
        else:
            return None
    if not saw_digit and total == 0:
        return None
    return total + cur


def cn2num(s: str) -> int | None:
    s = (s or "").strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if "万" in s:
        left, _, right = s.rpartition("万")
        base = _parse_plain(left)
        if base is None:
            return None
        rest = _parse_plain(right) if right else 0
        if rest is None:
            return None
        return base * 10000 + rest
    if all(ch in _DIGITS for ch in s):
        value = 0
        for ch in s:
            value = value * 10 + _DIGITS[ch]
        return value
    return _parse_plain(s)
