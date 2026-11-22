"""法律分词器：NFKC + casefold 归一、jieba 锁版本 + 法律词典、条号碎片与样板词过滤。"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import jieba

jieba.setLogLevel(60)

_MAX_QUERY_TERMS = 80

QUERY_STOPWORDS = frozenset({
    "规定", "应当", "本条例", "本法", "本规定", "本办法", "本解释", "本细则", "本",
    "根据", "依据", "依照", "按照", "什么", "怎么", "怎么办", "如何", "是否", "哪些", "哪个",
    "请问", "内容", "情况", "关于", "有关", "可以", "以及", "或者", "相关", "进行",
    "以下", "以上", "以内", "之后", "之前", "其他", "这个", "那个", "一个", "有", "后",
    "条", "款", "项", "目", "第", "之", "的", "了", "吗", "呢", "能", "不能", "能不能",
    "算不算", "有没有", "怎么判", "怎么处理", "合理", "合法",
})

_ARTICLE_TOKEN = re.compile(r"第?[零〇一二三四五六七八九十百千万0-9]*[条款项目号]?")

_STOPWORDS = set(
    "的了中在是与和及或等对于由从被把为之其这那并但而即则就都很再又才只仅还也"
)

_WORDISH = re.compile(r"[\w\u4e00-\u9fff]", re.UNICODE)


def _is_article_token(t: str) -> bool:
    return bool(t) and _ARTICLE_TOKEN.fullmatch(t) is not None


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").casefold()


def utf16_len(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


class Tokenizer:
    """进程内共享 jieba 状态，避免重复 load_userdict 导致词频漂移。"""

    _loaded_path: str | None = None

    def __init__(self, dict_path: Path | str | None = None):
        self.dict_path = Path(dict_path) if dict_path else None

    def _ensure_init(self) -> None:
        path_str = str(self.dict_path) if (self.dict_path and self.dict_path.is_file()) else ""
        if Tokenizer._loaded_path is not None:
            if path_str and path_str != Tokenizer._loaded_path:
                jieba.load_userdict(path_str)
                Tokenizer._loaded_path = path_str
            return
        if path_str:
            jieba.load_userdict(path_str)
        jieba.initialize()
        Tokenizer._loaded_path = path_str

    def cut(self, text: str) -> list[str]:
        self._ensure_init()
        norm = normalize(text)
        return [
            t for t in jieba.cut_for_search(norm)
            if t.strip()
            and _WORDISH.search(t)
            and not (utf16_len(t) == 1 and t in _STOPWORDS)
        ]

    def index_tokens(self, text: str) -> list[str]:
        return [t for t in self.cut(text) if not _is_article_token(t)]

    def query_terms(self, text: str, limit: int | None = _MAX_QUERY_TERMS) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for t in self.cut(text):
            if t in seen or t in QUERY_STOPWORDS or _is_article_token(t):
                continue
            seen.add(t)
            out.append(t)
            if limit is not None and len(out) >= limit:
                break
        return out

    def fts_body(self, text: str) -> str:
        return " ".join(self.index_tokens(text))


def match_or(terms: list[str]) -> str:
    return " OR ".join('"%s"' % t.replace('"', '""') for t in terms if t)


def match_and(terms: list[str]) -> str:
    return " ".join('"%s"' % t.replace('"', '""') for t in terms if t)
