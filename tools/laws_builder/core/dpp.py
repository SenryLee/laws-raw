"""贪心 DPP 多样性重排：兼顾质量分（JEV score）与内容去冗余。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .tokenizer import normalize

_JIEBA_LAZY = None


def _tokens(text: str) -> dict[str, int]:
    global _JIEBA_LAZY
    if _JIEBA_LAZY is None:
        import jieba

        jieba.setLogLevel(60)
        _JIEBA_LAZY = jieba
    out: dict[str, int] = {}
    for t in _JIEBA_LAZY.cut_for_search(normalize(text)):
        t = t.strip()
        if t and re.search(r"[\u4e00-\u9fff\w]", t):
            out[t] = out.get(t, 0) + 1
    return out


@dataclass
class DPPItem:
    key: int
    score: float
    content: str
    payload: dict = field(default_factory=dict)


def _tfidf_vectors(items: list[DPPItem]) -> list[dict[str, float]]:
    tfs = [_tokens(it.content) for it in items]
    n_docs = len(items)
    df: dict[str, int] = {}
    for tf in tfs:
        for w in tf:
            df[w] = df.get(w, 0) + 1
    vecs: list[dict[str, float]] = []
    for tf in tfs:
        v: dict[str, float] = {}
        for w, c in tf.items():
            idf = math.log((1 + n_docs) / (1 + df[w])) + 1
            v[w] = c * idf
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({w: x / norm for w, x in v.items()})
    return vecs


def _cos(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


def diversify(items: list[DPPItem], take: int = 10) -> tuple[list[DPPItem], list[DPPItem]]:
    if len(items) <= take:
        return list(items), []
    n = len(items)
    vecs = _tfidf_vectors(items)
    scores = [max(it.score, 1e-6) for it in items]

    K = [[scores[i] * scores[j] * _cos(vecs[i], vecs[j]) for j in range(n)] for i in range(n)]
    diag = [K[i][i] for i in range(n)]
    cis: list[list[float]] = [[0.0] * n for _ in range(n)]

    selected: list[int] = []
    rest = list(range(n))
    chosen: list[int] = []
    for _ in range(take):
        best_j = max(rest, key=lambda j: (diag[j], -j))
        e_score = diag[best_j]
        ci2 = sum(cis[t][best_j] * cis[t][best_j] for t in range(len(selected)))
        e_new = math.sqrt(max(e_score - ci2, 1e-12))
        chosen.append(best_j)
        selected.append(best_j)
        rest.remove(best_j)

        for j in rest:
            ei = K[best_j][j]
            ci = ei
            for t in range(len(selected) - 1):
                ci -= cis[t][best_j] * cis[t][j]
            cis[len(selected) - 1][j] = ci / e_new
            cis[len(selected) - 1][best_j] = e_new
            diag[j] = diag[j] - ci * ci / e_new

    picked = [items[i] for i in chosen]
    remainder = [it for idx, it in enumerate(items) if idx not in set(chosen)]
    return picked, remainder
