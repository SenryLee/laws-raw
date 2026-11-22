"""法锤官网 · 多轮法律法规问答引擎（基于 DeepSeek + JEV 召回，严格附带 [n] 原文引用）。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .config import LawsConfig
from .deepseek_l2_l3 import _DeepSeekClient, deep_search
from .search_l1 import LawHit
from .tokenizer import Tokenizer

ASK_SYSTEM_PROMPT = """你是法锤 AI 的法律法规问答助手。请严格依据提供的【检索到的法条依据】回答用户的法律问题，遵守以下铁律：
1. 只能基于给定法条作答，严禁编造不存在的法条号或法律规定。
2. 每个法律结论、期限、条件或责任后必须紧跟方括号引用编号，如 [1]、[2]（编号与给定依据序号一致）。
3. 结合多轮对话上下文理解用户的追问，回答结构清晰：先给直接结论，再分点说明适用条件与法律后果，最后提示实务注意点。
4. 如果给定依据不足以覆盖问题的某一方面，直接坦诚说明“当前检索到的条文未涵盖该细节”，不要猜测。
5. 语气专业、克制、直接，使用简体中文。"""


@dataclass
class AskTurn:
    role: str      # "user" | "assistant"
    content: str


@dataclass
class AskResult:
    question: str
    mode: str                 # "deepseek_ask" | "local_fallback"
    answer: str
    sources: list[LawHit] = field(default_factory=list)
    rewrite: dict = field(default_factory=dict)
    elapsed_ms: int = 0
    tokens_used: int = 0
    notice: str = ""


def _build_search_query(question: str, history: list[AskTurn]) -> str:
    """将多轮追问（如“那如果超过三年了呢？”）与上一轮用户核心问题拼接，避免脱敏/指代丢失。"""
    q = (question or "").strip()
    if not history or len(q) >= 22:
        return q
    prev_users = [t.content.strip() for t in history if t.role == "user" and t.content.strip()]
    if not prev_users:
        return q
    last_u = prev_users[-1][:80]
    return f"{last_u}；追问：{q}"


def _local_fallback_answer(question: str, sources: list[LawHit]) -> str:
    if not sources:
        return (
            "当前索引库中暂未检索到与该问题直接匹配的现行有效法条。"
            "建议尝试补充具体法律关系关键词（如“民间借贷”“诉讼时效”“违约金”“经济补偿”），"
            "或在左侧勾选扩展库后重试。"
        )
    lines = ["基于当前检索到的核心法条，为你整理如下直接法律依据：\n"]
    for idx, h in enumerate(sources[:5], 1):
        clause = f"{h.clause_path}：" if h.clause_path else ""
        snippet = h.content.replace("\n", " ").strip()
        if len(snippet) > 140:
            snippet = snippet[:140] + "…"
        lines.append(f"- **《{h.title}》{clause}** {snippet} [{idx}]")
    lines.append("\n提示：配置 DeepSeek 密钥后，问答助手将自动结合全部检索依据生成完整的法律分析与多轮追问解答。")
    return "\n".join(lines)


def run_ask(
    cfg: LawsConfig,
    tok: Tokenizer,
    question: str,
    *,
    history: list[AskTurn] | None = None,
    status: str = "effective",
    category: str | None = None,
    include_ext: bool = False,
) -> AskResult:
    t0 = time.time()
    hist = (history or [])[-6:]  # 最多保留最近 3 轮（6 条消息）防上下文过长
    retrieval_q = _build_search_query(question, hist)

    deep_res = deep_search(
        cfg,
        tok,
        retrieval_q,
        status=status,
        category=category,
        include_ext=include_ext,
        top_k=6,
    )
    sources = deep_res.results[:6]

    if not cfg.llm_enabled or not sources:
        return AskResult(
            question=question,
            mode="local_fallback",
            answer=_local_fallback_answer(question, sources),
            sources=sources,
            rewrite=deep_res.rewrite,
            elapsed_ms=int((time.time() - t0) * 1000),
            tokens_used=deep_res.tokens_used,
            notice=deep_res.notice or ("尚未配置 DeepSeek 密钥，当前展示法条依据摘要。" if not cfg.llm_enabled else ""),
        )

    # 组织法条依据块
    ev_lines = []
    for i, h in enumerate(sources, 1):
        clause = h.clause_path or h.heading
        ev_lines.append(
            f"[{i}] 《{h.title}》（{h.status_label}，{clause}）\n{h.content[:800]}"
        )
    evidence_block = "\n\n".join(ev_lines)

    messages = [{"role": "system", "content": ASK_SYSTEM_PROMPT}]
    for t in hist:
        if t.role in ("user", "assistant") and t.content.strip():
            messages.append({"role": t.role, "content": t.content.strip()[:600]})
    messages.append({
        "role": "user",
        "content": f"【用户当前问题】\n{question}\n\n【检索到的法条依据】\n{evidence_block}",
    })

    try:
        import httpx

        url = f"{cfg.api_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {cfg.api_secret}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=cfg.timeout_seconds) as client:
            resp = client.post(
                url,
                json={"model": cfg.jev_model, "temperature": 0.1, "messages": messages},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        answer = (data["choices"][0]["message"]["content"] or "").strip()
        used = deep_res.tokens_used + int(data.get("usage", {}).get("total_tokens", 0))
        return AskResult(
            question=question,
            mode="deepseek_ask",
            answer=answer,
            sources=sources,
            rewrite=deep_res.rewrite,
            elapsed_ms=int((time.time() - t0) * 1000),
            tokens_used=used,
            notice=deep_res.notice,
        )
    except Exception as exc:  # noqa: BLE001
        return AskResult(
            question=question,
            mode="local_fallback",
            answer=_local_fallback_answer(question, sources),
            sources=sources,
            rewrite=deep_res.rewrite,
            elapsed_ms=int((time.time() - t0) * 1000),
            tokens_used=deep_res.tokens_used,
            notice=f"DeepSeek 问答暂时不可用，已切换为法条依据摘要（{type(exc).__name__}）",
        )
