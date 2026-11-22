"""法锤官网 · 法律法规检索配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

LAWS_ROOT = Path(__file__).resolve().parent.parent


def _bool_opt(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int_opt(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _resolve_path(raw: str, base: Path) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


@dataclass
class LawsConfig:
    root: Path
    core_db: Path
    ext_db: Path
    cache_db: Path
    legal_dict_path: Path
    aliases_path: Path
    api_secret: str
    api_base_url: str
    rewrite_model: str
    jev_model: str
    timeout_seconds: int
    rate_limit_enabled: bool
    deep_max_per_min: int
    daily_llm_budget: int
    max_query_chars: int
    cors_origins: list[str]

    @property
    def llm_enabled(self) -> bool:
        return bool(self.api_secret.strip())


def load_config(root: Path | str | None = None) -> LawsConfig:
    base = Path(root).resolve() if root else LAWS_ROOT
    cors_raw = os.getenv("LAWS_CORS_ORIGINS", "*")
    cors_origins = [x.strip() for x in cors_raw.split(",") if x.strip()]

    return LawsConfig(
        root=base,
        core_db=_resolve_path(os.getenv("LAWS_CORE_DB", "data/laws_core.db"), base),
        ext_db=_resolve_path(os.getenv("LAWS_EXT_DB", "data/laws_ext.db"), base),
        cache_db=_resolve_path(os.getenv("LAWS_CACHE_DB", "data/cache.db"), base),
        legal_dict_path=base / "core" / "resources" / "legal_dict.txt",
        aliases_path=base / "core" / "resources" / "law_aliases.csv",
        api_secret=os.getenv("LAWS_LLM_AUTH", "").strip(),
        api_base_url=os.getenv("LAWS_API_BASE_URL", "https://api.deepseek.com/v1").rstrip("/"),
        rewrite_model=os.getenv("LAWS_REWRITE_MODEL", "deepseek-chat").strip(),
        jev_model=os.getenv("LAWS_JEV_MODEL", "deepseek-chat").strip(),
        timeout_seconds=_int_opt("LAWS_TIMEOUT_SECONDS", 45),
        rate_limit_enabled=_bool_opt("LAWS_RATE_LIMIT_ENABLED", False),
        deep_max_per_min=_int_opt("LAWS_DEEP_MAX_PER_MIN", 12),
        daily_llm_budget=_int_opt("LAWS_DAILY_LLM_BUDGET", 0),
        max_query_chars=_int_opt("LAWS_MAX_QUERY_CHARS", 200),
        cors_origins=cors_origins,
    )
