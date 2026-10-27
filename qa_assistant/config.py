"""Runtime settings, read from environment variables (and an optional .env file)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

VALID_MODES = ("auto", "gemini", "offline")
VALID_RETRIEVAL = ("hybrid", "keyword")


@dataclass(frozen=True)
class Settings:
    kb_dir: Path
    top_k: int
    gemini_model: str
    mode: str  # "auto" picks "gemini" when a Gemini key is set, otherwise "offline"
    retrieval: str = "hybrid"  # "hybrid" = BM25 + Gemini embeddings (Gemini mode only), "keyword" = BM25 only
    embedding_model: str = "gemini-embedding-001"
    min_similarity: float = 0.6  # cosine similarity below which a section is not a semantic match
    cache_dir: Path | None = None  # where section embeddings are cached; None disables the cache
    max_question_chars: int = 1000


def has_gemini_credentials() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def get_settings(**overrides) -> Settings:
    kb_dir = Path(os.getenv("QA_KB_DIR", PROJECT_ROOT / "knowledge_base"))
    if not kb_dir.is_absolute():
        kb_dir = PROJECT_ROOT / kb_dir

    values = {
        "kb_dir": kb_dir,
        # An alias that always points at Google's current Flash model; pin a name such as
        # "gemini-3-flash-preview" for reproducible results (list them with `python -m qa_assistant.gemini_models`).
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
        "top_k": int(os.getenv("QA_TOP_K", "5")),
        "mode": os.getenv("QA_MODE", "auto"),
        "retrieval": os.getenv("QA_RETRIEVAL", "hybrid"),
        "embedding_model": os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        "min_similarity": float(os.getenv("QA_MIN_SIMILARITY", "0.6")),
        "cache_dir": PROJECT_ROOT / ".cache",
    }
    values.update({k: v for k, v in overrides.items() if v is not None})

    if values["mode"] not in VALID_MODES:
        raise ValueError(f"QA_MODE must be one of {VALID_MODES}, got {values['mode']!r}")
    if values["retrieval"] not in VALID_RETRIEVAL:
        raise ValueError(f"QA_RETRIEVAL must be one of {VALID_RETRIEVAL}, got {values['retrieval']!r}")
    return Settings(**values)


def resolve_mode(mode: str) -> str:
    if mode == "auto":
        return "gemini" if has_gemini_credentials() else "offline"
    return mode
