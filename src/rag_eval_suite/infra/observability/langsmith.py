"""LangSmith tracing configuration shared by the API and RAG graph."""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv

from rag_eval_suite.utils.get_root import get_project_root

logger = logging.getLogger(__name__)


def configure_langsmith() -> bool:
    """Load LangSmith settings and report whether tracing is configured.

    LangChain and LangSmith read the ``LANGCHAIN_*`` and ``LANGSMITH_*``
    environment variables directly. This function intentionally does not copy
    or print credentials; it only makes sure the project `.env` is loaded
    before any LangChain runnables are created.
    """
    load_dotenv(dotenv_path=get_project_root() / ".env")

    tracing_enabled = _is_true(
        os.getenv("LANGSMITH_TRACING_V2")
        or os.getenv("LANGSMITH_TRACING")
        or os.getenv("LANGCHAIN_TRACING_V2")
    )
    api_key_configured = bool(
        os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    )
    project = (
        os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or "default"
    )

    if tracing_enabled and api_key_configured:
        logger.info("LangSmith tracing enabled for project '%s'", project)
    elif tracing_enabled:
        logger.warning("LangSmith tracing is enabled but no API key is configured")

    return tracing_enabled and api_key_configured


def langsmith_status() -> dict[str, Any]:
    """Return safe, non-secret tracing status for health and diagnostics."""
    enabled = configure_langsmith()
    project = (
        os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or "default"
    )
    return {
        "enabled": enabled,
        "project": project,
        "endpoint": os.getenv(
            "LANGSMITH_ENDPOINT",
            os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        ),
    }


def get_trace_config(retrieval_k: int) -> dict[str, Any]:
    """Build consistent trace metadata for one RAG request."""
    configure_langsmith()
    return {
        "run_name": "rag-question-answering",
        "tags": ["rag", "question-answering", "milvus-retrieval"],
        "metadata": {
            "retrieval_k": retrieval_k,
            "retriever": "milvus-hybrid-or-legacy-dense",
            "llm_provider": "groq",
        },
    }


def _is_true(value: str | None) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"} if value else False
