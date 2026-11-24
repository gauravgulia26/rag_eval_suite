"""Groq-hosted language model factory."""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from rag_eval_suite.utils.get_root import get_project_root

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


def get_groq_llm(
    model_name: str | None = None,
    temperature: float = 0.1,
) -> ChatGroq:
    """Create the configured Groq chat model from the project environment."""
    load_dotenv(dotenv_path=get_project_root() / ".env")
    api_key = os.getenv("GROQAPI_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQAPI_KEY is not configured in .env")

    return ChatGroq(
        api_key=api_key,
        model=model_name or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        temperature=temperature,
    )
