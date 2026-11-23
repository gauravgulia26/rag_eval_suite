import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from rag_eval_suite.utils.get_root import get_project_root


def get_openai_embeddings(
    model_name: str = "text-embedding-3-small",
    api_key: str | None = None,
):
    load_dotenv(dotenv_path=get_project_root() / ".env")
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured in .env")
    return OpenAIEmbeddings(model=model_name, api_key=api_key)
