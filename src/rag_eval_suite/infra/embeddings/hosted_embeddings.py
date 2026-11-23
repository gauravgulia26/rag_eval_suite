import os

from langchain_openai import OpenAIEmbeddings


def get_openai_embeddings(
    model_name: str = "text-embedding-3-small",
    api_key: str = os.getenv("OPENAI_API_KEY"),
):
    return OpenAIEmbeddings(model=model_name, api_key=api_key)
