from .hosted_embeddings import get_openai_embeddings

OPENAI_EMBEDDING = get_openai_embeddings()
__all__ = ["OPENAI_EMBEDDING"]
