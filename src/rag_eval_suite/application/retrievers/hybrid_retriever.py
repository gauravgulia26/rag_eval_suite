"""Retriever construction for Milvus dense + BM25 hybrid search."""

from langchain_core.retrievers import BaseRetriever
from langchain_milvus import Milvus


def get_hybrid_retriever(vector_store: Milvus, *, k: int = 5) -> BaseRetriever:
    """Adapt a hybrid-configured Milvus store to LangChain's retriever API.

    ``Milvus.similarity_search`` automatically executes dense and BM25 searches
    and fuses their results when the store was initialized with
    ``BM25BuiltInFunction``.  ``as_retriever`` keeps that behavior while making
    the graph independent of the concrete vector-store implementation.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero")

    search_kwargs = {"k": k}
    if len(vector_store.vector_fields) > 1:
        # RRF is robust when the server exposes more than one hybrid path. A
        # weighted reranker requires its weights list to exactly match the
        # number of ANN requests, which can differ for older collection schemas.
        search_kwargs.update(ranker_type="rrf", ranker_params={"k": 60})

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )


__all__ = ["get_hybrid_retriever"]
