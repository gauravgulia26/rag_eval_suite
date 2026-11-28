"""TypeSafe Jev document reranking for retrieved LangChain documents."""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_typesafe import Noul, TypeSafeClassifier

JEV_RELEVANCE_SCORE_KEY = "jev_relevance_score"
logger = logging.getLogger(__name__)


class JevReranker:
    """Rank retrieved documents by Jev's probability of answer relevance."""

    def __init__(
        self,
        *,
        classifier: TypeSafeClassifier | None = None,
        api_key: str | None = None,
        model: str = "jev-latest",
        max_concurrency: int = 8,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than zero")

        if classifier is not None:
            self.classifier = classifier
        else:
            classifier_kwargs = {"model": model}
            if api_key is not None:
                classifier_kwargs["api_key"] = api_key
            self.classifier = TypeSafeClassifier(**classifier_kwargs)
        self.max_concurrency = max_concurrency

    async def _score_document(
        self,
        query: str,
        document: Document,
        semaphore: asyncio.Semaphore,
    ) -> tuple[Document, float | None]:
        state = {"query": query, "passage": document.page_content}
        question = Noul(
            instructions=(
                "Could this passage directly help answer the query using evidence "
                "contained in the passage? Return true only when the passage is "
                "useful evidence for answering the query."
            )
        )
        async with semaphore:
            try:
                response = await self.classifier.ainvoke(
                    {"state": state, "questions": {"relevant": question}}
                )
                score = float(response.nouls["relevant"].noul)
            except Exception as exc:  # noqa: BLE001
                # A failed reranking call must not remove a document from the
                # retrieval result. It is sorted after documents Jev scored.
                logger.warning("Jev reranking failed for one document: %s", exc)
                score = None
        return document, score

    async def arerank(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        top_k: int | None = None,
    ) -> list[Document]:
        """Asynchronously rerank documents for ``query``.

        Documents that cannot be scored are retained after scored documents.
        Ties preserve the original retrieval order.
        """
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative or None")
        if not documents or top_k == 0:
            return []

        semaphore = asyncio.Semaphore(self.max_concurrency)
        scored = await asyncio.gather(
            *(
                self._score_document(query, document, semaphore)
                for document in documents
            )
        )

        ranked = []
        for index, (document, score) in enumerate(scored):
            copied = copy.deepcopy(document)
            if score is not None:
                copied.metadata = {
                    **(copied.metadata or {}),
                    JEV_RELEVANCE_SCORE_KEY: score,
                }
            ranked.append((score is not None, score or 0.0, -index, copied))

        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [item[3] for item in ranked[:top_k]]

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        top_k: int | None = None,
    ) -> list[Document]:
        """Synchronously rerank documents when no event loop is running."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arerank(query, documents, top_k=top_k))
        raise RuntimeError("Use `arerank` when an asyncio event loop is running")


__all__ = ["JEV_RELEVANCE_SCORE_KEY", "JevReranker"]
