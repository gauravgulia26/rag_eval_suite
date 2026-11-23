import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from ...contracts import IngestionResult
from ...infra.chunkers import Chunker
from ...infra.loaders.pymupdf_loader import Loader


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class IngestionPipeline:
    loader: Loader
    chunker: Chunker
    vector_store: VectorStore
    docs: list[Document] | None = None
    chunks: list[Document] | None = None

    def run(self) -> IngestionResult:
        loaded_docs = self.loader.run()
        loaded_chunks = self.chunker.run(corpus=loaded_docs)
        self.docs = loaded_docs
        self.chunks = loaded_chunks

        idx = self.vector_store.add_documents(loaded_chunks)
        return IngestionResult(
            total_chunks=len(loaded_chunks),
            total_documents=len(loaded_docs),
            inserted_ids=idx,
        )

    async def run_async(
        self,
        document_paths: Sequence[Path | str],
        batch_size: int,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> IngestionResult:
        """Ingest multiple documents concurrently and write chunks in batches.

        ``progress_callback`` receives ``(stage, completed, total)`` updates for
        the ``loading``, ``chunking``, and ``ingesting`` stages.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        paths = list(document_paths)
        if not paths:
            raise ValueError("at least one document path is required")

        def report(stage: str, completed: int, total: int) -> None:
            if progress_callback is not None:
                progress_callback(stage, completed, total)

        loaded_count = 0

        async def load_document(path: Path | str) -> list[Document]:
            nonlocal loaded_count
            documents = await type(self.loader)(document_path=path).run_async()
            loaded_count += 1
            report("loading", loaded_count, len(paths))
            return documents

        loaded_documents = await asyncio.gather(*(load_document(path) for path in paths))
        documents = [document for result in loaded_documents for document in result]

        report("chunking", 0, 1)
        chunks = self.chunker.run(corpus=documents)
        report("chunking", 1, 1)

        batches = [
            chunks[start : start + batch_size]
            for start in range(0, len(chunks), batch_size)
        ]
        report("ingesting", 0, len(batches))
        inserted_batch_count = 0

        async def insert_batch(batch: list[Document]) -> list[str]:
            nonlocal inserted_batch_count
            inserted_ids = await self.vector_store.aadd_documents(batch)
            inserted_batch_count += 1
            report("ingesting", inserted_batch_count, len(batches))
            return inserted_ids

        inserted_id_batches = await asyncio.gather(*(insert_batch(batch) for batch in batches))
        inserted_ids = [item for batch_ids in inserted_id_batches for item in batch_ids]

        self.docs = documents
        self.chunks = chunks
        return IngestionResult(
            total_chunks=len(chunks),
            total_documents=len(paths),
            inserted_ids=inserted_ids,
        )
