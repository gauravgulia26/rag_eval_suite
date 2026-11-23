from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel

from ..ingestion.chunking import Chunker
from ..ingestion.loading import Loader


class IngestionPipeline(BaseModel):
    loader: Loader
    chunker: Chunker
    vector_store: VectorStore

    @property
    def list_documents(self) -> list[Document]:
        return self.loader.run()

    @property
    def list_chunks(self) -> list[Document]:
        return self.chunker.run()

    def run(self):
        pass
