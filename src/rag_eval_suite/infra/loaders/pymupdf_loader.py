from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from pydantic.dataclasses import dataclass


@dataclass
class Loader:
    document_path: Path | str

    def run(self) -> list[Document]:
        return PyMuPDFLoader(file_path=self.document_path).load()

    async def run_async(self) -> list[Document]:
        """Load the document using the loader's native async interface."""
        return await PyMuPDFLoader(file_path=self.document_path).aload()
