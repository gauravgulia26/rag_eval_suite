from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Chunker:
    splitter: TextSplitter
    corpus: list[Document]

    def run(self) -> list[Document]:
        return self.splitter.split_documents(self.corpus)
