from typing import TypedDict


class IngestionResult(TypedDict):
    total_documents: int
    total_chunks: int
    inserted_ids: list[str]
