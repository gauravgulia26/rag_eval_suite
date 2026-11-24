"""HTTP API for the RAG question-answering graph."""

import json
from collections.abc import AsyncIterator
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from rag_eval_suite.graph import RAGGraph, get_fast_response, get_rag_graph

app = FastAPI(
    title="RAG Eval Suite API",
    description="Ask grounded questions over the documents indexed in Zilliz.",
    version="0.1.0",
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, description="Question to answer from the corpus")
    retrieval_k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    source: str
    page: int | None = None
    excerpt: str


class QuestionResponse(BaseModel):
    answer: str
    sources: list[Source]


@lru_cache(maxsize=8)
def _get_graph(retrieval_k: int) -> RAGGraph:
    return get_rag_graph(retrieval_k=retrieval_k)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest) -> QuestionResponse:
    if fast_response := get_fast_response(request.question):
        return QuestionResponse(answer=fast_response, sources=[])
    try:
        state = _get_graph(request.retrieval_k).invoke(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="RAG service is unavailable") from exc

    sources = []
    for document in state.get("documents", []):
        metadata = document.metadata or {}
        sources.append(
            Source(
                source=str(metadata.get("source", "unknown source")),
                page=metadata.get("page") + 1
                if isinstance(metadata.get("page"), int)
                else None,
                excerpt=document.page_content[:240].strip(),
            )
        )
    return QuestionResponse(answer=state.get("answer", ""), sources=sources)


@app.post("/ask/stream")
def ask_stream(request: QuestionRequest) -> StreamingResponse:
    """Stream source metadata and answer tokens as server-sent events."""

    async def events() -> AsyncIterator[str]:
        try:
            if fast_response := get_fast_response(request.question):
                yield 'data: {"type": "sources", "sources": []}\n\n'
                yield f"data: {json.dumps({'type': 'token', 'content': fast_response})}\n\n"
                yield "data: [DONE]\n\n"
                return
            graph = _get_graph(request.retrieval_k)
            for event in graph.stream_answer(request.question):
                yield f"data: {json.dumps(event)}\n\n"
            yield "data: [DONE]\n\n"
        except ValueError as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
        except Exception:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'detail': 'RAG service is unavailable'})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
