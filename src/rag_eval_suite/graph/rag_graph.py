"""End-to-end retrieval-augmented generation graph."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph

from rag_eval_suite.application.retrievers import get_hybrid_retriever
from rag_eval_suite.config.paths import PARAM_FILE_PATH, PROJECT_ROOT
from rag_eval_suite.infra.embeddings import get_openai_embeddings
from rag_eval_suite.infra.llm import get_groq_llm
from rag_eval_suite.infra.vectorstores import MilvusStore


class RAGState(TypedDict, total=False):
    """State passed through the retrieval and answer-generation nodes."""

    question: str
    documents: list[Document]
    context: str
    answer: str


FAST_RESPONSES = {
    "hi": "Hi! How can I help you explore the knowledge base?",
    "hello": "Hello! What would you like to learn from the knowledge base?",
    "hey": "Hey! Ask me anything about the indexed documents.",
    "thanks": "You’re welcome! Ask another question whenever you’re ready.",
    "thank you": "You’re welcome! Ask another question whenever you’re ready.",
    "bye": "Goodbye! Come back whenever you have another question.",
}


def get_fast_response(question: str) -> str | None:
    """Return a local response for conversational messages without using RAG."""
    normalized = re.sub(r"[^a-z0-9 ]", "", question.lower()).strip()
    return FAST_RESPONSES.get(normalized)


def _read_prompt() -> str:
    return (PROJECT_ROOT / "prompt" / "rag_prompt.md").read_text(encoding="utf-8")


def _format_context(documents: list[Document]) -> str:
    if not documents:
        return "No relevant context was retrieved."

    sections = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        source = metadata.get("source", "unknown source")
        page = metadata.get("page")
        location = f"{source}, page {page + 1}" if isinstance(page, int) else str(source)
        sections.append(f"[Context {index} | {location}]\n{document.page_content}")
    return "\n\n".join(sections)


class RAGGraph:
    """Compiled graph that retrieves context and generates a grounded answer."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm,
        retrieval_k: int = 5,
    ) -> None:
        if retrieval_k <= 0:
            raise ValueError("retrieval_k must be greater than zero")
        self.retriever = retriever
        self.llm = llm
        self.retrieval_k = retrieval_k
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", _read_prompt()), ("human", "{question}")]
        )
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        graph = StateGraph(RAGState)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("generate", self.generate)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    def retrieve(self, state: RAGState) -> RAGState:
        documents = self.retriever.invoke(state["question"])
        return {"documents": documents, "context": _format_context(documents)}

    def generate(self, state: RAGState) -> RAGState:
        response = (self.prompt | self.llm).invoke(
            {"context": state.get("context", ""), "question": state["question"]}
        )
        answer = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        return {"answer": answer}

    def invoke(self, question: str) -> RAGState:
        """Run retrieval and generation for one user question."""
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        if fast_response := get_fast_response(question):
            return {
                "question": question,
                "documents": [],
                "context": "",
                "answer": fast_response,
            }
        return self.workflow.invoke({"question": question})

    def stream_answer(self, question: str) -> Iterator[dict[str, object]]:
        """Yield retrieved sources followed by answer tokens."""
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        if fast_response := get_fast_response(question):
            yield {"type": "sources", "sources": []}
            yield {"type": "token", "content": fast_response}
            return

        retrieved = self.retrieve({"question": question})
        documents = retrieved.get("documents", [])
        yield {"type": "sources", "sources": _source_payload(documents)}

        chain = self.prompt | self.llm
        for chunk in chain.stream(
            {"context": retrieved["context"], "question": question}
        ):
            content = getattr(chunk, "content", "")
            if isinstance(content, str) and content:
                yield {"type": "token", "content": content}


def get_rag_graph(
    collection_name: str | None = None,
    retrieval_k: int = 5,
) -> RAGGraph:
    """Build the RAG graph connected to the existing Zilliz collection."""
    store = MilvusStore(
        embedding_function=get_openai_embeddings(),
        collection_name=collection_name
        or os.getenv("MILVUS_COLLECTION_NAME", "EvaluationRagCollection"),
        param_file_path=PARAM_FILE_PATH,
        param_file_key="milvus_db",
    ).run(hybrid=True)
    return RAGGraph(
        retriever=get_hybrid_retriever(store, k=retrieval_k),
        llm=get_groq_llm(),
        retrieval_k=retrieval_k,
    )


def _source_payload(documents: list[Document]) -> list[dict[str, object]]:
    """Convert retrieved documents into a JSON-friendly source payload."""
    sources = []
    for document in documents:
        metadata = document.metadata or {}
        page = metadata.get("page")
        sources.append(
            {
                "source": str(metadata.get("source", "unknown source")),
                "page": page + 1 if isinstance(page, int) else None,
                "excerpt": document.page_content[:240].strip(),
            }
        )
    return sources
