import asyncio
import os
from collections.abc import Sequence

from deepeval.evaluate.types import EvaluationResult
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models.system_one import TypeSafeModel
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

from rag_eval_suite.application.retrievers import get_hybrid_retriever
from rag_eval_suite.config.paths import MILVUS_COLLECTION_NAME, PARAM_FILE_PATH
from rag_eval_suite.eval.components import EvaluateRetriever
from rag_eval_suite.infra.embeddings.hosted_embeddings import get_openai_embeddings
from rag_eval_suite.infra.reranker import JevReranker
from rag_eval_suite.infra.vectorstores import MilvusStore
from rag_eval_suite.utils.get_root import get_project_root
from rag_eval_suite.utils.loaders import format_retrieved_context, load_json


def __get_vector_db():
    return MilvusStore(
        embedding_function=get_openai_embeddings(),
        collection_name=MILVUS_COLLECTION_NAME,
        param_file_key="milvus_db",
        param_file_path=PARAM_FILE_PATH,
    ).run(hybrid=True, inspect_existing_schema=False)


async def _make_test_case(
    index: int,
    record: dict[str, str],
    retriever,
    reranker: JevReranker,
    semaphore: asyncio.Semaphore,
) -> tuple[int, LLMTestCase]:
    async with semaphore:
        documents = await retriever.ainvoke(record["query"])
        documents = await reranker.arerank(record["query"], documents)
    return index, LLMTestCase(
        input=record["query"],
        expected_output=record["ideal_answer"],
        retrieval_context=[format_retrieved_context(documents)],
        actual_output="Not Evaluated",
    )


async def _make_test_cases(
    golden_data: Sequence[dict[str, str]],
    retriever,
    reranker: JevReranker,
    max_concurrency: int = 8,
) -> list[LLMTestCase]:
    """Retrieve context for all golden records concurrently with progress."""
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be greater than zero")

    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        asyncio.create_task(
            _make_test_case(index, record, retriever, reranker, semaphore)
        )
        for index, record in enumerate(golden_data)
    ]
    completed = []
    for task in tqdm_asyncio.as_completed(
        tasks,
        total=len(tasks),
        desc="Retrieving and reranking contexts",
    ):
        completed.append(await task)
    completed.sort(key=lambda item: item[0])
    return [test_case for _, test_case in completed]


async def _run_evaluate_retriever() -> EvaluationResult:
    load_dotenv(dotenv_path=get_project_root() / ".env")
    vector_db = __get_vector_db()
    retriever = get_hybrid_retriever(vector_store=vector_db, k=3)
    reranker = JevReranker(api_key=os.getenv("TYPESAFE_API_KEY"))
    golden_data = load_json(
        "/home/gaurav/Documents/rag_eval_suite/data/golden/lecture01_golden.json"
    )
    jev = TypeSafeModel(api_key=os.getenv("TYPESAFE_API_KEY"))
    precision = ContextualPrecisionMetric(
        threshold=0.7,
        system_one_model=jev,
        eval_mode="system_one",
        include_reason=True,
    )
    recall = ContextualRecallMetric(
        threshold=0.7,
        system_one_model=jev,
        eval_mode="system_one",
        include_reason=True,
    )
    all_test_cases = await _make_test_cases(golden_data, retriever, reranker)
    evaluator = EvaluateRetriever(retriever=None, metrics=[precision, recall])
    return await asyncio.to_thread(
        evaluator.evaluate,
        test_cases=all_test_cases,
    )


def RunEvaluateRetriever() -> EvaluationResult:
    """Run retrieval and evaluation for the complete golden dataset."""
    return asyncio.run(_run_evaluate_retriever())


if __name__ == "__main__":
    RunEvaluateRetriever()
