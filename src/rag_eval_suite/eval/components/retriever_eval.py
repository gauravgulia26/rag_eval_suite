from collections.abc import Sequence
from dataclasses import dataclass

from deepeval.evaluate import evaluate
from deepeval.evaluate.types import EvaluationResult
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from langchain_core.retrievers import BaseRetriever


@dataclass
class EvaluateRetriever:
    retriever: BaseRetriever
    metrics: list[BaseMetric]

    def __post_init__(self):
        if not isinstance(self.metrics, list):
            raise TypeError(
                f"Type {type(self.metrics)} is invalid for Type:[list[BaseMetric]]"
            )

    def evaluate(self, test_cases: Sequence[LLMTestCase]) -> EvaluationResult:
        """Evaluate one or more prepared test cases as a single run."""
        if not test_cases:
            raise ValueError("test_cases must contain at least one testcase")
        return evaluate(test_cases=list(test_cases), metrics=self.metrics)
