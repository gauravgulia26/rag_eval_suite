import os
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
from pydantic.dataclasses import dataclass

from rag_eval_suite.utils.loaders import load_yaml


@dataclass
class MilvusStore:
    embedding_function: Embeddings
    collection_name: str
    param_file_path: Path | str
    param_file_key: str
    """Key used to Retrieve Parameter from YAML File"""

    @property
    def list_parameters(self) -> dict:
        return load_yaml(path=self.param_file_path)[self.param_file_key]

    def run(self) -> Milvus:
        base_params = self.list_parameters
        return Milvus(
            embedding_function=self.embedding_function,
            collection_name=self.collection_name,
            connection_args={"uri": os.getenv("ZILLIZ_ENDPOINT")},
            index_params={
                "index_type": base_params.get("index_type"),
                "metric_type": base_params.get("metric_type"),
            },
            consistency_level=base_params.get("consistency_level"),
            auto_id=base_params.get("auto_id"),
        )
