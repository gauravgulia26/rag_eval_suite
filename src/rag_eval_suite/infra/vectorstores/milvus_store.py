import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from rag_eval_suite.utils.get_root import get_project_root
from rag_eval_suite.utils.loaders import load_yaml


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
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
        load_dotenv(dotenv_path=get_project_root() / ".env")
        base_params = self.list_parameters

        endpoint = os.getenv("ZILLIZ_ENDPOINT")
        api_key = os.getenv("ZILLIZ_APIKEY")
        if not endpoint:
            raise RuntimeError("ZILLIZ_ENDPOINT is not configured in .env")
        if not api_key:
            raise RuntimeError("ZILLIZ_APIKEY is not configured in .env")

        return Milvus(
            embedding_function=self.embedding_function,
            collection_name=self.collection_name,
            connection_args={"uri": endpoint, "token": api_key},
            index_params={
                "index_type": base_params.get("index_type"),
                "metric_type": base_params.get("metric_type"),
            },
            consistency_level=base_params.get("consistency_level"),
            auto_id=base_params.get("auto_id"),
        )
