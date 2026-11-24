import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_milvus import BM25BuiltInFunction, Milvus
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from pymilvus import MilvusClient

from rag_eval_suite.utils.get_root import get_project_root
from rag_eval_suite.utils.loaders import load_yaml

logger = logging.getLogger(__name__)


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

    def run(self, *, hybrid: bool = False) -> Milvus:
        """Create a Milvus vector store connection.

        Hybrid mode adds Milvus' server-side BM25 function alongside the dense
        embedding field.  The collection must have been created and populated
        with the same dense+sparse schema for this mode to work.
        """
        load_dotenv(dotenv_path=get_project_root() / ".env")
        base_params = self.list_parameters

        endpoint = os.getenv("ZILLIZ_ENDPOINT")
        api_key = os.getenv("ZILLIZ_APIKEY")
        if not endpoint:
            raise RuntimeError("ZILLIZ_ENDPOINT is not configured in .env")
        if not api_key:
            raise RuntimeError("ZILLIZ_APIKEY is not configured in .env")

        store_params = {
            "embedding_function": self.embedding_function,
            "collection_name": self.collection_name,
            "connection_args": {"uri": endpoint, "token": api_key},
            "consistency_level": base_params.get("consistency_level"),
            "auto_id": base_params.get("auto_id"),
        }

        dense_index_params = {
            "index_type": base_params.get("index_type"),
            "metric_type": base_params.get("metric_type"),
        }
        if hybrid:
            existing_vector_fields = self._existing_vector_fields(endpoint, api_key)
            if existing_vector_fields and not {"dense", "sparse"}.issubset(
                existing_vector_fields
            ):
                # A collection created before hybrid retrieval normally uses
                # LangChain's default `vector` field. Passing the new hybrid
                # fields to that collection makes langchain-milvus attempt to
                # create an index on a field that does not exist.
                legacy_vector_field = next(
                    (
                        field
                        for field in existing_vector_fields
                        if field not in {"sparse", "dense"}
                    ),
                    existing_vector_fields[0],
                )
                logger.warning(
                    "Collection %s is dense-only; using its existing vector field "
                    "%s until it is re-ingested with hybrid fields.",
                    self.collection_name,
                    legacy_vector_field,
                )
                return Milvus(
                    embedding_function=self.embedding_function,
                    collection_name=self.collection_name,
                    connection_args={"uri": endpoint, "token": api_key},
                    vector_field=legacy_vector_field,
                    consistency_level=base_params.get("consistency_level"),
                    auto_id=base_params.get("auto_id"),
                )

            store_params.update(
                {
                    "builtin_function": BM25BuiltInFunction(),
                    "vector_field": ["dense", "sparse"],
                    "index_params": [
                        dense_index_params,
                        {"index_type": "AUTOINDEX", "metric_type": "BM25"},
                    ],
                }
            )
        else:
            store_params["index_params"] = dense_index_params

        return Milvus(**store_params)

    def _existing_vector_fields(self, endpoint: str, api_key: str) -> list[str]:
        """Return vector fields for an existing collection, if it is reachable."""
        client = MilvusClient(uri=endpoint, token=api_key)
        if not client.has_collection(self.collection_name):
            return []

        fields = client.describe_collection(self.collection_name).get("fields", [])
        vector_types = {101, 102, 103, 104, 105, 106, 107}
        return [
            field["name"]
            for field in fields
            if field.get("type") in vector_types
            or "VECTOR" in str(field.get("data_type", "")).upper()
        ]
