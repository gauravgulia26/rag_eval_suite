"""Filesystem paths derived from project-level constants."""

from rag_eval_suite.constants import DATA_DIR, RAW_DATA_DIR
from rag_eval_suite.utils.get_root import get_project_root
from rag_eval_suite.utils.loaders import load_yaml

PROJECT_ROOT = get_project_root()
DATA_PATH = PROJECT_ROOT / DATA_DIR
RAW_DATA_PATH = DATA_PATH / RAW_DATA_DIR
PARAM_FILE_PATH = PROJECT_ROOT / "config/params.yaml"
MILVUS_COLLECTION_NAME = load_yaml(PARAM_FILE_PATH)["milvus_db"]["collection_name"]

__all__ = [
    "DATA_PATH",
    "MILVUS_COLLECTION_NAME",
    "PARAM_FILE_PATH",
    "PROJECT_ROOT",
    "RAW_DATA_PATH",
]
