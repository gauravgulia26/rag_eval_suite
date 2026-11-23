"""Filesystem paths derived from project-level constants."""

from rag_eval_suite.constants import DATA_DIR, RAW_DATA_DIR
from rag_eval_suite.utils.get_root import get_project_root

PROJECT_ROOT = get_project_root()
DATA_PATH = PROJECT_ROOT / DATA_DIR
RAW_DATA_PATH = DATA_PATH / RAW_DATA_DIR

__all__ = ["DATA_PATH", "PROJECT_ROOT", "RAW_DATA_PATH"]
