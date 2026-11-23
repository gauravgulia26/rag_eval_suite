"""Helpers for locating the repository root."""

from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory as a :class:`~pathlib.Path`.

    This module lives at ``<project-root>/src/rag_eval_suite/utils``.  Walking
    up from the resolved module path keeps callers independent of their
    current working directory.
    """

    return Path(__file__).resolve().parents[3]


__all__ = ["get_project_root"]
