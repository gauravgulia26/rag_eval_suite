"""Utilities for loading configuration files."""

from os import PathLike
from pathlib import Path

import yaml


def load_yaml(path: str | PathLike[str]) -> dict:
    """Load and safely parse a YAML file.

    Args:
        path: Path to the YAML file to load.

    Returns:
        The Python value represented by the YAML document. An empty document
        returns ``None``, matching :func:`yaml.safe_load`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    with Path(path).open(encoding="utf-8") as yaml_file:
        return yaml.safe_load(yaml_file)


__all__ = ["load_yaml"]
