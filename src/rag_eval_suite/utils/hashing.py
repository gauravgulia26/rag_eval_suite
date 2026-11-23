"""Utilities for hashing Python objects."""

import hashlib
from typing import Any


def hash_object(obj: Any) -> str:
    """Return a deterministic SHA-256 digest for an object's text value.

    Args:
        obj: The object to hash.

    Returns:
        The object's SHA-256 digest as a hexadecimal string.
    """
    return hashlib.sha256(str(obj).encode("utf-8")).hexdigest()


__all__ = ["hash_object"]
