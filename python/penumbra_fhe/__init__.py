"""Penumbra FHE: A homomorphic inference engine for privacy-preserving ML."""

from . import _bindings
from .errors import (
    PenumbraCompilerError,
    PenumbraDepthBudgetError,
    PenumbraError,
    PenumbraIngestionError,
    PenumbraRuntimeError,
)


def version() -> str:
    """
    Get the version of the penumbra-fhe core library.

    :returns: The version string.
    """
    return _bindings.version()


__all__ = [
    "PenumbraCompilerError",
    "PenumbraDepthBudgetError",
    "PenumbraError",
    "PenumbraIngestionError",
    "PenumbraRuntimeError",
    "version",
]
