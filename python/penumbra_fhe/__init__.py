"""Penumbra FHE: A homomorphic inference engine for privacy-preserving ML."""

from .errors import (
    PenumbraCompilerError,
    PenumbraDepthBudgetError,
    PenumbraError,
    PenumbraIngestionError,
    PenumbraRuntimeError,
)

__all__ = [
    "PenumbraCompilerError",
    "PenumbraDepthBudgetError",
    "PenumbraError",
    "PenumbraIngestionError",
    "PenumbraRuntimeError",
]
