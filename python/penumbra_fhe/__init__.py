"""
Penumbra FHE: A homomorphic inference engine for privacy-preserving ML.
"""

from .errors import (
    PenumbraError,
    PenumbraIngestionError,
    PenumbraCompilerError,
    PenumbraRuntimeError,
    PenumbraDepthBudgetError,
)

__all__ = [
    "PenumbraError",
    "PenumbraIngestionError",
    "PenumbraCompilerError",
    "PenumbraRuntimeError",
    "PenumbraDepthBudgetError",
]
