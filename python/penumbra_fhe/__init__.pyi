from .errors import (
    PenumbraCompilerError,
    PenumbraDepthBudgetError,
    PenumbraError,
    PenumbraIngestionError,
    PenumbraRuntimeError,
)

def version() -> str: ...

__all__ = [
    "PenumbraCompilerError",
    "PenumbraDepthBudgetError",
    "PenumbraError",
    "PenumbraIngestionError",
    "PenumbraRuntimeError",
    "version",
]
