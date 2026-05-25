"""Exception classes for Penumbra FHE."""


class PenumbraError(Exception):
    """Base class for all Penumbra exceptions."""

    pass


class PenumbraIngestionError(PenumbraError):
    """Raised when ONNX ingestion fails."""

    pass


class PenumbraCompilerError(PenumbraError):
    """Raised during IR compilation and lowering."""

    pass


class PenumbraRuntimeError(PenumbraError):
    """Raised during FHE runtime execution."""

    pass


class PenumbraDepthBudgetError(PenumbraError):
    """Raised when the multiplicative depth budget is exceeded."""

    pass
