"""ONNX ingestion for Penumbra FHE."""

import onnx  # type: ignore

from .errors import PenumbraIngestionError


def load_onnx(path: str) -> onnx.ModelProto:
    """
    Load an ONNX model from the given path.

    :param path: Path to the ONNX file.
    :returns: The loaded ONNX model.
    :raises PenumbraIngestionError: If the model cannot be loaded.
    """
    try:
        return onnx.load(path)
    except Exception as e:
        raise PenumbraIngestionError(f"Failed to load ONNX model: {e}") from e
