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
    "Ciphertext",
    "ClientKey",
    "PenumbraCompilerError",
    "PenumbraDepthBudgetError",
    "PenumbraError",
    "PenumbraIngestionError",
    "PenumbraRuntimeError",
    "SecurityParams",
    "ServerKey",
    "decrypt",
    "encrypt",
    "keygen",
    "set_server_key",
    "version",
]

# Expose the new bindings
SecurityParams = _bindings.SecurityParams
ClientKey = _bindings.ClientKey
ServerKey = _bindings.ServerKey
Ciphertext = _bindings.Ciphertext
keygen = _bindings.keygen
encrypt = _bindings.encrypt
decrypt = _bindings.decrypt
set_server_key = _bindings.set_server_key
