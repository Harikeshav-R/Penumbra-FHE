"""Bridge to the Rust runtime + the client-side inference round trip.

Drives the runtime to do keygen -> encrypt -> evaluate -> decrypt and return a
prediction, exposing the one-call convenience API ``model.predict_encrypted(x)``
(``PROJECT.md`` §12).

Bridge strategy (``PROJECT.md`` §15):
    - Phase 1-8: IR file + subprocess — Python writes ``model.fhe``, the Rust runtime
      reads it. Simplest; start here.
    - Phase 9: PyO3 in-process bindings — better ergonomics, no file/subprocess round
      trip. Be deliberate about what crosses the boundary (IR + ciphertext handles, not
      giant copies).

Privacy model (``PROJECT.md`` §11): the client holds the secret key (encrypt/decrypt);
the server holds the public server key + plaintext weights and only ever touches
ciphertext.

TODO(phase-2): minimal subprocess bridge to run an exported model end to end.
"""
