# MNIST — the reference example

The first end-to-end use case: **train → quantize → export IR → encrypted inference**.

- **Phase 2:** encrypted logistic regression / 1-layer net on MNIST 0-vs-1, proving the
  narrow waist (`Linear → Activation → Argmax`) and establishing the golden exactness test.
- **Phase 4:** a small CNN on 10-class MNIST, proving multi-layer eval + automatic
  bit-width management (`Conv2d`, `Pool`, `Requant`).

This example contains **no cryptography** — only a model graph and quantized weights
(`PROJECT.md` §4). _Scaffold placeholder; filled in at Phase 2._
