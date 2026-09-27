# Python API Reference

This page documents the public Python API of Penumbra-FHE.

## Model Loading and Quantization

::: penumbra.onnx_loader.load_onnx
::: penumbra.onnx_loader.UnsupportedModelError
::: penumbra.model.Model

## Float Layers

::: penumbra.layers.Conv2d
::: penumbra.layers.Linear
::: penumbra.layers.Pool
::: penumbra.layers.Activation
::: penumbra.layers.Add
::: penumbra.layers.Concat
::: penumbra.layers.Split
::: penumbra.layers.LayerNode
::: penumbra.layers.QuantConfig

## Encrypted Inference & Key Management

::: penumbra.client.KeySet
::: penumbra.client.run_encrypted
::: penumbra.client.available_backends
::: penumbra._penumbra.CryptoProfile

## Intermediate Representation (IR)

::: penumbra.ir.SCHEMA_VERSION
::: penumbra.ir.Graph
::: penumbra.ir.Node
::: penumbra.ir.OpSpec
::: penumbra.ir.LinearSpec
::: penumbra.ir.Conv2dSpec
::: penumbra.ir.ActivationSpec
::: penumbra.ir.ArgmaxSpec
::: penumbra.ir.CompareSpec
::: penumbra.ir.RequantSpec
::: penumbra.ir.PoolSpec
::: penumbra.ir.AddSpec
::: penumbra.ir.ConcatSpec
::: penumbra.ir.SplitSpec
::: penumbra.ir.build_linear_argmax_graph
::: penumbra.ir.topological_nodes

## Bit-Width Budget and Lowering

::: penumbra.bitwidth.output_bits
::: penumbra.bitwidth.output_bits_multi
::: penumbra.bitwidth.propagate_bit_widths
::: penumbra.bitwidth.minimal_num_blocks
::: penumbra.bitwidth.check_bit_width_budget
::: penumbra.bitwidth.radix_capacity_bits
::: penumbra.compile.insert_requants

## Reference Cleartext Oracle

::: penumbra.reference.evaluate_graph_int
