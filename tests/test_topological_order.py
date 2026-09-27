from __future__ import annotations

import pytest

from penumbra.ir import AddSpec, Node, topological_order
from penumbra.layers import Add, LayerNode, Split
from penumbra.model import Model


def test_already_sorted_graph_keeps_its_order() -> None:
    nodes = [
        Node(name="a", inputs=["x"], outputs=["t0"], op=AddSpec()),
        Node(name="b", inputs=["t0"], outputs=["t1"], op=AddSpec()),
        Node(name="c", inputs=["t1"], outputs=["y"], op=AddSpec()),
    ]
    assert topological_order(nodes, ["x"]) == [0, 1, 2]


def test_ready_ties_break_by_lowest_index() -> None:
    d = Node(name="d", inputs=["x"], outputs=["p"], op=AddSpec())
    c = Node(name="c", inputs=["x"], outputs=["q"], op=AddSpec())
    b = Node(name="b", inputs=["p", "q"], outputs=["r"], op=AddSpec())
    a = Node(name="a", inputs=["r"], outputs=["y"], op=AddSpec())

    nodes1 = [d, c, b, a]
    assert topological_order(nodes1, ["x"]) == [0, 1, 2, 3]

    nodes2 = [b, d, a, c]
    assert topological_order(nodes2, ["x"]) == [1, 3, 0, 2]


def test_undefined_input_fails_loudly() -> None:
    node = Node(name="a", inputs=["x", "missing"], outputs=["y"], op=AddSpec())
    with pytest.raises(ValueError, match="reads tensor 'missing', which no node produces"):
        _ = topological_order([node], ["x"])


def test_duplicate_output_fails_loudly() -> None:
    a = Node(name="a", inputs=["x"], outputs=["t"], op=AddSpec())
    b = Node(name="b", inputs=["x"], outputs=["t"], op=AddSpec())
    with pytest.raises(ValueError, match="writes tensor 't', which already exists"):
        _ = topological_order([a, b], ["x"])

    c = Node(name="c", inputs=["x"], outputs=["x"], op=AddSpec())
    with pytest.raises(ValueError, match="writes tensor 'x', which already exists"):
        _ = topological_order([c], ["x"])


def test_cycle_names_stuck_nodes() -> None:
    a = Node(name="a", inputs=["x"], outputs=["t0"], op=AddSpec())
    b = Node(name="b", inputs=["t0", "t2"], outputs=["t1"], op=AddSpec())
    c = Node(name="c", inputs=["t1"], outputs=["t2"], op=AddSpec())
    with pytest.raises(ValueError, match=r"graph has a cycle: node\(s\) \['b', 'c'\]"):
        _ = topological_order([a, b, c], ["x"])


def test_model_orders_layer_nodes() -> None:
    nodes = [
        LayerNode(name="sum", layer=Add(), inputs=["a", "b"], outputs=["y"]),
        LayerNode(name="split", layer=Split([2, 2]), inputs=["x"], outputs=["a", "b"]),
    ]
    model = Model(nodes, input_name="x")
    assert [n.name for n in model.nodes] == ["split", "sum"]


def test_model_rejects_cyclic_layer_nodes() -> None:
    a = LayerNode(name="a", layer=Add(), inputs=["x"], outputs=["t0"])
    b = LayerNode(name="b", layer=Add(), inputs=["t0", "t2"], outputs=["t1"])
    c = LayerNode(name="c", layer=Add(), inputs=["t1"], outputs=["t2"])
    with pytest.raises(ValueError, match="graph has a cycle"):
        _ = Model([a, b, c], input_name="x")
