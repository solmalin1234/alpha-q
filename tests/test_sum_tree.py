"""Tests for the sum tree data structure."""

from __future__ import annotations

from alpha_q.memory.sum_tree import SumTree


def test_empty_tree_total_is_zero() -> None:
    tree = SumTree(capacity=8)
    assert tree.total == 0.0


def test_update_single_leaf() -> None:
    tree = SumTree(capacity=4)
    tree.update(0, 1.5)
    assert tree.total == 1.5


def test_update_multiple_leaves() -> None:
    tree = SumTree(capacity=4)
    tree.update(0, 1.0)
    tree.update(1, 2.0)
    tree.update(2, 3.0)
    assert abs(tree.total - 6.0) < 1e-9


def test_update_overwrites_previous_priority() -> None:
    tree = SumTree(capacity=4)
    tree.update(0, 5.0)
    assert tree.total == 5.0
    tree.update(0, 2.0)
    assert abs(tree.total - 2.0) < 1e-9


def test_get_returns_correct_leaf() -> None:
    tree = SumTree(capacity=4)
    tree.update(0, 1.0)
    tree.update(1, 2.0)
    tree.update(2, 3.0)
    tree.update(3, 4.0)
    # total = 10.0, segments: [0,1), [1,3), [3,6), [6,10)

    idx, prio = tree.get(0.5)
    assert idx == 0
    assert prio == 1.0

    idx, prio = tree.get(2.0)
    assert idx == 1
    assert prio == 2.0

    idx, prio = tree.get(5.0)
    assert idx == 2
    assert prio == 3.0

    idx, prio = tree.get(8.0)
    assert idx == 3
    assert prio == 4.0


def test_get_boundary_goes_left() -> None:
    """When cumsum equals the left child exactly, it should go left."""
    tree = SumTree(capacity=4)
    tree.update(0, 1.0)
    tree.update(1, 1.0)
    idx, _ = tree.get(1.0)
    assert idx == 0


def test_proportional_sampling_distribution() -> None:
    """High-priority leaves should be sampled more often."""
    tree = SumTree(capacity=4)
    tree.update(0, 10.0)
    tree.update(1, 1.0)
    tree.update(2, 1.0)
    tree.update(3, 1.0)

    import numpy as np

    counts = np.zeros(4)
    rng = np.random.default_rng(42)
    n_samples = 10000
    for _ in range(n_samples):
        val = rng.uniform(0, tree.total)
        idx, _ = tree.get(val)
        counts[idx] += 1

    # Leaf 0 has ~77% of priority mass (10/13)
    assert counts[0] / n_samples > 0.7
