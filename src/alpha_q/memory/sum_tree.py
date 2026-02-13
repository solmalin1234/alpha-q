"""Binary sum tree for O(log n) proportional sampling."""

from __future__ import annotations

import numpy as np


class SumTree:
    """A binary tree where each leaf holds a priority value and internal
    nodes store the sum of their children.

    Supports O(log n) priority update and proportional sampling.
    Leaf *i* is stored at tree index ``i + capacity - 1``.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    # ── public API ────────────────────────────────────────────────────────

    def update(self, data_index: int, priority: float) -> None:
        """Set the priority for leaf *data_index*."""
        idx = data_index + self.capacity - 1
        delta = priority - self._tree[idx]
        self._tree[idx] = priority
        while idx > 0:
            idx = (idx - 1) // 2
            self._tree[idx] += delta

    def get(self, cumsum: float) -> tuple[int, float]:
        """Find the leaf whose cumulative sum region contains *cumsum*.

        Returns ``(data_index, priority)``.
        """
        idx = 0  # start at root
        while True:
            left = 2 * idx + 1
            if left >= len(self._tree):  # reached a leaf
                break
            right = left + 1
            if cumsum <= self._tree[left]:
                idx = left
            else:
                cumsum -= self._tree[left]
                idx = right
        data_index = idx - (self.capacity - 1)
        return data_index, float(self._tree[idx])

    @property
    def total(self) -> float:
        """Sum of all priorities (root value)."""
        return float(self._tree[0])
