"""N-step return wrapper for replay buffers."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class NStepBuffer:
    """Wraps a replay buffer to compute n-step returns.

    Accumulates transitions in a deque and pushes to the underlying
    buffer with the discounted n-step return once *n* transitions have
    been collected.  At episode boundaries (``done=True``) the remaining
    transitions are flushed with shorter-than-n returns.

    The stored transition ``(s_0, a_0, R_n, s_n, done_n)`` has:

    * ``R_n = r_0 + γ r_1 + … + γ^{n-1} r_{n-1}``
    * ``s_n`` — state *n* steps ahead (or terminal state)
    * ``done_n`` — whether ``s_n`` is terminal

    The agent must use ``γ^n`` (not ``γ``) for the bootstrap term.
    """

    def __init__(self, buffer: Any, n: int, gamma: float) -> None:
        self.buffer = buffer
        self.n = n
        self.gamma = gamma
        self._deque: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque()

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition; commit to the real buffer when ready."""
        self._deque.append((state, action, reward, next_state, done))

        if len(self._deque) == self.n:
            self._commit()

        if done:
            self._flush()

    # ── internals ─────────────────────────────────────────────────────────

    def _nstep_return(self) -> float:
        """Compute discounted return from current deque contents."""
        R = 0.0
        for i in reversed(range(len(self._deque))):
            R = self._deque[i][2] + self.gamma * R
        return R

    def _commit(self) -> None:
        """Push the oldest transition with its n-step return."""
        R = self._nstep_return()
        s, a, _, _, _ = self._deque[0]
        _, _, _, s_n, done_n = self._deque[-1]
        self.buffer.push(s, a, R, s_n, done_n)
        self._deque.popleft()

    def _flush(self) -> None:
        """Flush remaining transitions at episode end."""
        while self._deque:
            R = self._nstep_return()
            s, a, _, _, _ = self._deque[0]
            _, _, _, s_n, done_n = self._deque[-1]
            self.buffer.push(s, a, R, s_n, done_n)
            self._deque.popleft()

    # ── delegate to underlying buffer ─────────────────────────────────────

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        return self.buffer.sample(*args, **kwargs)

    def update_priorities(self, *args: Any, **kwargs: Any) -> Any:
        return self.buffer.update_priorities(*args, **kwargs)

    def __len__(self) -> int:
        return len(self.buffer)
