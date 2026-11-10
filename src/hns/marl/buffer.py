"""Replay buffers (SPEC §10): one per role, 100k transitions, 1,000-transition warm-up.

A transition is ``(s, a, R(s, a), r, s', d)``. Each agent's states form a
chain, so a buffer stores every observation once: entry ``i`` holds ``s``,
the branch set and masks of ``s``, the sub-step ``t`` at which ``s`` was
observed, the action and its realized branches, the reward, and ``nxt``,
the id of the entry that holds ``s'`` (the same agent's next decision
point). ``nxt`` is ``PENDING`` until that next state arrives, and
``TERMINAL`` when the world ends first (``d = 1``), or at once for a
transition that is always terminal (the inspector's per-tx flags). A sampled
transition's ``gap`` is ``t(s') - t(s)``, the sub-steps it spans, so the
target can discount by ``gamma ** gap`` (§10).

Ids count up forever; the ring keeps the newest ``capacity`` entries. A
later entry is always newer, so the next state of a live entry is live too.
Only finalized entries are sampled.

:class:`VecReplay` is the same thing for the inspector's global branches,
whose state is a plain vector (the global readout).
"""

from __future__ import annotations

import numpy as np

PENDING, TERMINAL = -2, -1


class Replay:
    def __init__(self, capacity: int, sizes: list[int]):
        self.cap, self.sizes, self.n = capacity, list(sizes), 0
        nb = len(sizes)
        self.obs: list = [None] * capacity
        self.masks = np.ones((capacity, sum(sizes)), bool)
        self.active = np.zeros((capacity, nb), bool)
        self.action = np.zeros((capacity, nb), np.int64)
        self.realized = np.zeros((capacity, nb), bool)
        self.reward = np.zeros(capacity, np.float32)
        self.nxt = np.full(capacity, PENDING, np.int64)
        self.t = np.zeros(capacity, np.int64)

    def alive(self, i: int) -> bool:
        return 0 <= i < self.n and i >= self.n - self.cap

    def add(self, obs, masks: np.ndarray, active: np.ndarray, t: int = 0) -> int:
        """Store a state observed at sub-step ``t``; returns its id. Its transition is filled in by :meth:`set` and
        :meth:`link`/:meth:`end`."""
        i, s = self.n, self.n % self.cap
        self.obs[s], self.masks[s], self.active[s], self.t[s] = obs, masks, active, t
        self.realized[s], self.reward[s], self.nxt[s] = False, 0.0, PENDING
        self.n += 1
        return i

    def t_of(self, i: int) -> int:
        return int(self.t[i % self.cap])

    def set(self, i: int, action: np.ndarray, realized: np.ndarray, reward: float) -> None:
        if self.alive(i):
            s = i % self.cap
            self.action[s], self.realized[s] = action, realized
            self.reward[s] += reward

    def add_reward(self, i: int, r: float) -> None:
        if self.alive(i):
            self.reward[i % self.cap] += r

    def link(self, i: int, j: int) -> None:
        """Entry ``j`` holds the next state of entry ``i``."""
        if self.alive(i):
            self.nxt[i % self.cap] = j

    def end(self, i: int) -> None:
        if self.alive(i):
            self.nxt[i % self.cap] = TERMINAL

    @property
    def ready(self) -> int:
        """Finalized transitions in the buffer."""
        return int(np.count_nonzero(self.nxt[:min(self.n, self.cap)] != PENDING))

    def sample(self, k: int, rng: np.random.Generator) -> dict:
        """``k`` finalized transitions, uniformly with replacement."""
        slots = np.flatnonzero(self.nxt[:min(self.n, self.cap)] != PENDING)
        s = slots[rng.integers(0, len(slots), k)]
        nxt = self.nxt[s]
        live = nxt >= 0
        ns = nxt[live] % self.cap
        return {"obs": [self.obs[i] for i in s], "masks": self.masks[s], "active": self.active[s],
                "action": self.action[s], "realized": self.realized[s], "reward": self.reward[s],
                "done": ~live, "next_obs": [self.obs[i] for i in ns], "next_masks": self.masks[ns],
                "next_active": self.active[ns], "gap": self.t[ns] - self.t[s[live]]}


class VecReplay:
    """Sequential transitions of one agent with vector states: entry ``i``'s next state is entry ``i + 1``."""

    def __init__(self, capacity: int, d: int, sizes: list[int]):
        self.cap, self.sizes, self.n = capacity, list(sizes), 0
        self.state = np.zeros((capacity, d), np.float32)
        self.action = np.zeros((capacity, len(sizes)), np.int64)
        self.reward = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, bool)
        self.final = np.zeros(capacity, bool)

    def add(self, state: np.ndarray, action: np.ndarray) -> int:
        s = self.n % self.cap
        if self.n and not self.final[(self.n - 1) % self.cap]:
            self.final[(self.n - 1) % self.cap] = True       # the previous entry's next state has arrived
        self.state[s], self.action[s], self.reward[s], self.done[s], self.final[s] = state, action, 0.0, False, False
        self.n += 1
        return self.n - 1

    def set_reward(self, r: float) -> None:
        self.reward[(self.n - 1) % self.cap] = r

    def end(self) -> None:
        """The newest entry is terminal (world end)."""
        if self.n:
            s = (self.n - 1) % self.cap
            self.done[s] = self.final[s] = True

    @property
    def ready(self) -> int:
        return int(np.count_nonzero(self.final[:min(self.n, self.cap)]))

    def sample(self, k: int, rng: np.random.Generator) -> dict:
        slots = np.flatnonzero(self.final[:min(self.n, self.cap)])
        s = slots[rng.integers(0, len(slots), k)]
        return {"state": self.state[s], "action": self.action[s], "reward": self.reward[s], "done": self.done[s],
                "next_state": self.state[(s + 1) % self.cap]}
