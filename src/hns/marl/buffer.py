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

**Archive** (repair cycle 2, ``replay = "archive"``): next to the ring, a
:class:`Archive` keeps a lifetime-uniform reservoir of finalized transitions
(Algorithm R: the k-th finalized transition is kept with probability
``capacity / k``). An archived transition owns its next observation, because
the ring's ``nxt`` ids only hold while the next entry is still in the ring. A
batch then draws ``ARCHIVE_SAMPLE`` of its rows from the archive and the rest
from the ring. The reservoir draws from its own seeded generator, which is
saved with the buffer.
"""

from __future__ import annotations

import numpy as np

PENDING, TERMINAL = -2, -1
ARCHIVE_SHARE, ARCHIVE_SAMPLE = 0.2, 0.25     # of the buffer's capacity; of a batch's rows
_KEYS = ("masks", "active", "action", "realized", "reward", "done", "next_masks", "next_active", "t")


class Archive:
    """A lifetime-uniform reservoir of finalized transitions, each stored whole."""

    def __init__(self, capacity: int, sizes: list[int], seed):
        self.cap, self.n, self.seen = capacity, 0, 0
        self.rng = np.random.default_rng(seed)
        nb, w = len(sizes), sum(sizes)
        self.obs, self.next_obs = [None] * capacity, [None] * capacity
        self.masks, self.next_masks = np.ones((capacity, w), bool), np.ones((capacity, w), bool)
        self.active, self.next_active = np.zeros((capacity, nb), bool), np.zeros((capacity, nb), bool)
        self.action, self.realized = np.zeros((capacity, nb), np.int64), np.zeros((capacity, nb), bool)
        self.reward, self.done = np.zeros(capacity, np.float32), np.zeros(capacity, bool)
        self.t, self.gap = np.zeros(capacity, np.int64), np.zeros(capacity, np.int64)

    def offer(self, tr: dict) -> None:
        self.seen += 1
        s = self.n if self.n < self.cap else int(self.rng.integers(0, self.seen))
        if s >= self.cap:
            return
        self.n = max(self.n, s + 1)
        self.obs[s], self.next_obs[s], self.gap[s] = tr["obs"], tr["next_obs"], tr["gap"]
        for k in _KEYS:
            getattr(self, k)[s] = tr[k]

    def sample(self, k: int, rng: np.random.Generator) -> dict:
        s = rng.integers(0, self.n, k)
        live = ~self.done[s]
        out = {key: getattr(self, key)[s] for key in _KEYS}
        return out | {"obs": [self.obs[i] for i in s], "next_obs": [self.next_obs[i] for i in s[live]],
                      "next_masks": self.next_masks[s[live]], "next_active": self.next_active[s[live]],
                      "gap": self.gap[s[live]]}


class Replay:
    def __init__(self, capacity: int, sizes: list[int], archive: int = 0, seed=0):
        self.cap, self.sizes, self.n = capacity, list(sizes), 0
        self.arch = Archive(archive, sizes, seed) if archive else None
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

    def _archive(self, s: int, n: int | None) -> None:
        """Offer the transition of slot ``s``, finalized just now, to the archive; ``n``: its next state's slot."""
        tr = {k: getattr(self, k)[s].copy() for k in ("masks", "active", "action", "realized", "reward", "t")}
        tr |= {"obs": self.obs[s], "done": n is None, "next_obs": None if n is None else self.obs[n],
               "next_masks": True if n is None else self.masks[n].copy(),
               "next_active": False if n is None else self.active[n].copy(),
               "gap": 0 if n is None else self.t[n] - self.t[s]}
        self.arch.offer(tr)

    def link(self, i: int, j: int) -> None:
        """Entry ``j`` holds the next state of entry ``i``."""
        if self.alive(i):
            self.nxt[i % self.cap] = j
            if self.arch is not None:
                self._archive(i % self.cap, j % self.cap)

    def end(self, i: int) -> None:
        if self.alive(i):
            self.nxt[i % self.cap] = TERMINAL
            if self.arch is not None:
                self._archive(i % self.cap, None)

    @property
    def ready(self) -> int:
        """Finalized transitions in the buffer."""
        return int(np.count_nonzero(self.nxt[:min(self.n, self.cap)] != PENDING))

    def sample(self, k: int, rng: np.random.Generator) -> dict:
        """``k`` finalized transitions, uniformly with replacement; with an archive, ``ARCHIVE_SAMPLE`` of them come
        from it (the last rows; ``n_archive`` says how many)."""
        ka = int(round(ARCHIVE_SAMPLE * k)) if self.arch is not None and self.arch.n else 0
        slots = np.flatnonzero(self.nxt[:min(self.n, self.cap)] != PENDING)
        s = slots[rng.integers(0, len(slots), k - ka)]
        nxt = self.nxt[s]
        live = nxt >= 0
        ns = nxt[live] % self.cap
        out = {"obs": [self.obs[i] for i in s], "masks": self.masks[s], "active": self.active[s],
               "action": self.action[s], "realized": self.realized[s], "reward": self.reward[s], "t": self.t[s],
               "done": ~live, "next_obs": [self.obs[i] for i in ns], "next_masks": self.masks[ns],
               "next_active": self.active[ns], "gap": self.t[ns] - self.t[s[live]], "n_archive": ka}
        if ka:
            a = self.arch.sample(ka, rng)
            out = {key: v if key == "n_archive" else (v + a[key] if isinstance(v, list) else np.concatenate([v, a[key]]))
                   for key, v in out.items()}
        return out

    def support(self) -> dict:
        """What the buffer holds (repair cycle 2's replay log): entries ever added, finalized entries, the sub-steps
        they were observed at (quantiles 0 / 10 / 50 / 90 / 100 %), and how often each option of each branch was
        taken and realized; the same for the archive."""
        def part(t, action, realized, n):
            counts = [np.bincount(action[realized[:, b], b], minlength=m).tolist() for b, m in enumerate(self.sizes)]
            return {"stored": int(n), "t": np.quantile(t, [0, 0.1, 0.5, 0.9, 1]).tolist() if n else [], "options": counts}

        k = min(self.n, self.cap)
        fin = self.nxt[:k] != PENDING
        out = {"added": self.n, **part(self.t[:k][fin], self.action[:k][fin], self.realized[:k][fin], fin.sum())}
        if self.arch is not None:
            a = self.arch
            out["archive"] = {"seen": a.seen, **part(a.t[:a.n], a.action[:a.n], a.realized[:a.n], a.n)}
        return out


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
