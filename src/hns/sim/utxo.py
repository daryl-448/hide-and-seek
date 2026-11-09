"""UTXO-lite ledger (SPEC §9.2, §9.7).

Each address holds spendable outputs. A tx spends **all** spendable outputs
of its input addresses (the caller lists them), pays fixed-amount outputs,
and splits the remainder ``in - fixed - fee`` equally over its ``rest``
outputs (change, or a template's split). ``fee = feerate * size`` with
``size = 10 + 148 n_in + 34 n_out`` over inputs (spent outputs) and outputs.
So ``in = out + fee`` holds by construction.

Genesis outputs fund addresses from outside the world (§9.7): their funding
tid is ``GENESIS``, and they never become a tx, a flow edge or a feature.
A flow edge ``a -> b`` exists when ``b`` spends an output of ``a`` and both
are in the same time step (§9.2). Frozen outputs (a flagged illicit tx's,
§9.3) stay in the ledger but are never spendable.

Only committed txs get a tid. Proposals are :class:`TxSpec` objects until the
env commits them, so a reverted tx leaves no trace here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hns.feats.tx_feats import tx_size

GENESIS = -1


@dataclass
class TxSpec:
    utxos: list[int]                                         # spent outputs: all spendable outputs of the input addresses
    pay: list[tuple[int, float]] = field(default_factory=list)  # fixed-amount outputs (address, BTC)
    rest: list[int] = field(default_factory=list)            # outputs sharing the remainder equally
    feerate: float = 0.0
    role: int = 0
    entity: int = -1
    illicit: bool = False
    meta: dict = field(default_factory=dict)                 # owner bookkeeping (not part of the ledger)

    @property
    def n_out(self) -> int:
        return len(self.pay) + len(self.rest)

    def fee(self) -> float:
        return self.feerate * tx_size(len(self.utxos), self.n_out)


@dataclass
class Resolved:
    spec: TxSpec
    in_total: float
    outputs: list[tuple[int, float]]
    fee: float


class Ledger:
    def __init__(self):
        self.addr_role: list[int] = []
        self.addr_owner: list[int] = []
        self.addr_utxos: list[dict[int, None]] = []   # spendable outputs, insertion-ordered
        self.balance: list[float] = []                 # spendable BTC per address
        self.u_addr: list[int] = []
        self.u_amt: list[float] = []
        self.u_tid: list[int] = []
        self.u_spent: list[bool] = []
        self.u_frozen: list[bool] = []
        self.genesis_total = 0.0
        # committed txs
        self.tx_ts: list[int] = []
        self.tx_sub: list[int] = []
        self.tx_block: list[int] = []
        self.tx_fee: list[float] = []
        self.tx_role: list[int] = []
        self.tx_entity: list[int] = []
        self.tx_truth: list[bool] = []
        self.tx_label: list[int] = []     # observed class: 1 illicit, 2 licit, 3 unknown (§9.4)
        self.tx_frozen: list[bool] = []
        self.in_rows: list[tuple[int, int, float]] = []   # (tid, address, amount), one per spent output
        self.out_rows: list[tuple[int, int, float]] = []  # (tid, address, amount), one per output
        self.edges: dict[tuple[int, int], None] = {}

    # --- addresses and genesis
    def new_address(self, role: int, owner: int) -> int:
        self.addr_role.append(role)
        self.addr_owner.append(owner)
        self.addr_utxos.append({})
        self.balance.append(0.0)
        return len(self.addr_role) - 1

    def _new_utxo(self, addr: int, amt: float, tid: int, frozen: bool) -> int:
        u = len(self.u_addr)
        self.u_addr.append(addr)
        self.u_amt.append(amt)
        self.u_tid.append(tid)
        self.u_spent.append(False)
        self.u_frozen.append(frozen)
        if not frozen:
            self.addr_utxos[addr][u] = None
            self.balance[addr] += amt
        return u

    def genesis(self, addr: int, amount: float) -> int:
        self.genesis_total += amount
        return self._new_utxo(addr, amount, GENESIS, False)

    def spendable(self, addrs) -> list[int]:
        """All spendable outputs of ``addrs``, in a deterministic order."""
        out = []
        for a in dict.fromkeys(addrs):
            out.extend(self.addr_utxos[a])
        return out

    # --- txs
    def resolve(self, spec: TxSpec) -> Resolved | None:
        """Amounts of a proposal on the current state, or ``None`` if it is invalid
        (no inputs, a spent/frozen input, no output, or not enough BTC)."""
        if not spec.utxos or spec.n_out == 0 or len(set(spec.utxos)) != len(spec.utxos):
            return None
        if any(self.u_spent[u] or self.u_frozen[u] for u in spec.utxos):
            return None
        in_total = sum(self.u_amt[u] for u in spec.utxos)
        fee = spec.fee()
        rem = in_total - sum(a for _, a in spec.pay) - fee
        if any(a <= 0 for _, a in spec.pay) or rem < 0 or (spec.rest and rem <= 0) or (not spec.rest and rem > 1e-12):
            return None
        outputs = list(spec.pay)
        if spec.rest:
            share = rem / len(spec.rest)
            outputs += [(a, share) for a in spec.rest]
        else:
            fee += rem  # rounding residue of an exact spend
        return Resolved(spec, in_total, outputs, fee)

    def commit(self, r: Resolved, ts: int, sub: int, block: int, frozen: bool, label: int) -> int:
        tid = len(self.tx_ts)
        s = r.spec
        for u in s.utxos:
            a = self.u_addr[u]
            self.u_spent[u] = True
            del self.addr_utxos[a][u]
            self.balance[a] -= self.u_amt[u]
            if not self.addr_utxos[a]:
                self.balance[a] = 0.0  # no float drift on emptied addresses
            self.in_rows.append((tid, a, self.u_amt[u]))
            p = self.u_tid[u]
            if p != GENESIS and self.tx_ts[p] == ts:
                self.edges[(p, tid)] = None
        for a, amt in r.outputs:
            self._new_utxo(a, amt, tid, frozen)
            self.out_rows.append((tid, a, amt))
        self.tx_ts.append(ts)
        self.tx_sub.append(sub)
        self.tx_block.append(block)
        self.tx_fee.append(r.fee)
        self.tx_role.append(s.role)
        self.tx_entity.append(s.entity)
        self.tx_truth.append(s.illicit)
        self.tx_label.append(label)
        self.tx_frozen.append(frozen)
        return tid

    # --- accounting
    def unspent_total(self) -> tuple[float, float]:
        """(spendable BTC, frozen BTC) over unspent outputs."""
        sp = fr = 0.0
        for amt, spent, frz in zip(self.u_amt, self.u_spent, self.u_frozen):
            if not spent:
                if frz:
                    fr += amt
                else:
                    sp += amt
        return sp, fr

    @property
    def n_tx(self) -> int:
        return len(self.tx_ts)
