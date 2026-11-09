"""M4: hand-built UTXO ledger: conservation, the flow edge rule, every manifest definition and sentinel."""

import numpy as np
import polars as pl
import pytest

from hns.feats import manifest, tx_feats, wallet_feats
from hns.sim.utxo import GENESIS, Ledger, TxSpec
from hns.sim.world import ROLE_INNOCENT

FR = 1e-6  # feerate BTC/byte


def _ledger():
    """Addresses a, b, c, d; genesis a = 1.0, b = 0.5. Four txs:

    t0 (ts 1, block 100): a -> {c: 0.4, a: change}           (self-loop a->a)
    t1 (ts 1, block 140): c, b -> {d: 0.3, d: 0.2, c: change}  (spends t0's output: flow edge t0 -> t1)
    t2 (ts 2, block 300): a -> {b: rest}                       (spends t0's change across a step: no edge)
    t3 (ts 2, block 340): d -> {d: rest}                       (spends t1's outputs across a step: no edge)
    """
    led = Ledger()
    a, b, c, d = (led.new_address(ROLE_INNOCENT, i) for i in range(4))
    led.genesis(a, 1.0)
    led.genesis(b, 0.5)
    txs = [
        (TxSpec(led.spendable([a]), pay=[(c, 0.4)], rest=[a], feerate=FR), 1, 100),
        (None, 1, 140),
        (None, 2, 300),
        (None, 2, 340),
    ]
    tids = []
    for k, (spec, ts, block) in enumerate(txs):
        if k == 1:
            spec = TxSpec(led.spendable([c, b]), pay=[(d, 0.3), (d, 0.2)], rest=[c], feerate=FR)
        elif k == 2:
            spec = TxSpec(led.spendable([a]), rest=[b], feerate=FR)
        elif k == 3:
            spec = TxSpec(led.spendable([d]), rest=[d], feerate=FR)
        r = led.resolve(spec)
        assert r is not None
        tids.append(led.commit(r, ts=ts, sub=k, block=block, frozen=False, label=3))
    return led, (a, b, c, d), tids


def _tables(led):
    ins = pl.DataFrame(led.in_rows, schema=["tid", "wid", "amount"], orient="row")
    outs = pl.DataFrame(led.out_rows, schema=["tid", "wid", "amount"], orient="row")
    edges = pl.DataFrame(list(led.edges), schema=["src", "dst"], orient="row")
    return ins, outs, edges


def test_conservation_per_tx_and_global():
    led, _, tids = _ledger()
    ins, outs, _ = _tables(led)
    for t in tids:
        i = ins.filter(pl.col("tid") == t)["amount"].sum()
        o = outs.filter(pl.col("tid") == t)["amount"].sum()
        assert i == pytest.approx(o + led.tx_fee[t], abs=1e-12)
    sp, fr = led.unspent_total()
    assert led.genesis_total == pytest.approx(sp + fr + sum(led.tx_fee), abs=1e-12)
    # size and fee: t1 spends 2 outputs and creates 3
    assert led.tx_fee[1] == pytest.approx(FR * (10 + 148 * 2 + 34 * 3))


def test_flow_edge_rule():
    led, _, _ = _ledger()
    assert list(led.edges) == [(0, 1)]           # same step only; never from genesis
    assert all(led.u_tid[u] == GENESIS for u in range(2))


def test_invalid_specs_resolve_to_none():
    led, (a, b, c, d), _ = _ledger()
    u = led.spendable([b])
    assert led.resolve(TxSpec(u, pay=[(a, 10.0)], rest=[b], feerate=FR)) is None          # insufficient
    spent = [x for x in range(len(led.u_addr)) if led.u_spent[x]]
    assert led.resolve(TxSpec(spent[:1], rest=[a], feerate=FR)) is None                 # already spent
    assert led.resolve(TxSpec([], rest=[a], feerate=FR)) is None                          # no input


def test_frozen_outputs_never_spendable():
    led, (a, b, c, d), _ = _ledger()
    before = led.balance[c]
    r = led.resolve(TxSpec(led.spendable([b]), rest=[c], feerate=FR))
    led.commit(r, ts=2, sub=9, block=350, frozen=True, label=1)
    frozen_u = len(led.u_addr) - 1
    assert led.u_frozen[frozen_u] and frozen_u not in led.spendable([c])
    assert led.balance[c] == pytest.approx(before)
    assert led.resolve(TxSpec([frozen_u], rest=[a], feerate=FR)) is None
    sp, fr = led.unspent_total()
    assert fr == pytest.approx(r.outputs[0][1]) and led.genesis_total == pytest.approx(sp + fr + sum(led.tx_fee))


def _expected_tx(led):
    """Hand-computed 17 features for t0..t3."""
    g = 1e-6
    f0, f1, f2, f3 = (g * (10 + 148 * 1 + 34 * 2), g * (10 + 148 * 2 + 34 * 3), g * (10 + 148 + 34), g * (10 + 148 * 2 + 34))
    ch0 = 1.0 - 0.4 - f0
    ch1 = 0.4 + 0.5 - 0.5 - f1
    rows = {
        0: dict(ins=[1.0], outs=[0.4, ch0], deg=(0, 1), fee=f0),
        1: dict(ins=[0.4, 0.5], outs=[0.3, 0.2, ch1], deg=(1, 0), fee=f1),
        2: dict(ins=[ch0], outs=[ch0 - f2], deg=(0, 0), fee=f2),
        3: dict(ins=[0.3, 0.2], outs=[0.5 - f3], deg=(0, 0), fee=f3),
    }
    out = {}
    for t, r in rows.items():
        i, o = np.array(r["ins"]), np.array(r["outs"])
        out[t] = {"in_txs_degree": r["deg"][0], "out_txs_degree": r["deg"][1], "total_BTC": o.sum(), "fees": r["fee"],
                  "size": 10 + 148 * len(i) + 34 * len(o), "num_input_addresses": len(i), "num_output_addresses": len(o),
                  "in_BTC_min": i.min(), "in_BTC_max": i.max(), "in_BTC_mean": i.mean(), "in_BTC_median": np.median(i),
                  "in_BTC_total": i.sum(), "out_BTC_min": o.min(), "out_BTC_max": o.max(), "out_BTC_mean": o.mean(),
                  "out_BTC_median": np.median(o), "out_BTC_total": o.sum()}
    return out


def test_every_tx_manifest_feature():
    led, _, tids = _ledger()
    ins, outs, edges = _tables(led)
    got = tx_feats.tx_features(pl.Series("tid", tids), ins, outs, edges)
    exp = _expected_tx(led)
    assert got.columns[1:] == manifest.features("tx")
    for t in tids:
        row = got.filter(pl.col("tid") == t).row(0, named=True)
        for f in manifest.features("tx"):
            assert row[f] == pytest.approx(exp[t][f], abs=1e-12), (t, f)


def _wallet_inputs(led):
    ins, outs, _ = _tables(led)
    agg = lambda df: df.group_by("wid", "tid").agg(pl.col("amount").sum())  # noqa: E731
    feats = tx_feats.tx_features(pl.Series("tid", range(led.n_tx)), ins, outs, pl.DataFrame(
        list(led.edges), schema=["src", "dst"], orient="row"))
    txs = pl.DataFrame({"tid": range(led.n_tx), "ts": led.tx_ts, "block": led.tx_block}).join(feats, on="tid")
    return agg(ins), agg(outs), txs


def test_every_wallet_manifest_feature_and_sentinel():
    led, (a, b, c, d), _ = _ledger()
    lin, lout, txs = _wallet_inputs(led)
    got = {r["wid"]: r for r in wallet_feats.wallet_features(lin, lout, txs).iter_rows(named=True)}
    fee = dict(zip(txs["tid"], txs["fees"]))
    share = {t: fee[t] / v for t, v in zip(txs["tid"], txs["in_BTC_total"])}
    ch0 = 1.0 - 0.4 - fee[0]
    # wallet b: genesis 0.5 -> input of t1 (sent 0.5), output of t2 (received ch0 - f2). Hand-computed:
    recv = [ch0 - fee[2]]
    eb = {
        "num_txs_as_sender": 1, "num_txs_as receiver": 1, "total_txs": 2, "num_timesteps_appeared_in": 2,
        "first_block_appeared_in": 140, "last_block_appeared_in": 300, "lifetime_in_blocks": 160,
        "first_sent_block": 140, "first_received_block": 300,
        "btc_transacted_total": 0.5 + recv[0], "btc_transacted_min": min(0.5, recv[0]),
        "btc_transacted_max": max(0.5, recv[0]), "btc_transacted_mean": (0.5 + recv[0]) / 2,
        "btc_transacted_median": (0.5 + recv[0]) / 2,
        **{f"btc_sent_{s}": v for s, v in zip(("total", "min", "max", "mean", "median"), [0.5] * 5)},
        **{f"btc_received_{s}": recv[0] for s in ("total", "min", "max", "mean", "median")},
        "fees_total": fee[1] + fee[2], "fees_min": min(fee[1], fee[2]), "fees_max": max(fee[1], fee[2]),
        "fees_mean": (fee[1] + fee[2]) / 2, "fees_median": (fee[1] + fee[2]) / 2,
        "fees_as_share_total": share[1] + share[2], "fees_as_share_min": min(share[1], share[2]),
        "fees_as_share_max": max(share[1], share[2]), "fees_as_share_mean": (share[1] + share[2]) / 2,
        "fees_as_share_median": (share[1] + share[2]) / 2,
        **{f"blocks_btwn_txs_{s}": 160 for s in ("total", "min", "max", "mean", "median")},
        # a single input tx and a single output tx: empty gap lists -> sentinel 0
        **{f"blocks_btwn_input_txs_{s}": 0 for s in ("total", "min", "max", "mean", "median")},
        **{f"blocks_btwn_output_txs_{s}": 0 for s in ("total", "min", "max", "mean", "median")},
        # counterparties: t1 -> {d, c}; t2 -> {a}; none repeated
        "num_addr_transacted_multiple": 0,
        "transacted_w_address_total": 3, "transacted_w_address_min": 1, "transacted_w_address_max": 2,
        "transacted_w_address_mean": 1.5, "transacted_w_address_median": 1.5,
    }
    names = wallet_feats.named_features()
    assert len(names) == 55 and set(eb) == set(names)
    for f in names:
        assert got[b][f] == pytest.approx(eb[f], abs=1e-12), f
    # wallet d: receives twice in t1 (one tx), then spends both outputs in t3 back to itself (a self-loop):
    # its own address is never its counterparty
    assert got[d]["num_txs_as_sender"] == 1 and got[d]["btc_received_total"] == pytest.approx(0.5 + (0.5 - fee[3]))
    assert got[d]["btc_received_max"] == pytest.approx(0.5) and got[d]["transacted_w_address_min"] == 0
    # wallet a: inputs t0, t2 (blocks 100, 300), output t0 (change): counterparty c in t0, b in t2
    assert got[a]["blocks_btwn_input_txs_total"] == 200 and got[a]["blocks_btwn_output_txs_max"] == 0
    assert got[a]["num_addr_transacted_multiple"] == 0
    # wallet c: output of t0 (from a), input + output (change) of t1 -> counterparty a (t0), d and b (t1)
    assert got[c]["num_addr_transacted_multiple"] == 0 and got[c]["transacted_w_address_max"] == 2


def test_wallet_sentinels_for_a_single_receive():
    """A wallet that only ever receives, once: every sender-side value and every gap is the sentinel 0 (§7)."""
    led = Ledger()
    a, e = led.new_address(ROLE_INNOCENT, 0), led.new_address(ROLE_INNOCENT, 1)
    led.genesis(a, 1.0)
    led.commit(led.resolve(TxSpec(led.spendable([a]), pay=[(e, 0.3)], rest=[a], feerate=FR)), ts=1, sub=0, block=100,
               frozen=False, label=3)
    lin, lout, txs = _wallet_inputs(led)
    r = wallet_feats.wallet_features(lin, lout, txs).filter(pl.col("wid") == e).row(0, named=True)
    zero = ["num_txs_as_sender", "first_sent_block", "lifetime_in_blocks", "num_addr_transacted_multiple"]
    zero += [f"{p}_{s}" for p in ("btc_sent", "blocks_btwn_txs", "blocks_btwn_input_txs", "blocks_btwn_output_txs")
             for s in ("total", "min", "max", "mean", "median")]
    assert all(r[f] == 0 for f in zero)
    assert r["first_received_block"] == 100 and r["first_block_appeared_in"] == 100 and r["total_txs"] == 1
    assert r["fees_total"] == pytest.approx(txs["fees"][0]) and r["transacted_w_address_total"] == 1


def test_counterparty_repeat_counts():
    """Two txs between the same pair make that counterparty 'multiple'."""
    led = Ledger()
    a, b = led.new_address(ROLE_INNOCENT, 0), led.new_address(ROLE_INNOCENT, 1)
    led.genesis(a, 1.0)
    for k in range(2):
        r = led.resolve(TxSpec(led.spendable([a]), pay=[(b, 0.1)], rest=[a], feerate=FR))
        led.commit(r, ts=1, sub=k, block=100 + k, frozen=False, label=3)
    lin, lout, txs = _wallet_inputs(led)
    got = {r["wid"]: r for r in wallet_feats.wallet_features(lin, lout, txs).iter_rows(named=True)}
    assert got[a]["num_addr_transacted_multiple"] == 1 and got[b]["num_addr_transacted_multiple"] == 1
    assert got[b]["blocks_btwn_output_txs_total"] == 1


def test_manifest_sentinels_are_zero_and_block_sentinel_maps():
    m = manifest.load()
    assert all(e["sentinel"] == 0 for g in ("tx", "wallet_T", "wallet_W") for e in m[g].values())
    assert manifest.apply("block_ts", np.array([0.0]))[0] == m["block_ts"]["sentinel_to"]
