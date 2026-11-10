"""M5 diagnostic: what do a finished run's launderer Q-networks output?

Loads <run>/ckpt/final.pt, plays two episodes of a fresh multiverse world with the trained policies at
epsilon 0.05 (no update, nothing written), and prints per branch the mean Q of every option, separately for
states with a tranche in flight and idle states (funds at the source, all six branches active). For idle
states it also prints the SPEC §10 target value of the state, mean_b max_a Q_b, minus Q(wait).

Usage: uv run python scripts/m5_qprobe.py runs/marl/nda/0
"""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import yaml
import hns.marl.train as mt
from hns.marl.learner import MarlConfig
from hns.sim.env import SimEnv
from hns.sim.live import LiveGraph
from hns.sim.multiverse import draw_world
from hns.sim.world import WorldParams, SUBSTEPS

run = Path(sys.argv[1])
cfgd = yaml.safe_load((run / "config.yaml").read_text())
cfg = MarlConfig(**{k: v for k, v in cfgd.items() if k in MarlConfig.__dataclass_fields__} | {"arm": "nda"})
centre = WorldParams.from_dict(cfgd["centre"])
tr = mt.build(cfg, torch.device("cuda"), centre, None, lambda m: None)
ck = torch.load(run / "ckpt" / "final.pt", map_location="cuda", weights_only=False)
for k, m in tr._models().items():
    m.load_state_dict(ck["models"][k])
tr.episode = 999
rec = defaultdict(list)
L = tr.roles["launderer"]
orig = L.q_values
sizes, names = L.sizes, list(L.branches)
def spy(ob, masks):
    q = orig(ob, masks)
    o = 0
    full = ob.x_c[:, 0] == 0          # no tranche in flight: all six branches active
    idle = np.flatnonzero(full)
    if len(idle):
        parts, o2 = [], 0
        for n in sizes:
            parts.append(np.nanmax(np.where(np.isfinite(q[idle, o2:o2 + n]), q[idle, o2:o2 + n], np.nan), 1)); o2 += n
        rec["target_minus_wait"].extend(np.mean(parts, 0) - q[idle, 0])     # SPEC target value of the state - Q(wait)
        rec["target_minus_initmax"].extend(np.mean(parts, 0) - parts[0])
        rec["greedy_go"].extend(q[idle, 1] > q[idle, 0])
        rec["q_wait"].extend(q[idle, 0])
    for b, n in zip(names, sizes):
        qb = q[:, o:o + n]
        qb = np.where(np.isfinite(qb), qb, np.nan)
        for j in range(len(q)):
            rec[(b, "idle" if full[j] else "flight")].append(qb[j])
        o += n
    return q
L.q_values = spy
ss = np.random.SeedSequence([cfg.seed, 12345])
params = draw_world(centre, np.random.default_rng(ss.spawn(1)[0]))
env = SimEnv(params, int(ss.generate_state(1)[0]), *tr.policies)
env.graph = LiveGraph()
tr.det.pool.begin(env.graph, 1, params.mask_rate)
ep = defaultdict(float)
for e in range(2):
    env.start_episode(e)
    for _ in range(SUBSTEPS):
        res = env.step(); tr.after_step(env, res, ep)
    env.end_episode()
print(run, "n_L", params.n_launderers, "tranches", int(env.acct["strategy"].sum()), "cashed", round(env.acct["cashed"], 2), "frozen", round(env.acct["frozen"], 2), "proceeds", round(env.acct["proceeds"], 2))
def g(k):
    return np.array(rec[k], float)


print(f"  idle states n={len(g('q_wait'))}: Q(wait) mean {g('q_wait').mean():.2f}; greedy initiate = go in {g('greedy_go').mean():.2f}; "
      f"SPEC target value - Q(wait): mean {g('target_minus_wait').mean():+.3f}; - max_init Q: mean {g('target_minus_initmax').mean():+.3f} "
      f"(share > 0: {(g('target_minus_initmax') > 0).mean():.2f})")
