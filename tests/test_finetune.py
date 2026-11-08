"""Fine-tune loop: per-epoch resume from ckpt/last.pt (SPEC §6)."""

import json

import numpy as np
import pytest
import torch

from hns.models.hrg import Views
from hns.transfer.finetune import RunConfig, train


class FakeData:
    """One-node views over 40 txs with 3 features; illicit iff feature 0 > 0."""

    n_tx, d_tx, d_w, d_anon = 100, 3, 2, 0

    def __init__(self):
        rng = np.random.default_rng(0)
        self.x = rng.standard_normal((100, 3)).astype(np.float32)

    def targets(self, split, fraction=1.0):
        g = np.arange(0, 40) if split == "fit" else np.arange(40, 60)
        return {"g": g, "t": np.full(len(g), -1), "y": (self.x[g, 0] > 0).astype(np.float32)}

    def batch(self, g, t, seed):
        n = len(g)
        v = Views(node_g=np.asarray(g, np.int64), hop=np.zeros(n, np.int8), view=np.arange(n, dtype=np.int32),
                  target=np.arange(n), src=np.zeros(0, np.int64), dst=np.zeros(0, np.int64),
                  etype=np.zeros(0, np.int8), logmult=np.zeros(0, np.float32))
        return v, {"x_tx": self.x[g], "x_w": np.zeros((0, 2), np.float32)}


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA"))])
def test_resume_from_last_checkpoint(tmp_path, device):
    data = FakeData()
    cfg = RunConfig("T", 0, 1.0, 0.1, batch=16, micro_batch=8, max_epochs=1, patience=50)
    train(cfg, data, tmp_path, torch.device(device), log=lambda *_: None)
    out = cfg.run_dir(tmp_path)
    (out / "result.json").unlink()  # force the resume path instead of "done, reusing"
    res = train(cfg, data, tmp_path, torch.device(device), log=lambda *_: None)
    assert res["best_epoch"] == 0
    lines = (out / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(x)["epoch"] for x in lines] == [0]
