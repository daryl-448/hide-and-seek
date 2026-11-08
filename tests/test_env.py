"""M0 environment asserts (SPEC §3, E12): arch list, and one GAT forward/backward on the GPU."""

import os

import pytest
import torch


def test_arch_list_covers_local_and_a30():
    archs = torch.cuda.get_arch_list()
    assert "sm_75" in archs, archs  # GTX 1650 Super
    assert "sm_80" in archs, archs  # A30


def test_no_compiled_pyg_extensions_needed():
    import torch_geometric  # noqa: F401

    for mod in ("torch_scatter", "torch_sparse", "pyg_lib"):
        with pytest.raises(ImportError):
            __import__(mod)


@pytest.mark.skipif(os.environ.get("HNS_NO_GPU") == "1", reason="HNS_NO_GPU=1")
def test_gat_forward_backward_on_gpu():
    from torch_geometric.nn import GATConv

    assert torch.cuda.is_available(), "CUDA not available (set HNS_NO_GPU=1 to skip on a CPU-only host)"
    dev = torch.device("cuda")
    torch.manual_seed(0)
    n, d = 512, 32
    x = torch.randn(n, d, device=dev)
    ei = torch.randint(0, n, (2, 4096), device=dev)
    conv = GATConv(d, 64, heads=4, dropout=0.5).to(dev)
    out = conv(x, ei)
    assert out.shape == (n, 256)
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in conv.parameters())
