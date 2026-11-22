import torch

from openpi.training.cpu_offload_adamw import CPUOffloadAdamW


def _model(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4))


def _train(model, opt, steps=5):
    torch.manual_seed(1)
    for i in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = 1e-2 * (i + 1)
        x = torch.randn(32, 8)
        loss = model(x).pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)


def test_matches_torch_adamw():
    kw = {"lr": 1e-2, "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 1e-4}
    ref, off = _model(0), _model(0)
    _train(ref, torch.optim.AdamW(ref.parameters(), **kw))
    _train(off, CPUOffloadAdamW(off.parameters(), **kw))
    for a, b in zip(ref.parameters(), off.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_state_dict_roundtrip():
    kw = {"lr": 1e-2, "weight_decay": 1e-4}
    a, b = _model(0), _model(0)
    opt_a = CPUOffloadAdamW(a.parameters(), **kw)
    _train(a, opt_a, steps=2)
    b.load_state_dict(a.state_dict())
    opt_b = CPUOffloadAdamW(b.parameters(), **kw)
    opt_b.load_state_dict(opt_a.state_dict())
    _train(a, opt_a, steps=2)
    _train(b, opt_b, steps=2)
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(p, q)


def test_resume_from_mmap_file(tmp_path):
    # Mirrors scripts/train_pytorch.py load_checkpoint: optimizer.pt loaded to CPU with mmap.
    kw = {"lr": 1e-2, "weight_decay": 1e-4}
    a, b = _model(0), _model(0)
    opt_a = CPUOffloadAdamW(a.parameters(), **kw)
    _train(a, opt_a, steps=2)
    torch.save(opt_a.state_dict(), tmp_path / "optimizer.pt")
    b.load_state_dict(a.state_dict())
    opt_b = CPUOffloadAdamW(b.parameters(), **kw)
    opt_b.load_state_dict(torch.load(tmp_path / "optimizer.pt", map_location="cpu", mmap=True, weights_only=False))
    _train(a, opt_a, steps=2)
    _train(b, opt_b, steps=2)
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(p, q)
