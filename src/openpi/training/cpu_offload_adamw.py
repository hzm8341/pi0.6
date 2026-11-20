"""AdamW with the fp32 master weights and moments kept in CPU memory.

For full fine-tuning on a single 32GB GPU (docs/pi05全量训练_单卡5090方案.md, plan B): the GPU only holds the bf16
weights, bf16 gradients and activations; the fp32 master copy, m and v (12 bytes/param, ~45GB for ~3.7B params) live
in host RAM and the update runs on the CPU. Same math as torch.optim.AdamW (decoupled weight decay, bias correction).
"""

from __future__ import annotations

import math

import torch


class CPUOffloadAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self._init_state(p)

    def _init_state(self, p: torch.Tensor) -> None:
        master = p.detach().to("cpu", dtype=torch.float32, copy=True)
        self.state[p] = {
            "step": 0,
            "master": master,
            "exp_avg": torch.zeros_like(master),
            "exp_avg_sq": torch.zeros_like(master),
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                st["step"] += 1
                t = st["step"]
                g = p.grad.detach().to("cpu").float()
                master, m, v = st["master"], st["exp_avg"], st["exp_avg_sq"]
                master.mul_(1 - lr * wd)
                m.lerp_(g, 1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                denom = v.sqrt().div_(math.sqrt(1 - b2**t)).add_(eps)
                master.addcdiv_(m, denom, value=-lr / (1 - b1**t))
                p.copy_(master.to(p.dtype), non_blocking=False)
        return loss

    def load_state_dict(self, state_dict):
        # The base class would move the state to the parameter's device/dtype; keep it on the CPU in fp32.
        params = [p for g in self.param_groups for p in g["params"]]
        saved_groups = state_dict["param_groups"]
        for group, saved in zip(self.param_groups, saved_groups, strict=True):
            group.update({k: v for k, v in saved.items() if k != "params"})
        for idx, st in state_dict["state"].items():
            p = params[idx]
            self.state[p] = {
                "step": int(st["step"]),
                **{k: st[k].to("cpu", dtype=torch.float32, copy=True) for k in ("master", "exp_avg", "exp_avg_sq")},
            }
