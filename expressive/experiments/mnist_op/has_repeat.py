"""
Real NeSyDM on a NON-decomposable, symmetric task: "has a repeat" over n MNIST images.
y = 1 if any two of the n digits are equal, else 0. This constraint couples all n objects
(NeSyDM must SAMPLE it -- no decomposable shortcut) and is invariant under permuting the
objects (S_n symmetry). We compare this real NeSyDM against our lifted-exact WMC elsewhere.

Run: uv run python -m expressive.experiments.mnist_op.has_repeat --n 3 --epochs 5
"""

import math
import time

import numpy as np
import torch
from torch import Tensor
from torch.nn import Linear
from tap import Tap

from expressive.args import MNISTAbsorbingArguments
from expressive.experiments.mnist_op.data import get_mnist_op_dataloaders
from expressive.experiments.mnist_op.models import MNISTEncoder
from expressive.methods.base_model import Problem
from expressive.methods.simple_nesy_diff import SimpleNeSyDiffusion
from expressive.models.diffusion_model import WY_DATA, UnmaskingModel
from expressive.models.dit import hidden_size
from expressive.util import get_device


def has_repeat_op(labels):
    return int(len(set(int(l) for l in labels)) < len(labels))


# subset-DP tables for exact P(all distinct) over 10 values (shared, fair readout)
_TGT = [[] for _ in range(10)]; _SRC = [[] for _ in range(10)]
for _S in range(1 << 10):
    for _v in range(10):
        if _S & (1 << _v):
            _TGT[_v].append(_S); _SRC[_v].append(_S ^ (1 << _v))


def p_all_distinct(p):                                   # p:[B,n,10] -> P(all distinct) [B]
    B, n, _ = p.shape
    dp = p.new_zeros(B, 1 << 10); dp[:, 0] = 1.0
    for t in range(n):
        nd = p.new_zeros(B, 1 << 10); pt = p[:, t]
        for v in range(10):
            tgt = torch.tensor(_TGT[v], device=p.device); src = torch.tensor(_SRC[v], device=p.device)
            nd[:, tgt] += dp[:, src] * pt[:, v:v + 1]
        dp = nd
    return dp.sum(1)


def ece(conf, correct, n_bins=15):
    edges = torch.linspace(0, 1, n_bins + 1); e = 0.0
    for b in range(n_bins):
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if m.any():
            e += m.float().mean() * (conf[m].mean() - correct[m].float().mean()).abs()
    return float(e)


def sum_dist(p):                                              # p:[B,n,10] -> [B,9n+1] exact
    B, n, _ = p.shape
    d = p[:, 0]
    for i in range(1, n):
        L = d.shape[1]; nd = p.new_zeros(B, L + 9)
        for v in range(10):
            nd[:, v:v + L] = nd[:, v:v + L] + d * p[:, i, v:v + 1]
        d = nd
    return d


@torch.no_grad()
def evaluate(model, loader, n, device, task="repeat"):
    """Fair readout from the denoiser concept marginal. repeat: exact P(repeat) via subset-DP;
    sum: exact sum distribution via convolution. Both report label-acc AND ECE."""
    dacc, conf_all, corr_all = [], [], []
    for batch in loader:
        imgs, concepts, label = batch[:n], batch[n:2 * n], batch[-1]
        x = torch.cat(imgs, dim=1).to(device)
        gt = torch.stack(concepts, dim=1).to(device)          # [B,n]
        y = label.to(device).long()
        B = x.shape[0]
        enc = model.p.encode_x(x)
        masked = torch.full((B, n), 10, device=device)
        p = model.p.distribution(masked, enc, torch.zeros(B, device=device))[..., :10]
        dacc.append((p.argmax(-1) == gt).float().mean().item())
        if task == "sum":
            sd = sum_dist(p); pred = sd.argmax(-1)
            conf = sd.max(-1).values; corr = (pred == y)
        else:
            prep = (1 - p_all_distinct(p)).clamp(0, 1)
            conf = torch.where(prep > 0.5, prep, 1 - prep); corr = (prep > 0.5).long() == y
        conf_all.append(conf.cpu()); corr_all.append(corr.cpu())
    conf = torch.cat(conf_all); corr = torch.cat(corr_all)
    return dict(concept_acc=float(np.mean(dacc)), label_acc=float(corr.float().mean()),
                label_ece=ece(conf, corr))


class HasRepeatModel(UnmaskingModel):
    def __init__(self, n: int, args) -> None:
        super().__init__(vocab_dim=10, w_dims=n, seq_length=n + 1, args=args)
        self.encoder = MNISTEncoder(hidden_size(args.model))
        self.output_layer = Linear(hidden_size(args.model), 10)
        self.n = n

    def encode_x(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def logits_t0(self, wy_t: WY_DATA, x_encoding: Tensor, t: Tensor) -> Tensor:
        return self.output_layer(torch.relu(x_encoding))


class HasRepeatProblem(Problem):
    """task='repeat' (non-decomposable) or 'sum' (decomposable control)."""
    def __init__(self, n: int, task: str = "repeat"):
        self.n = n; self.task = task

    def shape_w(self) -> torch.Size:
        return (self.n, 10)

    def shape_y(self) -> torch.Size:
        return (1, 2) if self.task == "repeat" else (1, 9 * self.n + 1)

    def y_from_w(self, w_SKBn: torch.Tensor) -> torch.Tensor:
        assert (w_SKBn < 10).all()
        if self.task == "sum":
            return w_SKBn.sum(-1, keepdim=True)
        eq = (w_SKBn.unsqueeze(-1) == w_SKBn.unsqueeze(-2)).sum(-1).sum(-1)  # n + 2*#pairs
        has = (eq > self.n).long()
        return has.unsqueeze(-1)


def main():
    import sys
    task = "sum" if "--task=sum" in sys.argv or ("--task" in sys.argv and "sum" in sys.argv) else "repeat"
    args = MNISTAbsorbingArguments(explicit_bool=True).parse_args(known_only=True)
    n = args.N                                   # reuse --N as the SET SIZE
    device = get_device(args)
    print(f"NeSyDM task={task}  n={n}  device={device}  epochs={args.epochs}")

    model = SimpleNeSyDiffusion(HasRepeatModel(n, args), HasRepeatProblem(n, task), args).to(device)

    op = (lambda labels: int(sum(int(l) for l in labels))) if task == "sum" else has_repeat_op
    train_loader, val_loader, test_loader = get_mnist_op_dataloaders(
        count_train=int(50000 / n), count_val=int(10000 / n), count_test=int(10000 / n),
        batch_size=args.batch_size, n_operands=n, op=op, shuffle=True,
    )

    class _Log:                                  # minimal stand-in for TrainingLog
        def __init__(self):
            object.__setattr__(self, "_d", {"w_preds": np.array([], dtype=int),
                                            "w_targets": np.array([], dtype=int)})
        def __getattr__(self, k): return self._d.get(k, 0.0)
        def __setattr__(self, k, v): self._d[k] = v
    log = _Log()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        t0 = time.time()
        for batch in train_loader:
            optim.zero_grad()
            imgs, concepts, label = batch[:n], batch[n:2 * n], batch[-1]
            x = torch.cat(imgs, dim=1).to(device)              # [B, n, 28, 28]
            w_labels = torch.stack(concepts, dim=1).to(device) # [B, n]
            label = label.to(device).long().reshape(-1, 1)     # [B, 1] in {0,1}
            loss = model.loss(x, label, log, w_labels)
            loss.backward(); optim.step()
        print(f"epoch {epoch} loss {float(loss):.4f} time {time.time()-t0:.1f}s")
    m = evaluate(model, test_loader, n, device, task)
    print(f"NeSyDM task={task} n={n}: concept-acc {m['concept_acc']:.3f}  label-acc {m['label_acc']:.3f}  "
          f"label-ECE {m['label_ece']:.3f}")


if __name__ == "__main__":
    main()
