"""
REAL NeSyDM on has-a-repeat, with optional ORBIT-AVERAGING of the RLOO score (--orbit_M M).
  orbit_M=0  -> vanilla NeSyDM (baseline).
  orbit_M>0  -> orbit-aware NeSyDM: the score dlog q(w) is q-weighted-averaged over M value-
                relabelings sigma in S_10 (the label-symmetry group; graft in simple_nesy_diff.py).

Same model/problem/eval as has_repeat.py. Adds per-epoch training logs and per-instance
predictions (P(has-repeat) + label) to runs_orbit_nesydm/<tag>/ (hygiene rule), so bag-acc /
ECE etc. are recomputable and a failed run is inspectable.

Run: uv run python -m expressive.experiments.mnist_op.orbit_nesydm --N 4 --epochs 8 --orbit_M 16
"""

import os
import sys
import time

import numpy as np
import torch

from expressive.args import MNISTAbsorbingArguments
from expressive.experiments.mnist_op.data import get_mnist_op_dataloaders
from expressive.experiments.mnist_op.has_repeat import (
    HasRepeatModel, HasRepeatProblem, has_repeat_op, evaluate, p_all_distinct)
from expressive.methods.simple_nesy_diff import SimpleNeSyDiffusion
from expressive.util import get_device


def _get_int(flag, default):
    for i, a in enumerate(sys.argv):
        if a == flag:
            return int(sys.argv[i + 1])
        if a.startswith(flag + "="):
            return int(a.split("=", 1)[1])
    return default


class _Log:
    def __init__(self):
        object.__setattr__(self, "_d", {"w_preds": np.array([], dtype=int),
                                        "w_targets": np.array([], dtype=int)})
    def __getattr__(self, k): return self._d.get(k, 0.0)
    def __setattr__(self, k, v): self._d[k] = v


@torch.no_grad()
def save_preds(model, loader, n, device, path):
    P, Y = [], []
    for batch in loader:
        imgs = batch[:n]; label = batch[-1]
        x = torch.cat(imgs, dim=1).to(device); B = x.shape[0]
        enc = model.p.encode_x(x)
        masked = torch.full((B, n), 10, device=device)
        p = model.p.distribution(masked, enc, torch.zeros(B, device=device))[..., :10]
        P.append((1 - p_all_distinct(p)).cpu()); Y.append(label.cpu())
    np.savez_compressed(path, p_hasrep=torch.cat(P).numpy().astype(np.float32),
                        label=torch.cat(Y).numpy().astype(np.int64))


def main():
    orbit_M = _get_int("--orbit_M", 0)
    args = MNISTAbsorbingArguments(explicit_bool=True).parse_args(known_only=True)
    n = args.N; device = get_device(args)
    tag = f"n{n}_orbitM{orbit_M}"
    outdir = os.path.join("runs_orbit_nesydm", tag); os.makedirs(outdir, exist_ok=True)
    logpath = os.path.join(outdir, "train.log"); open(logpath, "w").close()
    print(f"NeSyDM has-repeat n={n} orbit_M={orbit_M} device={device} epochs={args.epochs} out={outdir}")

    model = SimpleNeSyDiffusion(HasRepeatModel(n, args), HasRepeatProblem(n, "repeat"), args).to(device)
    model.orbit_M = orbit_M                                        # <- activates the graft

    train_loader, val_loader, test_loader = get_mnist_op_dataloaders(
        count_train=int(50000 / n), count_val=int(10000 / n), count_test=int(10000 / n),
        batch_size=args.batch_size, n_operands=n, op=has_repeat_op, shuffle=True)

    log = _Log(); optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        t0 = time.time(); tot = 0.0; nb = 0
        for batch in train_loader:
            optim.zero_grad()
            imgs, concepts, label = batch[:n], batch[n:2 * n], batch[-1]
            x = torch.cat(imgs, dim=1).to(device)
            w_labels = torch.stack(concepts, dim=1).to(device)
            label = label.to(device).long().reshape(-1, 1)
            loss = model.loss(x, label, log, w_labels)
            loss.backward(); optim.step(); tot += float(loss); nb += 1
        ess = getattr(model, "orbit_ess", float("nan"))               # diagnostic (see graft)
        with open(logpath, "a") as f:
            f.write(f"epoch {epoch}\tloss {tot / nb:.6f}\torbit_ess {ess:.3f}\ttime_s {time.time() - t0:.1f}\n")
        print(f"epoch {epoch} loss {tot / nb:.4f} orbit_ess {ess:.3f} (of M={orbit_M}) time {time.time() - t0:.1f}s")

    m = evaluate(model, test_loader, n, device, "repeat")
    save_preds(model, test_loader, n, device, os.path.join(outdir, "preds.npz"))
    print(f"\nNeSyDM has-repeat n={n} orbit_M={orbit_M}: concept-acc {m['concept_acc']:.3f}  "
          f"label-acc {m['label_acc']:.3f}  label-ECE {m['label_ece']:.3f}")
    print(f"saved predictions+log under {outdir}")


if __name__ == "__main__":
    main()
