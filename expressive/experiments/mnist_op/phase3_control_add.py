"""
PHASE 3 CONTROL — vanilla NeSyDM on a NON-symmetric task (MNIST addition, op=sum), same machinery.

Purpose: isolate the has-a-repeat failure to the SYMMETRIC reward. Phase3_diag showed vanilla
NeSyDM stays at chance concept-acc on has-a-repeat (a reasoning shortcut). If the SAME NeSyDM
config LEARNS concepts on addition (a non-symmetric reward, no shortcut), that pins the collapse on
the reward symmetry, not a config bug -> the cleanest control for C1.

Mirrors mnistop.py's train/eval but writes concept-acc (w_acc_avg) + label-acc per epoch to a FILE
(the harness only logs to wandb; hygiene needs it on disk).

Run (workstation):
  CUDA_VISIBLE_DEVICES=0 uv run python -m expressive.experiments.mnist_op.phase3_control_add --op sum --N 1 --epochs 12 --use_wandb False
  CUDA_VISIBLE_DEVICES=1 uv run python -m expressive.experiments.mnist_op.phase3_control_add --op sum --N 2 --epochs 12 --use_wandb False
"""

import math
import os
import time

import torch

from expressive.util import get_device
from expressive.args import MNISTAbsorbingArguments
from expressive.experiments.mnist_op.absorbing_mnist import (
    create_mnistadd, vector_to_base10)
from expressive.experiments.mnist_op.data import (
    create_nary_multidigit_operation, get_mnist_op_dataloaders)
from expressive.methods.logger import TestLog

args = MNISTAbsorbingArguments(explicit_bool=True).parse_args()


@torch.no_grad()
def evaluate_concept(model, loader, device, N):
    log = TestLog(args, "val")
    nb = 0
    for batch in loader:
        mn_digits, label_digits, label = batch[:2 * N], batch[2 * N:-1], batch[-1]
        x = torch.cat(mn_digits, dim=1).to(device)
        model.evaluate(
            x, vector_to_base10(label.to(device), N + 1),
            torch.stack(label_digits, dim=-1).to(device), log)
        nb += 1
    d = log.create_dict(nb)
    # keys are prefixed with "val/"
    return {k.split("/", 1)[1]: v for k, v in d.items()}


def main():
    device = get_device(args)
    N = args.N
    tag = f"phase3_ctrl_add_{args.op}_N{N}"
    outdir = os.path.join("runs_orbit_nesydm", tag); os.makedirs(outdir, exist_ok=True)
    logpath = os.path.join(outdir, "train.log")
    with open(logpath, "w") as f:
        f.write("epoch\ttrain_loss\tval_w_acc_avg\tval_w_acc_top\tval_y_acc_avg\ttime_s\n")
    print(f"CONTROL addition op={args.op} N={N} device={device} epochs={args.epochs} out={outdir}")

    model = create_mnistadd(args).to(device)
    arity = 2
    n_operands = arity * N
    bin_op = sum if args.op == "sum" else math.prod if args.op == "product" else None
    op = create_nary_multidigit_operation(arity, bin_op)

    train_loader, val_loader, test_loader = get_mnist_op_dataloaders(
        count_train=int(50000 / n_operands), count_val=int(10000 / n_operands),
        count_test=int(10000 / n_operands), batch_size=args.batch_size,
        n_operands=n_operands, op=op, shuffle=True)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # a tiny throwaway logger object for model.loss (accumulates; we only read train loss)
    from expressive.methods.logger import TrainingLog
    for epoch in range(args.epochs):
        t0 = time.time(); tot = 0.0; nb = 0
        tlog = TrainingLog(args)
        model.train()
        for batch in train_loader:
            optim.zero_grad()
            mn_digits, label, w_labels = batch[:2 * N], batch[-1], batch[2 * N:-1]
            x = torch.cat(mn_digits, dim=1).to(device)
            w_labels = torch.stack(w_labels, dim=1).to(device)
            label = vector_to_base10(label.to(device), N + 1)
            loss = model.loss(x, label, tlog, w_labels)
            loss.backward(); optim.step(); tot += float(loss); nb += 1
        m = evaluate_concept(model, val_loader, device, N)
        row = (f"{epoch}\t{tot/nb:.6f}\t{m.get('w_acc_avg', float('nan')):.4f}\t"
               f"{m.get('w_acc_top', float('nan')):.4f}\t{m.get('y_acc_avg', float('nan')):.4f}\t"
               f"{time.time()-t0:.1f}")
        with open(logpath, "a") as f:
            f.write(row + "\n")
        print(f"ep{epoch} loss {tot/nb:.4f} val_concept_acc {m.get('w_acc_avg', float('nan')):.3f} "
              f"val_label_acc {m.get('y_acc_avg', float('nan')):.3f} ({time.time()-t0:.0f}s)")

    m = evaluate_concept(model, test_loader, device, N)
    with open(logpath, "a") as f:
        f.write(f"# TEST\tw_acc_avg {m.get('w_acc_avg', float('nan')):.4f}\t"
                f"y_acc_avg {m.get('y_acc_avg', float('nan')):.4f}\n")
    print(f"\nTEST addition op={args.op} N={N}: concept-acc {m.get('w_acc_avg', float('nan')):.3f} "
          f"label-acc {m.get('y_acc_avg', float('nan')):.3f}")
    print(f"saved log under {outdir}")


if __name__ == "__main__":
    main()
