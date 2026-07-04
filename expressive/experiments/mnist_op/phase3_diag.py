"""
PHASE 3 DIAGNOSTIC — why did orbit-averaging not transfer to REAL NeSyDM on has-a-repeat?

Two hypotheses (see notes/PHASE3_NESYDM.md):
  H1 (machinery): vanilla NeSyDM's objective is FLAT on has-a-repeat (loss ~1.1 every epoch in
       the old runs) -> it never learns concepts, so there is no gradient direction for orbit-
       averaging to fix. Diagnose by logging the SUB-losses + reward mean + concept accuracy of
       the variational sampler per epoch.
  H2 (carry-over ESS collapse): the graft's own comment predicts orbit_ess ~ 1 because carried-over
       one-hot positions send log q(sigma.w) -> -inf for relabeled worlds, so softmax collapses to
       identity -> orbit-averaging degenerates to vanilla. Diagnose by logging orbit_ess per epoch.

This instruments the SAME model/problem/eval as orbit_nesydm.py but writes ALL diagnostics to disk
per epoch (hygiene) so the failure is inspectable without re-running.

Run (workstation, one GPU each):
  CUDA_VISIBLE_DEVICES=0 uv run python -m expressive.experiments.mnist_op.phase3_diag --N 4 --epochs 12 --orbit_M 0
  CUDA_VISIBLE_DEVICES=1 uv run python -m expressive.experiments.mnist_op.phase3_diag --N 4 --epochs 12 --orbit_M 16
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


# per-epoch accumulator with reset (the base _Log accumulates forever)
_FIELDS = ["w_denoise", "y_denoise", "var_entropy", "unmasking_entropy",
           "avg_constraints", "avg_var_violations", "var_accuracy_y", "var_accuracy_w"]


class _Log:
    def __init__(self):
        object.__setattr__(self, "_d", {})
        self.reset()

    def reset(self):
        for k in _FIELDS:
            self._d[k] = 0.0
        self._d["w_preds"] = np.array([], dtype=int)
        self._d["w_targets"] = np.array([], dtype=int)

    def __getattr__(self, k):
        return self._d.get(k, 0.0)

    def __setattr__(self, k, v):
        self._d[k] = v


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
    tag = f"phase3_n{n}_orbitM{orbit_M}"
    outdir = os.path.join("runs_orbit_nesydm", tag); os.makedirs(outdir, exist_ok=True)
    logpath = os.path.join(outdir, "train.log")
    # header: full diagnostic columns
    cols = ("epoch\tloss\tL_y\tL_w\tvar_ent\tunmask_ent\treward_mean\tvar_viol\t"
            "concept_acc_train\tvar_acc_y\torbit_ess\teval_concept_acc\teval_label_acc\t"
            "eval_label_ece\ttime_s")
    with open(logpath, "w") as f:
        f.write(cols + "\n")
    print(f"PHASE3 has-repeat n={n} orbit_M={orbit_M} device={device} epochs={args.epochs} out={outdir}")

    model = SimpleNeSyDiffusion(HasRepeatModel(n, args), HasRepeatProblem(n, "repeat"), args).to(device)
    model.orbit_M = orbit_M

    train_loader, val_loader, test_loader = get_mnist_op_dataloaders(
        count_train=int(50000 / n), count_val=int(10000 / n), count_test=int(10000 / n),
        batch_size=args.batch_size, n_operands=n, op=has_repeat_op, shuffle=True)

    log = _Log(); optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        t0 = time.time(); tot = 0.0; nb = 0
        log.reset()
        model.orbit_ess = float("nan")
        for batch in train_loader:
            optim.zero_grad()
            imgs, concepts, label = batch[:n], batch[n:2 * n], batch[-1]
            x = torch.cat(imgs, dim=1).to(device)
            w_labels = torch.stack(concepts, dim=1).to(device)
            label = label.to(device).long().reshape(-1, 1)
            loss = model.loss(x, label, log, w_labels)
            loss.backward(); optim.step(); tot += float(loss); nb += 1
        ess = getattr(model, "orbit_ess", float("nan"))
        # per-epoch means of the accumulated diagnostics
        d = {k: log._d[k] / nb for k in _FIELDS}
        # cheap eval every 2 epochs (and last)
        if epoch % 2 == 0 or epoch == args.epochs - 1:
            m = evaluate(model, val_loader, n, device, "repeat")
            ev = (m["concept_acc"], m["label_acc"], m["label_ece"])
        else:
            ev = (float("nan"), float("nan"), float("nan"))
        row = (f"{epoch}\t{tot/nb:.6f}\t{d['y_denoise']:.6f}\t{d['w_denoise']:.6f}\t"
               f"{d['var_entropy']:.6f}\t{d['unmasking_entropy']:.6f}\t{d['avg_constraints']:.6f}\t"
               f"{d['avg_var_violations']:.6f}\t{d['var_accuracy_w']:.6f}\t{d['var_accuracy_y']:.6f}\t"
               f"{ess:.3f}\t{ev[0]:.4f}\t{ev[1]:.4f}\t{ev[2]:.4f}\t{time.time()-t0:.1f}")
        with open(logpath, "a") as f:
            f.write(row + "\n")
        print(f"ep{epoch} loss {tot/nb:.4f} L_y {d['y_denoise']:.4f} L_w {d['w_denoise']:.4f} "
              f"reward {d['avg_constraints']:.3f} c_acc_tr {d['var_accuracy_w']:.3f} "
              f"ess {ess:.2f} eval_c_acc {ev[0]:.3f} eval_l_acc {ev[1]:.3f} ({time.time()-t0:.0f}s)")

    m = evaluate(model, test_loader, n, device, "repeat")
    save_preds(model, test_loader, n, device, os.path.join(outdir, "preds.npz"))
    with open(logpath, "a") as f:
        f.write(f"# TEST\tconcept_acc {m['concept_acc']:.4f}\tlabel_acc {m['label_acc']:.4f}\t"
                f"label_ece {m['label_ece']:.4f}\n")
    print(f"\nTEST n={n} orbit_M={orbit_M}: concept-acc {m['concept_acc']:.3f}  "
          f"label-acc {m['label_acc']:.3f}  label-ECE {m['label_ece']:.3f}")
    print(f"saved predictions+log under {outdir}")


if __name__ == "__main__":
    main()
