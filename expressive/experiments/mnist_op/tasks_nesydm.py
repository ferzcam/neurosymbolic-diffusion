"""
Real NeSyDM (masked-diffusion + RLOO score estimator, expressive/methods/simple_nesy_diff.py)
run as a BASELINE on our three symmetric NeSy tasks, so it slots alongside orbit / rloo / indecater:

  repeat : has-a-repeat over n=4 MNIST images, 10-value support (S_10 symmetry). y=1 iff two match.
           Uses the repo's HasRepeatProblem (phi) directly.
  conn   : graph connectivity. 2*M_EDGES=14 MNIST images -> M_EDGES edges over V=6 nodes.
           y=1 iff connected. phi ported EXACTLY from method/graph_train.py (`connected`).
  col3   : graph 3-colorability. Same shape_w. y=1 iff a proper 3-coloring exists.
           phi ported EXACTLY from method/graph_train.py (`is_3colorable`).

Perception is OUR CNN (method/groupavg_rloo.py:CNN), wrapped as a NeSyDM UnmaskingModel, so all
methods share the exact same encoder (fair comparison). NeSyDM diffusion internals (timesteps,
loss-term weights, beta, entropy coef, simple_model) stay at MNISTAbsorbingArguments DEFAULTS.
Only lr and K (world samples -> loss_S = variational_K = test_K) are gridded.

Balanced data generation is ported from method/graph_train.py (gen_conn_sets / gen_color_sets) and
method/groupavg_rloo.py (make_sets_balanced) so the label distribution matches our other runs.

Hygiene: per-epoch training log (loss, val/test label+concept acc) to <outdir>/<tag>.log; a JSON
line with best-val / test-at-best-val / converged to <outdir>/<tag>.json. Metric = label accuracy;
convergence = test-at-best-val >= --conv_thresh (default 0.9), reported as #converged over seeds.

Run: uv run python -m expressive.experiments.mnist_op.tasks_nesydm --task conn --K 16 --lr 1e-3 \
     --seed 0 --epochs 60 --batch_size 32 --n_sets 8000 --patience 15
"""
import argparse, json, os, random, re, time, itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from expressive.args import MNISTAbsorbingArguments
from expressive.methods.base_model import Problem
from expressive.methods.simple_nesy_diff import SimpleNeSyDiffusion
from expressive.models.diffusion_model import UnmaskingModel
from expressive.experiments.mnist_op.has_repeat import HasRepeatProblem

MNIST_ROOT = os.environ.get("MNIST_ROOT", os.path.expanduser("~/mnist_data"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"

V_GRAPH, M_EDGES = 6, 7                                   # graph tasks: nodes, edges (n = 2*M_EDGES)
V_REPEAT, N_REPEAT = 10, 4                                # has-a-repeat: full 10-value support, n=4


# ----------------------------------------------------------------------------------------------
# EXACT phi ported from method/graph_train.py  (functions `connected`, `is_3colorable`)
# ----------------------------------------------------------------------------------------------
def connected(w):                                        # w[...,2m] node-ids in {0..V-1} -> bool[...]
    lead = w.shape[:-1]
    e = w.reshape(-1, M_EDGES, 2)                         # [N,m,2]
    N = e.shape[0]
    A = torch.zeros(N, V_GRAPH, V_GRAPH, device=w.device)
    idx = torch.arange(N, device=w.device).unsqueeze(1)  # [N,1]
    u, v = e[..., 0], e[..., 1]                           # [N,m]
    A[idx, u, v] = 1.0; A[idx, v, u] = 1.0
    A = A + torch.eye(V_GRAPH, device=w.device)          # self-loops
    R = (A > 0).float()
    for _ in range(int(np.ceil(np.log2(V_GRAPH))) + 1):  # transitive closure by repeated squaring
        R = (R @ R > 0).float()
    conn = (R.sum(-1) == V_GRAPH).all(-1)                # every node reaches all V
    return conn.reshape(lead)


_COLORINGS = None
def _colorings(dev):                                     # all 3^V colorings [3^V, V]
    global _COLORINGS
    if _COLORINGS is None:
        _COLORINGS = torch.tensor(list(itertools.product(range(3), repeat=V_GRAPH)))
    return _COLORINGS.to(dev)


def is_3colorable(w):                                    # w[...,2m] -> bool[...] (NP-hard; exact via enum)
    lead = w.shape[:-1]
    e = w.reshape(-1, M_EDGES, 2); N = e.shape[0]
    A = torch.zeros(N, V_GRAPH, V_GRAPH, device=w.device)
    idx = torch.arange(N, device=w.device).unsqueeze(1)
    A[idx, e[..., 0], e[..., 1]] = 1.0; A[idx, e[..., 1], e[..., 0]] = 1.0   # no self-loops added
    C = _colorings(w.device)                             # [nc, V]
    same = (C.unsqueeze(2) == C.unsqueeze(1)).float()    # [nc, V, V]
    mono = torch.einsum('nij,cij->nc', A, same)          # [N, nc] monochromatic-edge count
    return (mono == 0).any(1).reshape(lead)              # exists a proper 3-coloring


# ----------------------------------------------------------------------------------------------
# EXACT balanced data generation ported from method/graph_train.py and method/groupavg_rloo.py
# ----------------------------------------------------------------------------------------------
def gen_color_sets(ypool, n_sets, seed):
    """Balanced 3-colorable / non-3-colorable graphs; y=0 embeds a K4 (needs 4 colors). m>=6 required."""
    rng = random.Random(seed)
    by_digit = [(ypool == d).nonzero().flatten() for d in range(V_GRAPH)]
    n = 2 * M_EDGES
    idx = torch.zeros(n_sets, n, dtype=torch.long); lab = torch.zeros(n_sets, dtype=torch.long)
    for j in range(n_sets):
        if rng.random() < 0.5:                           # 3-COLORABLE: edges only between diff-colored nodes
            col = [i % 3 for i in range(V_GRAPH)]; rng.shuffle(col)
            cross = [(a, b) for a in range(V_GRAPH) for b in range(a + 1, V_GRAPH) if col[a] != col[b]]
            edges = rng.sample(cross, M_EDGES); lab[j] = 1
        else:                                            # NON-3-COLORABLE: embed a K4 + extra edges
            q = rng.sample(range(V_GRAPH), 4)
            edges = [(q[a], q[b]) for a in range(4) for b in range(a + 1, 4)]     # K4 = 6 edges
            while len(edges) < M_EDGES:
                a, b = rng.randrange(V_GRAPH), rng.randrange(V_GRAPH)
                if a != b: edges.append((a, b))
            lab[j] = 0
        flat = [x for e in edges for x in e]
        for c, d in enumerate(flat):
            pool = by_digit[d]; idx[j, c] = pool[rng.randrange(len(pool))]
    return idx, lab


def gen_conn_sets(ypool, n_sets, seed):
    """Balanced connected/disconnected graphs; concepts are MNIST image indices for the endpoint digits."""
    rng = random.Random(seed)
    by_digit = [(ypool == d).nonzero().flatten() for d in range(V_GRAPH)]
    n = 2 * M_EDGES
    idx = torch.zeros(n_sets, n, dtype=torch.long); lab = torch.zeros(n_sets, dtype=torch.long)
    for j in range(n_sets):
        if rng.random() < 0.5:                           # CONNECTED: spanning tree + extra edges
            perm = list(range(V_GRAPH)); rng.shuffle(perm)
            edges = [(perm[i], perm[rng.randrange(i)]) for i in range(1, V_GRAPH)]     # random tree
            while len(edges) < M_EDGES:
                a, b = rng.randrange(V_GRAPH), rng.randrange(V_GRAPH)
                if a != b: edges.append((a, b))
            lab[j] = 1
        else:                                            # DISCONNECTED: isolate a nonempty subset
            k = rng.randint(1, V_GRAPH - 1); part = set(rng.sample(range(V_GRAPH), k))
            edges = []
            while len(edges) < M_EDGES:
                a, b = rng.randrange(V_GRAPH), rng.randrange(V_GRAPH)
                if a != b and ((a in part) == (b in part)): edges.append((a, b))   # only within a side
            lab[j] = 0
        flat = [x for e in edges for x in e]
        for c, d in enumerate(flat):
            pool = by_digit[d]; idx[j, c] = pool[rng.randrange(len(pool))]
    return idx, lab


def make_repeat_sets(y, n_sets, seed):
    """~50/50 has-repeat vs all-distinct over 10 digit values, n=4 (ported from groupavg_rloo.make_sets_balanced)."""
    g = random.Random(seed); V, n = V_REPEAT, N_REPEAT
    by_digit = [(y == d).nonzero().flatten() for d in range(V)]
    idx = torch.zeros(n_sets, n, dtype=torch.long); lab = torch.zeros(n_sets, dtype=torch.long)
    for j in range(n_sets):
        if g.random() < 0.5:                                  # all-distinct
            digs = g.sample(range(V), n); lab[j] = 0
        else:                                                 # forced repeat
            d = g.randrange(V)
            digs = [d, d] + [g.randrange(V) for _ in range(n - 2)]; g.shuffle(digs); lab[j] = 1
        for k, dd in enumerate(digs):
            pool = by_digit[dd]; idx[j, k] = pool[g.randrange(len(pool))]
    return idx, lab


# ----------------------------------------------------------------------------------------------
# OUR CNN (method/groupavg_rloo.py:CNN), wrapped as a NeSyDM UnmaskingModel
# ----------------------------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1); self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64); self.head = nn.Linear(64, V)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2); x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.head(F.relu(self.fc1(x.flatten(1))))


class CNNUnmaskingModel(UnmaskingModel):
    """Independent per-item concept denoiser using OUR CNN (mirrors HasRepeatModel: logits ignore the
    diffusion state, so it is a simple NeSy predictor). encode_x returns per-item V-way logits."""
    def __init__(self, n: int, V: int, args):
        super().__init__(vocab_dim=V, w_dims=n, seq_length=n + 1, args=args)
        self.cnn = CNN(V); self.n = n; self.V = V

    def encode_x(self, x):                               # x [B, n, 28, 28] -> logits [B, n, V]
        B, n = x.shape[:2]
        return self.cnn(x.reshape(B * n, 1, 28, 28)).reshape(B, n, self.V)

    def logits_t0(self, wy_t, x_encoding, t):            # independent of (wy_t, t): carry-over handled upstream
        return x_encoding


# ----------------------------------------------------------------------------------------------
# Problems
# ----------------------------------------------------------------------------------------------
class GraphProblem(Problem):
    """conn (connectivity) or col3 (3-colorability): 14 node-id concepts in {0..5}, binary y."""
    def __init__(self, task):
        self.task = task; self.phi = connected if task == "conn" else is_3colorable

    def shape_w(self):
        return (2 * M_EDGES, V_GRAPH)                     # (14, 6)

    def shape_y(self):
        return (1, 2)                                    # binary

    def y_from_w(self, w):                               # w[...,14] -> y[...,1] in {0,1}
        return self.phi(w).long().unsqueeze(-1)


TASKS = {
    "repeat": dict(V=V_REPEAT, n=N_REPEAT, gen=make_repeat_sets, problem=lambda: HasRepeatProblem(N_REPEAT, "repeat")),
    "conn":   dict(V=V_GRAPH, n=2 * M_EDGES, gen=gen_conn_sets, problem=lambda: GraphProblem("conn")),
    "col3":   dict(V=V_GRAPH, n=2 * M_EDGES, gen=gen_color_sets, problem=lambda: GraphProblem("col3")),
}


class _Log:                                              # minimal stand-in for TrainingLog (from has_repeat.py)
    def __init__(self):
        object.__setattr__(self, "_d", {"w_preds": np.array([], dtype=int), "w_targets": np.array([], dtype=int)})
    def __getattr__(self, k): return self._d.get(k, 0.0)
    def __setattr__(self, k, v): self._d[k] = v


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load(train):
    from torchvision import datasets
    ds = datasets.MNIST(MNIST_ROOT, train=train, download=True)
    x = ((ds.data.float() / 255.0 - 0.1307) / 0.3081).unsqueeze(1)   # [N,1,28,28]
    return x, ds.targets.long()


@torch.no_grad()
def evaluate(model, imgs, ylab, idx, lab, problem, V, n, bs=256):
    """Concept-marginal readout (mirrors has_repeat.evaluate): concept argmax then phi.
    label-acc = phi(argmax concept) vs lab ; concept-acc = argmax vs true digit ids."""
    model.eval(); lc = cc = tot = 0
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        x = imgs[b].squeeze(2).to(DEV)                   # [B,n,28,28]
        gt = ylab[b].to(DEV)                             # [B,n] true node-ids
        B = x.shape[0]
        enc = model.p.encode_x(x)
        masked = torch.full((B, n), V, device=DEV)       # mask value = vocab_dim = V
        p = model.p.distribution(masked, enc, torch.zeros(B, device=DEV))[..., :V]   # [B,n,V]
        w_hat = p.argmax(-1)                             # [B,n]
        pred = problem.y_from_w(w_hat).squeeze(-1).long().cpu()
        lc += (pred == lab[i:i + bs]).sum().item()
        cc += (w_hat == gt).float().sum().item(); tot += B
    model.train()
    return lc / len(idx), cc / (tot * n)


@torch.no_grad()
def collect_preds(model, imgs, ylab, idx, lab, problem, V, n, bs=256):
    """Dump test-set predictions for concept_acc.py: concept marginal probs (same readout as
    evaluate), predicted label (phi of argmax), true label, true concept ids."""
    model.eval(); cps, pls, tcs = [], [], []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        x = imgs[b].squeeze(2).to(DEV)                   # [B,n,28,28]
        B = x.shape[0]
        enc = model.p.encode_x(x)
        masked = torch.full((B, n), V, device=DEV)
        p = model.p.distribution(masked, enc, torch.zeros(B, device=DEV))[..., :V]   # [B,n,V]
        pred = problem.y_from_w(p.argmax(-1)).squeeze(-1).long().cpu()
        cps.append(p.cpu().numpy().astype(np.float32))
        pls.append(pred.numpy()); tcs.append(ylab[b].numpy())
    model.train()
    return (np.concatenate(cps), np.concatenate(pls),
            lab.numpy().astype(np.int64), np.concatenate(tcs).astype(np.int64))


def run_one(task, K, lr, seed, epochs, n_sets, patience, batch_size, outdir, conv_thresh,
            orbit_M=0, orbit_weights="uniform"):
    set_seed(seed)
    cfg = TASKS[task]; V, n = cfg["V"], cfg["n"]

    # NeSyDM args: defaults for all diffusion internals; only lr and world-sample count K overridden.
    args = MNISTAbsorbingArguments(explicit_bool=True).parse_args(
        ["--use_wandb", "False", "--lr", str(lr), "--batch_size", str(batch_size)], known_only=True)
    args.loss_S = K; args.variational_K = K; args.test_K = K   # our K == NeSyDM world samples

    xtr, ytr = load(True); xte, yte = load(False)
    xval, yval = xtr[50000:], ytr[50000:]; xtr, ytr = xtr[:50000], ytr[:50000]
    tr_idx, tr_lab = cfg["gen"](ytr, n_sets, seed)
    va_idx, va_lab = cfg["gen"](yval, 4000, seed + 777)
    te_idx, te_lab = cfg["gen"](yte, 4000, 424242)

    problem = cfg["problem"]()
    model = SimpleNeSyDiffusion(CNNUnmaskingModel(n, V, args), problem, args).to(DEV)
    model.orbit_M = orbit_M                                    # 0 = vanilla NeSyDM; >0 activates OrbitA graft
    model.orbit_weights = orbit_weights                       # "uniform" (ESS=M) or "softmax" (H2 risk)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log = _Log()

    otag = "" if orbit_M == 0 else f"_M{orbit_M}{orbit_weights[0]}"
    tag = f"{task}_lr{lr:g}_K{K}_M{orbit_M}{orbit_weights[0] if orbit_M else ''}_seed{seed}"
    logf = os.path.join(outdir, tag + ".log")
    with open(logf, "w") as f:
        f.write(f"# task={task} K={K} lr={lr} seed={seed} n={n} V={V} n_sets={n_sets} "
                f"batch_size={batch_size} loss_S={K} variational_K={K} epochs={epochs} patience={patience} "
                f"orbit_M={orbit_M} orbit_weights={orbit_weights}\n")

    best_val, best_test, best_ep, since, best_ess = -1.0, 0.0, -1, 0, float("nan")
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n_sets); t0 = time.time()
        ess_sum, ess_n = 0.0, 0
        for i in range(0, n_sets, batch_size):
            b = tr_idx[perm[i:i + batch_size]]
            x = xtr[b].squeeze(2).to(DEV)                # [B,n,28,28]
            w_labels = ytr[b].to(DEV)                    # [B,n] concept ids (for logging)
            label = tr_lab[perm[i:i + batch_size]].to(DEV).long().reshape(-1, 1)   # [B,1]
            loss = model.loss(x, label, log, w_labels)
            opt.zero_grad(); loss.backward(); opt.step()
            if orbit_M > 0:
                ess_sum += getattr(model, "orbit_ess", float("nan")); ess_n += 1
        ess = ess_sum / ess_n if ess_n else float("nan")   # mean effective # relabelings this epoch
        va, vca = evaluate(model, xval, yval, va_idx, va_lab, problem, V, n)
        te, tca = evaluate(model, xte, yte, te_idx, te_lab, problem, V, n)
        with open(logf, "a") as f:
            f.write(f"epoch {ep}\tloss {float(loss):.4f}\tval_acc {va:.4f}\ttest_acc {te:.4f}"
                    f"\tval_cacc {vca:.4f}\ttest_cacc {tca:.4f}\torbit_ess {ess:.3f}\ttime {time.time()-t0:.1f}\n")
        print(f"[{tag}] ep{ep} loss {float(loss):.3f} val {va:.3f} test {te:.3f} cacc {tca:.3f} ess {ess:.2f}", flush=True)
        if va > best_val + 1e-4:
            best_ess = ess
            best_val, best_test, best_ep, since = va, te, ep, 0
            # Save test-set predictions AT best-val (overwrite) for concept_acc.py (up-to-relabeling).
            cp, pl, tl, tc = collect_preds(model, xte, yte, te_idx, te_lab, problem, V, n)
            np.savez(os.path.join(outdir, tag + "_preds.npz"),
                     concept_probs=cp, pred_label=pl, true_label=tl, true_concepts=tc)
        else:
            since += 1
        if since >= patience:
            break

    res = dict(task=task, K=K, lr=lr, seed=seed, best_val=best_val, test_at_bestval=best_test,
               best_epoch=best_ep, converged=int(best_test >= conv_thresh), conv_thresh=conv_thresh,
               n=n, V=V, n_sets=n_sets, batch_size=batch_size, loss_S=K, variational_K=K,
               orbit_M=orbit_M, orbit_weights=orbit_weights, orbit_ess_at_bestval=best_ess)
    with open(os.path.join(outdir, tag + ".json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"== RESULT {tag}: test@bestval {best_test:.4f} (val {best_val:.4f}, ep {best_ep}) "
          f"converged={res['converged']} ==", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="conn", choices=["repeat", "conn", "col3"])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n_sets", type=int, default=8000)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--outdir", default="runs_tasks_nesydm")
    ap.add_argument("--conv_thresh", type=float, default=0.9)
    ap.add_argument("--orbit_M", type=int, default=0, help="0=vanilla NeSyDM; >0 activates OrbitA graft")
    ap.add_argument("--orbit_weights", default="uniform", choices=["uniform", "softmax"])
    a, _ = ap.parse_known_args()
    os.makedirs(a.outdir, exist_ok=True)
    print(f"NeSyDM task={a.task} K={a.K} lr={a.lr} seed={a.seed} orbit_M={a.orbit_M} "
          f"orbit_weights={a.orbit_weights} device={DEV} MNIST_ROOT={MNIST_ROOT}", flush=True)
    run_one(a.task, a.K, a.lr, a.seed, a.epochs, a.n_sets, a.patience, a.batch_size, a.outdir,
            a.conv_thresh, a.orbit_M, a.orbit_weights)


if __name__ == "__main__":
    main()
