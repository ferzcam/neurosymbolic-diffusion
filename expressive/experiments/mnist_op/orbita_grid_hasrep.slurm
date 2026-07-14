#!/bin/bash
#SBATCH --job-name=orbita_hasrep
#SBATCH --array=0-191%16
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/ibex/user/zhapacfp/neurosymbolic-diffusion/slurm_logs/orbita_hasrep_%A_%a.out
#SBATCH --error=/ibex/user/zhapacfp/neurosymbolic-diffusion/slurm_logs/orbita_hasrep_%A_%a.out

# OrbitA (orbit-averaged score) has-a-repeat grid: 64 configs x M in {4,8,16} = 192.
# task=repeat, K=16, entropy_variant=exact_conditional, beta=10, seed=0, orbit_weights=uniform.
# Grid: lr(4) x gamma_H(8) x gamma_c(2) x M(3).
# SLURM_ARRAY_TASK_ID -> (lr, gamma_H, gamma_c, M) via mixed-radix indexing.

set -euo pipefail

REPO=/ibex/user/zhapacfp/neurosymbolic-diffusion
PY=$HOME/hott-nesy-venv/bin/python
OUT=runs_nesydm_orbita_grid_hasrep

lrs=(5e-5 1e-4 3e-4 1e-3)          # 4
gHs=(0.01 0.1 0.5 1.0 1.3 1.6 2.0 2.5)  # 8
gcs=(1e-6 1e-4)                    # 2
Ms=(4 8 16)                        # 3

id=${SLURM_ARRAY_TASK_ID}
gc_idx=$(( id % 2 ))
gH_idx=$(( (id / 2) % 8 ))
lr_idx=$(( (id / 16) % 4 ))
M_idx=$(( (id / 64) % 3 ))

lr=${lrs[$lr_idx]}
gH=${gHs[$gH_idx]}
gc=${gcs[$gc_idx]}
M=${Ms[$M_idx]}

echo "task_id=$id lr=$lr gamma_H=$gH gamma_c=$gc orbit_M=$M orbit_weights=uniform"

cd "$REPO"
srun "$PY" -m expressive.experiments.mnist_op.tasks_nesydm \
  --task repeat --K 16 --lr "$lr" \
  --gamma_H "$gH" --gamma_c "$gc" --beta 10 \
  --entropy_variant exact_conditional --seed 0 \
  --orbit_M "$M" --orbit_weights uniform \
  --epochs 200 --patience 30 \
  --no_signal_epoch 60 --no_signal_thresh 0.55 \
  --n_sets 8000 --outdir "$OUT"
