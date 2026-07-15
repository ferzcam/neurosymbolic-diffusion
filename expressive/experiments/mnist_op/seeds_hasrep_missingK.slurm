#!/bin/bash
#SBATCH --job-name=nesy_hrK
#SBATCH --array=0-47%16
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/ibex/user/zhapacfp/neurosymbolic-diffusion/slurm_logs/hrK_%A_%a.out
#SBATCH --error=/ibex/user/zhapacfp/neurosymbolic-diffusion/slurm_logs/hrK_%A_%a.out

# has-a-repeat tuned winner at 16 seeds x the missing K in {4,8,64} (K=16 already done in
# runs_nesydm_grid_hasrep_seeds). Winner config from the 64-config grid (exact_conditional):
#   lr=3e-4, gamma_H=0.01, gamma_c=1e-4, beta=10.
# K(3) x seed(16) = 48. index -> (K, seed).

set -euo pipefail

REPO=/ibex/user/zhapacfp/neurosymbolic-diffusion
PY=$HOME/hott-nesy-venv/bin/python
OUT=runs_nesydm_grid_hasrep_seeds

Ks=(4 8 64)
id=${SLURM_ARRAY_TASK_ID}
seed=$(( id % 16 ))
K_idx=$(( (id / 16) % 3 ))
K=${Ks[$K_idx]}

echo "task_id=$id task=repeat K=$K seed=$seed (winner: lr=3e-4 gH=0.01 gc=1e-4 exact_conditional)"

cd "$REPO"
srun "$PY" -m expressive.experiments.mnist_op.tasks_nesydm \
  --task repeat --K "$K" --lr 3e-4 \
  --gamma_H 0.01 --gamma_c 1e-4 --beta 10 \
  --entropy_variant exact_conditional --seed "$seed" --orbit_M 0 \
  --epochs 200 --patience 30 \
  --no_signal_epoch 60 --no_signal_thresh 0.55 \
  --n_sets 8000 --outdir "$OUT"
