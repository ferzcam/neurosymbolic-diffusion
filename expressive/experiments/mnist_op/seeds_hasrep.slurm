#!/bin/bash
#SBATCH --job-name=nesy_hasrep_seeds
#SBATCH --array=0-15%16
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/ibex/user/zhapacfp/neurosymbolic-diffusion/slurm_logs/hasrep_seeds_%A_%a.out
#SBATCH --error=/ibex/user/zhapacfp/neurosymbolic-diffusion/slurm_logs/hasrep_seeds_%A_%a.out

# Phase 2: winning has-a-repeat config across 16 seeds.
# Winner: lr=3e-4, gamma_H=0.01, gamma_c=1e-4, beta=10, entropy_variant=exact_conditional.
# SLURM_ARRAY_TASK_ID = seed (0..15).

set -euo pipefail

REPO=/ibex/user/zhapacfp/neurosymbolic-diffusion
PY=$HOME/hott-nesy-venv/bin/python
OUT=runs_nesydm_grid_hasrep_seeds

seed=${SLURM_ARRAY_TASK_ID}

echo "seed=$seed"
cd "$REPO"
srun "$PY" -m expressive.experiments.mnist_op.tasks_nesydm \
  --task repeat --K 16 --lr 3e-4 \
  --gamma_H 0.01 --gamma_c 1e-4 --beta 10 \
  --entropy_variant exact_conditional --seed "$seed" \
  --epochs 200 --patience 30 \
  --no_signal_epoch 60 --no_signal_thresh 0.55 \
  --n_sets 8000 --outdir "$OUT"
