#!/bin/bash
#SBATCH --job-name=nesy_graph
#SBATCH --array=0-35%16
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/ibex/user/zhapacfp/neurosymbolic-diffusion/slurm_logs/graphgrid_%A_%a.out
#SBATCH --error=/ibex/user/zhapacfp/neurosymbolic-diffusion/slurm_logs/graphgrid_%A_%a.out

# Vanilla NeSyDM (orbit_M=0) reduced hyperparameter grid on the graph tasks.
# Tune {lr, gamma_H, gamma_c} at K=16 with exact_conditional, per task; then 16-seed the winner.
# task(2) x gamma_H(3) x lr(3) x gamma_c(2) = 36 configs, seed 0.
# SLURM_ARRAY_TASK_ID -> (task, gamma_H, lr, gamma_c) via mixed-radix indexing.

set -euo pipefail

REPO=/ibex/user/zhapacfp/neurosymbolic-diffusion
PY=$HOME/hott-nesy-venv/bin/python
OUT=runs_nesydm_grid_graph

tasks=(conn col3)                  # 2
gHs=(0.01 0.5 1.6)                 # 3
lrs=(1e-4 3e-4 1e-3)               # 3
gcs=(1e-6 1e-4)                    # 2

id=${SLURM_ARRAY_TASK_ID}
gc_idx=$(( id % 2 ))
lr_idx=$(( (id / 2) % 3 ))
gH_idx=$(( (id / 6) % 3 ))
task_idx=$(( (id / 18) % 2 ))

task=${tasks[$task_idx]}
gH=${gHs[$gH_idx]}
lr=${lrs[$lr_idx]}
gc=${gcs[$gc_idx]}

echo "task_id=$id task=$task lr=$lr gamma_H=$gH gamma_c=$gc"

cd "$REPO"
srun "$PY" -m expressive.experiments.mnist_op.tasks_nesydm \
  --task "$task" --K 16 --lr "$lr" \
  --gamma_H "$gH" --gamma_c "$gc" --beta 10 \
  --entropy_variant exact_conditional --seed 0 --orbit_M 0 \
  --epochs 200 --patience 30 \
  --no_signal_epoch 60 --no_signal_thresh 0.55 \
  --n_sets 8000 --outdir "$OUT"
