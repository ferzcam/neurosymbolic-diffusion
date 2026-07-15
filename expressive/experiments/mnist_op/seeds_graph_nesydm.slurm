#!/bin/bash
#SBATCH --job-name=nesy_gseed
#SBATCH --array=0-127%16
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/ibex/user/zhapacfp/neurosymbolic-diffusion/slurm_logs/gseed_%A_%a.out
#SBATCH --error=/ibex/user/zhapacfp/neurosymbolic-diffusion/slurm_logs/gseed_%A_%a.out

# Vanilla NeSyDM (orbit_M=0) tuned-winner at 16 seeds x K in {8,16,32,64}, per graph task.
# Winners from the reduced grid (unconditional entropy):
#   conn : gamma_H=0.01, gamma_c=1e-4, lr=1e-4
#   col3 : gamma_H=1.6,  gamma_c=1e-4, lr=1e-4
# task(2) x K(4) x seed(16) = 128. index -> (task, K, seed) via mixed-radix.

set -euo pipefail

REPO=/ibex/user/zhapacfp/neurosymbolic-diffusion
PY=$HOME/hott-nesy-venv/bin/python
OUT=runs_nesydm_graph_seeds

tasks=(conn col3)
Ks=(8 16 32 64)

id=${SLURM_ARRAY_TASK_ID}
seed=$(( id % 16 ))
K_idx=$(( (id / 16) % 4 ))
task_idx=$(( (id / 64) % 2 ))

task=${tasks[$task_idx]}
K=${Ks[$K_idx]}
if [ "$task" = "conn" ]; then gH=0.01; else gH=1.6; fi
gc=1e-4
lr=1e-4

echo "task_id=$id task=$task K=$K seed=$seed gamma_H=$gH"

cd "$REPO"
srun "$PY" -m expressive.experiments.mnist_op.tasks_nesydm \
  --task "$task" --K "$K" --lr "$lr" \
  --gamma_H "$gH" --gamma_c "$gc" --beta 10 \
  --entropy_variant unconditional --seed "$seed" --orbit_M 0 \
  --epochs 200 --patience 30 \
  --no_signal_epoch 60 --no_signal_thresh 0.55 \
  --n_sets 8000 --outdir "$OUT"
