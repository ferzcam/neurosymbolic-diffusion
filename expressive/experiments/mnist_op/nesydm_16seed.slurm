#!/bin/bash
# NeSyDM at 16 seeds, FEASIBLE cells only (fairness). best-lr-per-config, prediction-saving driver.
# Cells (task,K,lr): repeat{4,8,16,64}; conn{8,16}; col3{8,16}. NO K>=32 on graph tasks.
#   repeat: (4,3e-4)(8,3e-4)(16,1e-3)(64,1e-3)   conn: (8,1e-3)(16,3e-4)   col3: (8,3e-3)(16,3e-4)
# id -> cell=id/16 (0..7) ; seed=id%16
#SBATCH --job-name=nesydm16
#SBATCH --array=0-127%40
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/ibex/user/zhapacfp/neurosymbolic-diffusion/slurm_logs/%A_%a.out
#SBATCH --error=/ibex/user/zhapacfp/neurosymbolic-diffusion/slurm_logs/%A_%a.out
set -e
REPO=/ibex/user/zhapacfp/neurosymbolic-diffusion
PY=$HOME/nesydm-venv/bin/python
export MNIST_ROOT=$HOME/mnist_data
OUT=$REPO/runs_nesydm_16seed
mkdir -p "$OUT" "$REPO/slurm_logs"
CTASK=(repeat repeat repeat repeat conn conn col3 col3)
CK=(4 8 16 64 8 16 8 16)
CLR=(3e-4 3e-4 1e-3 1e-3 1e-3 3e-4 3e-3 3e-4)
id=$SLURM_ARRAY_TASK_ID
cell=$(( id / 16 )); seed=$(( id % 16 ))
task=${CTASK[$cell]}; K=${CK[$cell]}; lr=${CLR[$cell]}
echo "HOST $(hostname) TASK_ID $id -> task=$task K=$K lr=$lr seed=$seed"
cd "$REPO"
srun "$PY" -m expressive.experiments.mnist_op.tasks_nesydm \
  --task "$task" --K "$K" --lr "$lr" --seed "$seed" \
  --epochs 100 --n_sets 8000 --batch_size 32 --patience 20 \
  --outdir "$OUT" --conv_thresh 0.9
