#!/bin/bash
# NeSyDM re-run of FAST cells WITH prediction saving (for concept-accuracy up-to-relabeling).
# best-lr per cell (from finished grid 48338384 aggregation), 8 seeds each = 64 tasks.
# Cells (task,K,lr):
#   repeat: (8,3e-4) (16,1e-3) (32,3e-4) (64,1e-3)     conn: (8,1e-3) (16,3e-4)     col3: (8,3e-3) (16,3e-4)
# id -> cell = id/8 ; seed = id%8.  K>=32 graph cells deliberately EXCLUDED (infeasible).
#SBATCH --job-name=nesydm_preds
#SBATCH --array=0-63%40
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
OUT=$REPO/runs_tasks_nesydm_preds
mkdir -p "$OUT" "$REPO/slurm_logs"

CTASK=(repeat repeat repeat repeat conn conn col3 col3)
CK=(8 16 32 64 8 16 8 16)
CLR=(3e-4 1e-3 3e-4 1e-3 1e-3 3e-4 3e-3 3e-4)
id=$SLURM_ARRAY_TASK_ID
cell=$(( id / 8 )); seed=$(( id % 8 ))
task=${CTASK[$cell]}; K=${CK[$cell]}; lr=${CLR[$cell]}

echo "HOST $(hostname) TASK_ID $id -> cell=$cell task=$task lr=$lr K=$K seed=$seed"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO"
srun "$PY" -m expressive.experiments.mnist_op.tasks_nesydm \
  --task "$task" --K "$K" --lr "$lr" --seed "$seed" \
  --epochs 100 --n_sets 8000 --batch_size 32 --patience 20 \
  --outdir "$OUT" --conv_thresh 0.9
