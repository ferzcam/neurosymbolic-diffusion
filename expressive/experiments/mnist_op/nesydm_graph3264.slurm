#!/bin/bash
# NeSyDM on graph tasks at K in {32,64} (previously wrongly excluded as "infeasible" -- actually ~cheap).
# task{conn,col3} x K{32,64} x 16 seeds = 64. best-lr=3e-4 (per earlier grid + col3 K16 best).
# id -> cell=id/16 (0..3); seed=id%16. cells: (conn,32)(conn,64)(col3,32)(col3,64)
#SBATCH --job-name=nd_g3264
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
OUT=$REPO/runs_nesydm_16seed
mkdir -p "$OUT" "$REPO/slurm_logs"
CTASK=(conn conn col3 col3); CK=(32 64 32 64)
id=$SLURM_ARRAY_TASK_ID
cell=$(( id / 16 )); seed=$(( id % 16 ))
task=${CTASK[$cell]}; K=${CK[$cell]}
echo "HOST $(hostname) TASK_ID $id -> task=$task K=$K lr=3e-4 seed=$seed"
cd "$REPO"
srun "$PY" -m expressive.experiments.mnist_op.tasks_nesydm \
  --task "$task" --K "$K" --lr 3e-4 --seed "$seed" \
  --epochs 100 --n_sets 8000 --batch_size 32 --patience 20 \
  --outdir "$OUT" --conv_thresh 0.9
