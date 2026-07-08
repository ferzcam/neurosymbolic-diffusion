#!/bin/bash
# NeSyDM baseline grid on our 3 symmetric NeSy tasks (repeat, conn, col3).
# Grid: task(3) x lr{3e-4,1e-3,3e-3}(3) x K{8,16,32,64}(4) x seed{0..7}(8) = 288 cells.
# K = world samples -> loss_S = variational_K = test_K = K. All diffusion internals at repo defaults.
# id -> seed = id%8 ; K = Ks[(id/8)%4] ; lr = lrs[(id/32)%3] ; task = tasks[id/96]
#SBATCH --job-name=nesydm_tasks
#SBATCH --array=0-287%40
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
OUT=$REPO/runs_tasks_nesydm
mkdir -p "$OUT" "$REPO/slurm_logs"

tasks=(repeat conn col3); lrs=(3e-4 1e-3 3e-3); Ks=(8 16 32 64)
id=$SLURM_ARRAY_TASK_ID
seed=$(( id % 8 )); r=$(( id / 8 ))
K=${Ks[$(( r % 4 ))]}; r2=$(( r / 4 ))
lr=${lrs[$(( r2 % 3 ))]}; task=${tasks[$(( r2 / 3 ))]}

echo "HOST $(hostname) TASK_ID $id -> task=$task lr=$lr K=$K seed=$seed"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO"
srun "$PY" -m expressive.experiments.mnist_op.tasks_nesydm \
  --task "$task" --K "$K" --lr "$lr" --seed "$seed" \
  --epochs 100 --n_sets 8000 --batch_size 32 --patience 20 \
  --outdir "$OUT" --conv_thresh 0.9
