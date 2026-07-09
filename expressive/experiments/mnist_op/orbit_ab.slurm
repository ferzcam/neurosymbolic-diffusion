#!/bin/bash
# OrbitA A/B on NeSyDM: vanilla (orbit_M=0) vs NeSyDM+OrbitA (orbit_M=16, UNIFORM weights, ESS=M).
# task{repeat,conn,col3} x K{8,16} x arm{M=0, M=16u} x seed{0..7} = 96 tasks. best-lr per task.
# id -> seed=id%8 ; arm=(id/8)%2 (0->M0, 1->M16) ; K=Ks[((id/16))%2] ; task=tasks[(id/32)]
#SBATCH --job-name=orbit_ab
#SBATCH --array=0-95%40
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
OUT=$REPO/runs_orbit_ab
mkdir -p "$OUT" "$REPO/slurm_logs"

tasks=(repeat conn col3); TLR=(1e-3 1e-3 3e-4); Ks=(8 16); MS=(0 16)
id=$SLURM_ARRAY_TASK_ID
seed=$(( id % 8 )); r=$(( id / 8 ))
arm=$(( r % 2 )); r2=$(( r / 2 ))
K=${Ks[$(( r2 % 2 ))]}; ti=$(( r2 / 2 ))
task=${tasks[$ti]}; lr=${TLR[$ti]}; M=${MS[$arm]}

echo "HOST $(hostname) TASK_ID $id -> task=$task lr=$lr K=$K orbit_M=$M seed=$seed"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO"
srun "$PY" -m expressive.experiments.mnist_op.tasks_nesydm \
  --task "$task" --K "$K" --lr "$lr" --seed "$seed" \
  --orbit_M "$M" --orbit_weights uniform \
  --epochs 100 --n_sets 8000 --batch_size 32 --patience 20 \
  --outdir "$OUT" --conv_thresh 0.9
