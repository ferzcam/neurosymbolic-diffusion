#!/bin/bash
# NeSyDM vs NeSyDM+OrbitA on visual Latin-square n in {5,6} (graded reward in RLOO estimator).
# orbit_M{0,16 uniform} x n{5,6} x K{8,16,32} x seed{0..7} = 96 tasks. lr=1e-3.
#SBATCH --job-name=latin56_nd
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
OUT=$REPO/runs_latin56_nesydm
mkdir -p "$OUT" "$REPO/slurm_logs"
MS=(0 16); ns=(5 6); Ks=(8 16 32)
id=$SLURM_ARRAY_TASK_ID
seed=$(( id % 8 )); r=$(( id / 8 ))
K=${Ks[$(( r % 3 ))]}; r2=$(( r / 3 ))
n=${ns[$(( r2 % 2 ))]}; M=${MS[$(( r2 / 2 ))]}
echo "HOST $(hostname) TASK_ID $id -> orbit_M=$M n=$n K=$K seed=$seed"
cd "$REPO"
srun "$PY" -m expressive.experiments.mnist_op.tasks_nesydm \
  --task latin --latin_n "$n" --K "$K" --lr 1e-3 --seed "$seed" \
  --orbit_M "$M" --orbit_weights uniform \
  --epochs 300 --n_sets 8000 --batch_size 32 --patience 80 \
  --outdir "$OUT" --conv_thresh 0.9
