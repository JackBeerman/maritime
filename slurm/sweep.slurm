#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J maritime_sweep
#SBATCH -o logs/sweep_%j.out
#SBATCH -e logs/sweep_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 8:00:00
#SBATCH -c 4
#SBATCH --mem=200G

# One axis per submission. Run mesh first -- it also warms the AIS cache
# that the other two axes reuse, so they start much faster.
#   sbatch --export=AXIS=mesh    slurm/sweep.slurm
#   sbatch --export=AXIS=context slurm/sweep.slurm
#   sbatch --export=AXIS=horizon slurm/sweep.slurm

AXIS="${AXIS:-mesh}"

mkdir -p logs results cache
cd ~/maritime
source venv/bin/activate

export CUDA_LAUNCH_BLOCKING=1

python3 sweep.py \
    --axis "$AXIS" \
    --out-dir "results/sweep_${AXIS}" \
    --ais-glob "/scratch/jtb3sud/maritime/ais/aisdk-2026-08-25.csv" \
    --n-underway 60 --n-stationary 60 \
    --hidden 128 --n-layers 3 \
    --batch-size 32 --gnn-chunk-size 32 \
    --n-epochs 15 \
    --skip-existing

echo "Job finished with exit code $?"
