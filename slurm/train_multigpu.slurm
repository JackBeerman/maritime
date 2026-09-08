#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J maritime_ddp
#SBATCH -o logs/maritime_ddp_%j.out
#SBATCH -e logs/maritime_ddp_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH -t 72:00:00
#SBATCH -c 16
#SBATCH --mem=750G
#SBATCH --nodes=1

# 4 GPUs on one node. --ntasks is deliberately NOT set: torchrun spawns
# the worker processes itself, so SLURM should allocate one task holding
# all four GPUs rather than four competing tasks.

mkdir -p logs checkpoints
cd ~/maritime
source venv/bin/activate

# Snapshots live on CPU and only chunks move to GPU, so each rank adds
# little GPU memory -- but each rank does build its own copy of the world
# snapshots in CPU RAM, which is why --mem is generous.
export OMP_NUM_THREADS=4

torchrun --standalone --nproc_per_node=4 train.py \
    --distributed \
    --ais-glob "/scratch/jtb3sud/maritime/ais/aisdk-*.csv" \
    --land-shp data/ne_10m_land/ne_10m_land.shp \
    --ports-csv data/ports_denmark.csv \
    --n-underway 300 \
    --n-stationary 300 \
    --seq-len 16 \
    --future-len 4 \
    --hidden 256 \
    --n-layers 6 \
    --batch-size 32 \
    --gnn-chunk-size 32 \
    --lr-scale-rule sqrt \
    --n-epochs 60 \
    --checkpoint-path checkpoints/checkpoint_ddp.pt \
    --resume

echo "Job finished with exit code $?"
