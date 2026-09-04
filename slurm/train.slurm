#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J maritime_train
#SBATCH -o logs/maritime_train_%j.out
#SBATCH -e logs/maritime_train_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=350G

mkdir -p logs checkpoints
cd ~/maritime
source venv/bin/activate
export CUDA_LAUNCH_BLOCKING=1

python3 train.py \
    --ais-glob "/scratch/jtb3sud/maritime/ais/aisdk-*.csv" \
    --land-shp data/ne_10m_land/ne_10m_land.shp \
    --ports-csv data/ports_denmark.csv \
    --n-underway 300 \
    --n-stationary 300 \
    --seq-len 12 \
    --future-len 4 \
    --hidden 128 \
    --n-layers 4 \
    --batch-size 32 \
    --gnn-chunk-size 32 \
    --n-epochs 60 \
    --checkpoint-path checkpoints/checkpoint_batched.pt

echo "Job finished with exit code $?"
