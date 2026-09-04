#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J irr_large
#SBATCH -o logs/irr_large_%j.out
#SBATCH -e logs/irr_large_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=350G

mkdir -p logs checkpoints
cd ~/maritime-irregular
source venv/bin/activate

export CUDA_LAUNCH_BLOCKING=1

python3 train_irregular.py \
    --ais-glob "/scratch/jtb3sud/maritime/ais/aisdk-*.csv" \
    --land-shp data/ne_10m_land/ne_10m_land.shp \
    --ports-csv data/ports_denmark.csv \
    --n-underway 300 \
    --n-stationary 300 \
    --min-ping-gap-sec 900 \
    --staleness-cutoff-sec 1800 \
    --neighbor-radius-deg 0.5 \
    --max-neighbors 150 \
    --seq-len 24 \
    --future-len 8 \
    --hidden 256 \
    --n-layers 6 \
    --batch-size 32 \
    --gnn-chunk-size 16 \
    --n-epochs 60 \
    --checkpoint-path checkpoints/irregular_large.pt

echo "Job finished with exit code $?"
