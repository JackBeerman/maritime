#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J irr_smoke
#SBATCH -o logs/irr_smoke_%j.out
#SBATCH -e logs/irr_smoke_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -c 4
#SBATCH --mem=64G

mkdir -p logs checkpoints
cd ~/maritime
source venv/bin/activate

python3 train_irregular.py \
    --ais-glob "/scratch/jtb3sud/maritime/ais/aisdk-2026-08-25.csv" \
    --land-shp data/ne_10m_land/ne_10m_land.shp \
    --ports-csv data/ports_denmark.csv \
    --n-underway 20 --n-stationary 20 \
    --min-ping-gap-sec 60 \
    --seq-len 12 --future-len 4 \
    --hidden 64 --n-layers 3 \
    --n-epochs 3 \
    --checkpoint-path checkpoints/irr_smoke.pt

echo "Job finished with exit code $?"
