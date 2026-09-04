#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J maritime_smoke
#SBATCH -o logs/maritime_smoke_%j.out
#SBATCH -e logs/maritime_smoke_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 0:30:00
#SBATCH -c 4
#SBATCH --mem=64G

mkdir -p logs checkpoints
cd ~/maritime
source venv/bin/activate

python3 train.py \
    --ais-glob "/scratch/jtb3sud/maritime/ais/aisdk-*.csv" \
    --land-shp data/ne_10m_land/ne_10m_land.shp \
    --ports-csv data/ports_denmark.csv \
    --n-underway 30 \
    --n-stationary 30 \
    --n-epochs 5 \
    --checkpoint-path checkpoints/smoke_checkpoint.pt

echo "Job finished with exit code $?"
