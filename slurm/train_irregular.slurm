#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J maritime_irreg
#SBATCH -o logs/maritime_irreg_%j.out
#SBATCH -e logs/maritime_irreg_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=350G

mkdir -p logs checkpoints
cd ~/maritime-irregular
source venv/bin/activate

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
    --seq-len 12 \
    --future-len 4 \
    --hidden 128 \
    --n-layers 4 \
    --batch-size 32 \
    --gnn-chunk-size 16 \
    --n-epochs 60 \
    --checkpoint-path checkpoints/irregular_batched.pt

echo "Job finished with exit code $?"
