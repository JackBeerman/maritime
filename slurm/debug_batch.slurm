#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J dbg_batch
#SBATCH -o logs/dbg_batch_%j.out
#SBATCH -e logs/dbg_batch_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 0:45:00
#SBATCH -c 4
#SBATCH --mem=128G

mkdir -p logs
cd ~/maritime
source venv/bin/activate

python3 debug_batching.py \
    --ais-glob "/scratch/jtb3sud/maritime/ais/aisdk-2026-08-25.csv" \
    --n-underway 30 --n-stationary 30 \
    --seq-len 12 --batch-size 32 --gnn-chunk-size 32
