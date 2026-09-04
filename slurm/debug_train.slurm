#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J dbg_train
#SBATCH -o logs/dbg_train_%j.out
#SBATCH -e logs/dbg_train_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -c 4
#SBATCH --mem=128G

cd ~/maritime
source venv/bin/activate

export CUDA_LAUNCH_BLOCKING=1

python3 debug_training.py \
    --ais-glob "/scratch/jtb3sud/maritime/ais/aisdk-2026-08-25.csv" \
    --n-underway 30 --n-stationary 30 \
    --seq-len 12 --batch-size 32 --gnn-chunk-size 32 \
    --hidden 128 --n-layers 4 --n-batches 200
