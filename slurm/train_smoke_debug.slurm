#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J smoke_dbg
#SBATCH -o logs/smoke_dbg_%j.out
#SBATCH -e logs/smoke_dbg_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH -c 4
#SBATCH --mem=128G

cd ~/maritime
source venv/bin/activate

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

python3 train.py \
    --ais-glob "/scratch/jtb3sud/maritime/ais/aisdk-2026-08-25.csv" \
    --n-underway 30 --n-stationary 30 \
    --n-epochs 1 \
    --checkpoint-path checkpoints/smoke_debug.pt
