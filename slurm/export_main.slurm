#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J export_main
#SBATCH -o logs/export_main_%j.out
#SBATCH -e logs/export_main_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -c 4
#SBATCH --mem=200G

cd ~/maritime
source venv/bin/activate

python3 export_metrics.py \
    --checkpoint checkpoints/checkpoint_batched.pt \
    --out results/main_metrics.json \
    --seq-len 12 --future-len 4 \
    --n-underway 300 --n-stationary 300

echo "Job finished with exit code $?"
