#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J irr_ddp
#SBATCH -o logs/irr_ddp_%j.out
#SBATCH -e logs/irr_ddp_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH -t 72:00:00
#SBATCH -c 16
#SBATCH --mem=750G
#SBATCH --nodes=1

mkdir -p logs checkpoints
cd ~/maritime-irregular
source venv/bin/activate

export OMP_NUM_THREADS=4

torchrun --standalone --nproc_per_node=4 train_irregular.py \
    --distributed \
    --ais-glob "/scratch/jtb3sud/maritime/ais/aisdk-*.csv" \
    --land-shp data/ne_10m_land/ne_10m_land.shp \
    --ports-csv data/ports_denmark.csv \
    --n-underway 300 \
    --n-stationary 300 \
    --min-ping-gap-sec 900 \
    --seq-len 16 \
    --future-len 4 \
    --hidden 256 \
    --n-layers 6 \
    --batch-size 32 \
    --gnn-chunk-size 16 \
    --lr-scale-rule sqrt \
    --n-epochs 500 \
    --checkpoint-path checkpoints/irregular_ddp.pt \
    --resume

echo "Job finished with exit code $?"
