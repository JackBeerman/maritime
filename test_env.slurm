#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J env_test
#SBATCH -o env_test.out
#SBATCH -e env_test.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 0:05:00
#SBATCH -c 2
#SBATCH --mem=16G

source ~/maritime/venv/bin/activate
python3 -c "
import torch
print('torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
import torch_geometric
print('torch_geometric:', torch_geometric.__version__)
import geopandas
print('geopandas:', geopandas.__version__)
import shapely
print('shapely:', shapely.__version__)
print('ALL IMPORTS OK')
"
