#!/bin/bash
#SBATCH -J yolo_1
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH --gres=gpu:2
module load anaconda3/2019.07
module load cuda/9.2
source activate detect
python -u -m pip install -e detectron2
