#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=siliwang94@gmail.com
#SBATCH --partition=bme.gpustudent.q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:0
#SBATCH --job-name=pcsf
#SBATCH --output=/home/bme001/20225898/OC_classification/ffinetune/result/pcs.out

module load cuda11.6/toolkit

source ~/.bashrc
source /home/bme001/20225898/miniconda3/etc/profile.d/conda.sh
conda activate StableDiffusion

cd /home/bme001/20225898/OC_classification/ffinetune/
srun python pcs.py