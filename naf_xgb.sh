#!/bin/bash

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=x.lan@student.tue.nl
#SBATCH --partition=bme.gpustudent.q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:0
#SBATCH --job-name=standard_one_week_jobnaf
#SBATCH --output=/home/bme001/20225898/OC_classification/xgboost/out_world_naf.out

module load cuda11.6/toolkit

source ~/.bashrc
source /home/bme001/20225898/miniconda3/etc/profile.d/conda.sh
conda activate job

cd /home/bme001/20225898/OC_classification/
srun python naf_xgb.py
