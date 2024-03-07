#!/bin/bash

#SBATCH --output=slurm_seismic/output_%j.txt 
#SBATCH --error=slurm_seismic/error_%j.txt 
#SBATCH --job-name=linear_classification
#SBATCH --time=6:00:00
#SBATCH -w gorgona2 
#SBATCH --mail-user=gabrielliman2002@gmail.com
#SBATCH --mail-type=ALL

set -x

source /home/grad/ccomp/21/nuneslima/miniconda3/etc/profile.d/conda.sh

cd /home/grad/ccomp/21/nuneslima/Seismic-Analysis/Seismic_Analysis_Tensorflow

conda run -n seismic_tf2 python ./train_linear.py --name "linear_50_50" --stride1 50 --stride2 50  --delta 1e-4 --patience 30 --epochs 150
conda run -n seismic_tf2 python ./train_linear.py --name "linear_50_40" --stride1 40 --stride2 40  --delta 1e-4 --patience 30 --epochs 150
conda run -n seismic_tf2 python ./train_linear.py --name "linear_40_10" -s1 40 -s2 40 --stride1 10 --stride2 10  --delta 1e-4 --patience 30 --epochs 150