#!/bin/bash

#SBATCH --output=slurm_seismic/output_%j.txt 
#SBATCH --error=slurm_seismic/error_%j.txt 
#SBATCH --job-name=seismic_bayesian_opt
#SBATCH --time=200:00:00
#SBATCH -w gorgona2 
#SBATCH --mail-user=gabrielliman2002@gmail.com
#SBATCH --mail-type=ALL

set -x

source /home/grad/ccomp/21/nuneslima/miniconda3/etc/profile.d/conda.sh

cd /home/grad/ccomp/21/nuneslima/Seismic-Analysis/Seismic_Analysis_Tensorflow

conda run -n seismic_tf_yes python ./train_bayes.py --name "penobscot_bayes_10_100" -o 0 -s1 1024 -s2 192 --stride1 256 --stride2 64 --delta 1e-4 --patience 10 --loss_function 0 --folder "penobscot_bayes" --init_points 10 --num_iter 100 --epochs 100 --dataset 1