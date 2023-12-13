#!/bin/bash

#SBATCH --output=slurm_seismic/output_%j.txt 
#SBATCH --error=slurm_seismic/error_%j.txt 
#SBATCH --job-name=seismic_bayesian_opt
#SBATCH --time=96:00:00
#SBATCH -w gorgona2 
#SBATCH --mail-user=gabrielliman2002@gmail.com
#SBATCH --mail-type=ALL

set -x

source /home/grad/ccomp/21/nuneslima/miniconda3/etc/profile.d/conda.sh

cd /home/grad/ccomp/21/nuneslima/Seismic-Analysis/Seismic_Analysis_Tensorflow

conda run -n seismic_tf python ./train_bayes.py --name "attention_bayes_20_100_continue" --last_iter "attention_bayes_20_100_checkpoint" -o 0 -s2 192 --stridetrain 64 --delta 1e-4 --patience 5 --loss_function 0 --folder "bayes" --init_points 5 --num_iter 100 --epochs 100
