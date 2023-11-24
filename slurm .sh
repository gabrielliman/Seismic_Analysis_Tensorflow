#!/bin/bash

#SBATCH --output=SEISMIC/output_%j.txt 
#SBATCH --error=SEISMIC/error_%j.txt 
#SBATCH --job-name=seismic_bayesian_opt
#SBATCH --time=48:00:00
#SBATCH -w gorgona2 
#SBATCH --mail-user=gabrielliman2002@gmail.com
#SBATCH --mail-type=ALL

set -x

source /home/grad/ccomp/21/nuneslima/miniconda3/etc/profile.d/conda.sh

cd ~/Seismic-Analysis/Seismic_Analysis_Tensorflow/

conda run -n seismic_tf ./train_bayes.py --name "attention_bayes_10_20" -o 0 -s2 192 --stridetrain 64 --delta 1e-4 --patience 3 --loss_function 0 --folder "bayes" --init_points 10 --num_iter 20 --epochs 100
conda run -n pfenet ./train.sh pascal split0_resnet50

echo "Acabooou!!!"
