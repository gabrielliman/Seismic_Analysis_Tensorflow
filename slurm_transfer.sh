#!/bin/bash

#SBATCH --output=slurm_seismic_transfer/output_%j.txt 
#SBATCH --error=slurm_seismic_transfer/error_%j.txt 
#SBATCH --job-name=seismic_transfer_learning
#SBATCH --time=11:00:00
#SBATCH -w gorgona2 
#SBATCH --mail-user=gabrielliman2002@gmail.com
#SBATCH --mail-type=ALL

set -x

source /home/grad/ccomp/21/nuneslima/miniconda3/etc/profile.d/conda.sh

cd /home/grad/ccomp/21/nuneslima/Seismic-Analysis/Seismic_Analysis_Tensorflow

#AERIAL IMAGERY

#treinando primeiro
conda run -n seismic_tf python ./train_transfer.py -m 2 --name "aerial_imagery" -o 0 -g 3.6 -s1 496 -s2 192 --stridetrain 64 --delta 1e-4 --patience 5 --loss_function 0 --folder "aerial_imagery_train" --epochs 100 --dataset 1 -b 4 -k 7

conda run -n seismic_tf python ./train_transfer.py -m 2 --name "transfer_aerial" -o 0 -g 3.6 -s1 992 -s2 192 --stridetrain 64 --delta 1e-4 --patience 5 --loss_function 0 --folder "transfer_results" --epochs 100 --dataset 0 -b 4 -k 7 --weights_path "./checkpoints/aerial_imagery_train/checkpoint_aerial_imagery.h5"

#NAO FUNCIONA PQ TEM NUMERO DE CLASSES DIFERENTE
#FOREST IMAGERY
# conda run -n seismic_tf python ./train_transfer.py -m 2 --name "forest" -o 0 -g 3.6 -s1 256 -s2 256 --delta 1e-4 --patience 5 --loss_function 0 --folder "forest_train" --epochs 100 --dataset 2 -b 4 -k 7 --classes 2

# conda run -n seismic_tf python ./train_transfer.py -m 2 --name "transfer_forest" -o 0 -g 3.6 -s1 992 -s2 192 --stridetrain 64 --delta 1e-4 --patience 5 --loss_function 0 --folder "transfer_results" --epochs 100 --dataset 0 -b 4 -k 7 --weights_path "./checkpoints/forest_train/checkpoint_forest.h5"

