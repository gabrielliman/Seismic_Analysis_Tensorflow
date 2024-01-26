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
conda run -n seismic_tf python ./train_transfer.py -m 2 --name "new_aerial_imagery" -o 0 -g 3.6 -s1 496 -s2 192 --stridetrain 64 --delta 1e-5 --patience 5 --loss_function 0 --folder "new_aerial_imagery_train" --epochs 100 --dataset 1 -b 4 -k 3 --dropout 0.5 --pre_classes 6

conda run -n seismic_tf python ./train_transfer.py -m 2 --name "new_transfer_aerial" -o 0 -g 3.6 -s1 992 -s2 192 --stridetrain 64 --delta 1e-5 --patience 5 --loss_function 0 --folder "aerial_transfer_results" --epochs 100 --dataset 0 -b 4 -k 3 --weights_path "./checkpoints/new_aerial_imagery_train/checkpoint_new_aerial_imagery.h5" --dropout 0.5 --pre_classes 6 --classes 6


#FOREST IMAGERY
conda run -n seismic_tf python ./train_transfer.py -m 2 --name "forest" -o 0 -g 3.6 -s1 256 -s2 256 --delta 1e-5 --patience 5 --loss_function 0 --folder "forest_train" --epochs 100 --dataset 2 -b 4 -k 3 --dropout 0.5 --pre_classes 2

conda run -n seismic_tf python ./train_transfer.py -m 2 --name "transfer_forest" -o 0 -g 3.6 -s1 992 -s2 192 --stridetrain 64 --delta 1e-5 --patience 5 --loss_function 0 --folder "forest_transfer_results" --epochs 100 --dataset 0 -b 4 -k 3 --weights_path "./checkpoints/forest_train/checkpoint_forest.h5" --pre_classes 2 --classes 6

#DRONE
conda run -n seismic_tf python ./train_transfer.py -m 2 --name "drone" -o 0 -g 3.6 -s1 512 -s2 512 --stridetrain 400 --delta 1e-5 --patience 5 --loss_function 0 --folder "drone_train" --epochs 100 --dataset 3 -b 4 -k 3 --dropout 0.5 --pre_classes 20

conda run -n seismic_tf python ./train_transfer.py -m 2 --name "transfer_drone" -o 0 -g 3.6 -s1 992 -s2 192 --stridetrain 64 --delta 1e-5 --patience 5 --loss_function 0 --folder "drone_transfer_results" --epochs 100 --dataset 0 -b 4 -k 3 --weights_path "./checkpoints/drone_train/checkpoint_drone.h5" --dropout 0.5 --pre_classes 20 --classes 6
