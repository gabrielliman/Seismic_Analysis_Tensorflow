#!/bin/bash

#SBATCH --job-name=train_segmentation
#SBATCH --time=200:00:00
#SBATCH -w gorgona2 
#SBATCH --mail-user=gabrielliman2002@gmail.com
#SBATCH --mail-type=ALL

set -x

source /home/grad/ccomp/21/nuneslima/miniconda3/etc/profile.d/conda.sh

cd /home/grad/ccomp/21/nuneslima/Seismic-Analysis/Seismic_Analysis_Tensorflow


# echo "Começou"
# conda run -n seismic_tf2 python ./train_linear.py --folder classification --name "parihaka" -s1 50 -s2 50 --stride1 50 --stride2 50  --delta 1e-4 --patience 30 --epochs 100 -b 16 --dataset 0 --limit True
# echo "Terminou o primeiro"
# conda run -n seismic_tf2 python ./train_linear.py --folder classification --name "penobscot" -s1 50 -s2 50 --stride1 50 --stride2 50  --delta 1e-4 --patience 30 --epochs 100 -b 16 --dataset 1 --limit True
# echo "Terminou o segundo"
# conda run -n seismic_tf2 python ./train_linear.py --folder classification --name "netherlands" -s1 50 -s2 50 --stride1 50 --stride2 50  --delta 1e-4 --patience 30 --epochs 100 -b 16 --dataset 2 --limit True
# echo "Terminou o terceiro"





echo "Começou"
conda run -n seismic_tf2 python ./train_loop.py