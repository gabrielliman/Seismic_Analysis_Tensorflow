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


# echo "Começou"
# conda run -n seismic_tf2 python ./train_linear.py --folder classification --name "parihaka" -s1 50 -s2 50 --stride1 50 --stride2 50  --delta 1e-4 --patience 30 --epochs 100 -b 16 --dataset 0 --limit True
# echo "Terminou o primeiro"
# conda run -n seismic_tf2 python ./train_linear.py --folder classification --name "penobscot" -s1 50 -s2 50 --stride1 50 --stride2 50  --delta 1e-4 --patience 30 --epochs 100 -b 16 --dataset 1 --limit True
# echo "Terminou o segundo"
# conda run -n seismic_tf2 python ./train_linear.py --folder classification --name "netherlands" -s1 50 -s2 50 --stride1 50 --stride2 50  --delta 1e-4 --patience 30 --epochs 100 -b 16 --dataset 2 --limit True
# echo "Terminou o terceiro"





echo "Começou"
conda run -n seismic_tf2 python ./train_linear.py --folder classification_nolimit --name "parihaka" -s1 50 -s2 50 --stride1 50 --stride2 50  --delta 1e-4 --patience 30 --epochs 100 -b 16 --dataset 0 --limit False
echo "Terminou o primeiro"
conda run -n seismic_tf2 python ./train_linear.py --folder classification_limitclass --name "penobscot" -s1 40 -s2 40 --stride1 20 --stride2 20  --delta 1e-4 --patience 30 --epochs 100 -b 16 --dataset 1 --limit True
echo "Terminou o segundo"


conda run -n seismic_tf2 python ./train_linear.py --folder classification_limitclass --name "netherlands" -s1 40 -s2 40 --stride1 10 --stride2 10  --delta 1e-4 --patience 30 --epochs 100 -b 256 --dataset 1 --limit True

conda run -n seismic_tf2 python ./train_linear.py --folder classification_nolimit  --name "netherlands" -s1 40 -s2 40 --stride1 10 --stride2 10  --delta 1e-4 --patience 30 --epochs 100 -b 16 --dataset 2 --limit False
echo "Terminou o terceiro"
