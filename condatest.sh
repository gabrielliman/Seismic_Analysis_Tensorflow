#!/bin/bash
set -x
# Get the directory of the script
script_dir=$(dirname "$(readlink -f "$0")")

# Echo the full path to the script
echo "Running script: $script_dir/$0"



source /home/grad/ccomp/21/nuneslima/miniconda3/lib/python3.11/site-packages/conda/shell/etc/profile.d/conda.sh
#source /home/grad/ccomp/21/nuneslima/miniconda3/pkgs/conda-23.7.3-py311h06a4308_0/lib/python3.11/site-packages/conda/shell/etc/profile.d/conda.sh
#source /home/grad/ccomp/21/nuneslima/miniconda3/pkgs/conda-23.7.3-py311h06a4308_0/etc/profile.d/conda.sh
#source /home/grad/ccomp/21/nuneslima/miniconda3/pkgs/conda-23.5.2-py311h06a4308_0/lib/python3.11/site-packages/conda/shell/etc/profile.d/conda.sh
#source /home/grad/ccomp/21/nuneslima/miniconda3/pkgs/conda-23.5.2-py311h06a4308_0/etc/profile.d/conda.sh
#source /home/grad/ccomp/21/nuneslima/miniconda3/etc/profile.d/conda.sh


cd ~/Seismic-Analysis/Seismic_Analysis_Tensorflow/

conda run -n seismic_tf python ./train_bayes.py --name "script test" -o 0 -s2 192 --stridetrain 128 --delta 1e-4 --patience 5 --loss_function 0 --folder "bayes" --init_points 1 --num_iter 0 --epochs 1
conda run -n seismic_tf python ./train_bayes.py --name "script_test2" -o 0 -s2 192 --stridetrain 128 --delta 1e-4 --patience 5 --loss_function 0 --folder "bayes" --init_points 1 --num_iter 1 --epochs 1

echo "Acabooou!!!"


