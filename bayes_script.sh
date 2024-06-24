#!/bin/bash

cd ~/Seismic-Analysis/Seismic_Analysis_Tensorflow/

python ./train_bayes.py --name "penobscot_bayes_10_100" -o 0 -s1 1024 -s2 192 --stride1 256 --stride2 64 --delta 1e-4 --patience 10 --loss_function 0 --folder "penobscot_bayes" --init_points 10 --num_iter 100 --epochs 100 --dataset 1
exit
echo "Acabooou!!!"