#!/bin/bash

cd ~/Seismic-Analysis/Seismic_Analysis_Tensorflow/

python ./train_bayes.py --name "attention_bayes_10_20" -o 0 -s2 192 --stridetrain 64 --delta 1e-4 --patience 5 --loss_function 0 --folder "bayes" --init_points 10 --num_iter 20 --epochs 100
python ./train_bayes.py --name "attention_bayes_20_10" -o 0 -s2 192 --stridetrain 64 --delta 1e-4 --patience 5 --loss_function 0 --folder "bayes" --init_points 20 --num_iter 10 --epochs 100
exit
echo "Acabooou!!!"
