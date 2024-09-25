#!/bin/bash
readonly GPU_ID=0

for num_extra_train in {33..100}; do
  python ./train.py \
    --num_extra_train $num_extra_train \
    --optimizer 0 \
    --gamma 3.6 \
    --model 2 \
    --epochs 100 \
    --batch_size 4 \
    --name "$num_extra_train"_slice \
    --stride1 128 \
    --stride2 64 \
    --stridetest1 128 \
    --stridetest2 64 \
    --slice_shape1 992 \
    --slice_shape2 192 \
    --delta 1e-4 \
    --patience 15 \
    --loss_function 1 \
    --folder only_train_slices \
    --dataset 5 \
    --kernel 7 \
    --dropout 0.5 \
    --gpuID $GPU_ID
done