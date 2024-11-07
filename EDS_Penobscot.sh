#!/bin/bash
readonly GPU_ID=0
#Penobscot [5,15,25,35,45]
    #BridgeNet
        for num_extra_train in 5 15 25 35 45; do
            python ./train.py \
                --num_extra_train $num_extra_train \
                --optimizer 0 \
                --model 3 \
                --epochs 100 \
                --batch_size 16 \
                --name "$num_extra_train"_slice \
                --stride1 15 \
                --stride2 5 \
                --stridetest1 128 \
                --stridetest2 192 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder BridgeNet_only_train_slices_penobscot_992 \
                --dataset 9 \
                --kernel 11 \
                --dropout 0 \
                --gpuID $GPU_ID
        done
    #Attention
        for num_extra_train in 5 15 25 35 45; do
            python ./train.py \
                --num_extra_train $num_extra_train \
                --optimizer 0 \
                --model 2 \
                --epochs 100 \
                --batch_size 16 \
                --name "$num_extra_train"_slice \
                --stride1 15 \
                --stride2 5 \
                --stridetest1 128 \
                --stridetest2 192 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder Attention_only_train_slices_penobscot_992 \
                --dataset 9 \
                --kernel 11 \
                --dropout 0 \
                --gpuID $GPU_ID
        done
    #UNet
        for num_extra_train in 5 15 25 35 45; do
            python ./train.py \
                --num_extra_train $num_extra_train \
                --optimizer 0 \
                --model 0 \
                --epochs 100 \
                --batch_size 16 \
                --name "$num_extra_train"_slice \
                --stride1 15 \
                --stride2 5d \
                --stridetest1 128 \
                --stridetest2 192 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder UNet_only_train_slices_penobscot_992 \
                --dataset 9 \
                --kernel 11 \
                --dropout 0 \
                --gpuID $GPU_ID
        done
    #Unet3+
        for num_extra_train in 5 15 25 35 45; do
            python ./train.py \
                --num_extra_train $num_extra_train \
                --optimizer 0 \
                --model 1 \
                --epochs 100 \
                --batch_size 16 \
                --name "$num_extra_train"_slice \
                --stride1 15 \
                --stride2 5 \
                --stridetest1 128 \
                --stridetest2 192 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder UNet3+_only_train_slices_penobscot_992 \
                --dataset 9 \
                --kernel 11 \
                --dropout 0 \
                --gpuID $GPU_ID
        done
    #LWBAUNET
        for num_extra_train in 5 15 25 35 45; do
            python ./train.py \
                --num_extra_train $num_extra_train \
                --optimizer 0 \
                --model 11 \
                --epochs 100 \
                --batch_size 16 \
                --name "$num_extra_train"_slice \
                --stride1 15 \
                --stride2 5 \
                --stridetest1 128 \
                --stridetest2 192 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder LWBNA_only_train_slices_penobscot_992 \
                --dataset 9 \
                --kernel 11 \
                --dropout 0 \
                --gpuID $GPU_ID
        done
    #CPFNetM
        for num_extra_train in 5 15 25 35 45; do
            python ./train.py \
                --num_extra_train $num_extra_train \
                --optimizer 0 \
                --model 4 \
                --epochs 100 \
                --batch_size 16 \
                --name "$num_extra_train"_slice \
                --stride1 15 \
                --stride2 5 \
                --stridetest1 128 \
                --stridetest2 192 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder CPFNetM_only_train_slices_penobscot_992 \
                --dataset 9 \
                --kernel 11 \
                --dropout 0 \
                --gpuID $GPU_ID
        done
    #ENet
        for num_extra_train in 5 15 25 35 45; do
            python ./train.py \
                --num_extra_train $num_extra_train \
                --optimizer 0 \
                --model 6 \
                --epochs 100 \
                --batch_size 16 \
                --name "$num_extra_train"_slice \
                --stride1 15 \
                --stride2 5 \
                --stridetest1 128 \
                --stridetest2 192 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder ENet_only_train_slices_penobscot_992 \
                --dataset 9 \
                --kernel 11 \
                --dropout 0 \
                --gpuID $GPU_ID
        done
    #ESPNet
        for num_extra_train in 5 15 25 35 45; do
            python ./train.py \
                --num_extra_train $num_extra_train \
                --optimizer 0 \
                --model 7 \
                --epochs 100 \
                --batch_size 16 \
                --name "$num_extra_train"_slice \
                --stride1 15 \
                --stride2 5 \
                --stridetest1 128 \
                --stridetest2 192 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder ESPNet_only_train_slices_penobscot_992 \
                --dataset 9 \
                --kernel 11 \
                --dropout 0 \
                --gpuID $GPU_ID
        done
    #ICNet
        for num_extra_train in 5 15 25 35 45; do
            python ./train.py \
                --num_extra_train $num_extra_train \
                --optimizer 0 \
                --model 8 \
                --epochs 100 \
                --batch_size 16 \
                --name "$num_extra_train"_slice \
                --stride1 15 \
                --stride2 5 \
                --stridetest1 128 \
                --stridetest2 192 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder ICNet_only_train_slices_penobscot_992 \
                --dataset 9 \
                --kernel 11 \
                --dropout 0 \
                --gpuID $GPU_ID
        done
    #EfficientNet B1
        for num_extra_train in 5 15 25 35 45; do
            python ./train.py \
                --num_extra_train $num_extra_train \
                --optimizer 0 \
                --model 10 \
                --epochs 100 \
                --batch_size 16 \
                --name "$num_extra_train"_slice \
                --stride1 15 \
                --stride2 5 \
                --stridetest1 128 \
                --stridetest2 192 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder EfficientNetB1_only_train_slices_penobscot_992 \
                --dataset 9 \
                --kernel 11 \
                --dropout 0 \
                --gpuID $GPU_ID
        done