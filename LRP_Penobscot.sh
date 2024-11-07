#!/bin/bash
readonly GPU_ID=1

    #BridgeNet
        python ./train.py \
        --optimizer 0 \
        --model 3 \
        --epochs 100 \
        --batch_size 16 \
        --name BridgeNet_baseline_penobscot \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 1472 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder Baseline_penobscot \
        --dataset 1 \
        --kernel 11 \
        --dropout 0 \
        --gpuID $GPU_ID 
    #Attention
        python ./train.py \
        --optimizer 0 \
        --model 2 \
        --epochs 100 \
        --batch_size 16 \
        --name Attention_baseline_penobscot \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 1472 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder Baseline_penobscot \
        --dataset 1 \
        --kernel 11 \
        --dropout 0 \
        --gpuID $GPU_ID
    #UNet
        python ./train.py \
        --optimizer 0 \
        --model 0 \
        --epochs 100 \
        --batch_size 16 \
        --name UNet_baseline_penobscot \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 1472 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder Baseline_penobscot \
        --dataset 1 \
        --kernel 11 \
        --dropout 0 \
        --gpuID $GPU_ID 
    #Unet3+
        python ./train.py \
        --optimizer 0 \
        --model 1 \
        --epochs 100 \
        --batch_size 16 \
        --name UNet3+_baseline_penobscot \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 1472 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder Baseline_penobscot \
        --dataset 1 \
        --kernel 11 \
        --dropout 0 \
        --gpuID $GPU_ID 
    #LWBAUNET 
        python ./train.py \
        --optimizer 0 \
        --model 11 \
        --epochs 100 \
        --batch_size 16 \
        --name LWBNA_baseline_penobscot \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 1472 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder Baseline_penobscot \
        --dataset 1 \
        --kernel 11 \
        --dropout 0 \
        --gpuID $GPU_ID 
    #CPFNetM
        python ./train.py \
        --optimizer 0 \
        --model 4 \
        --epochs 100 \
        --batch_size 16 \
        --name CPFNetM_baseline_penobscot \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 1472 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder Baseline_penobscot \
        --dataset 1 \
        --kernel 11 \
        --dropout 0 \
        --gpuID $GPU_ID 
    #ENet
        python ./train.py \
        --optimizer 0 \
        --model 6 \
        --epochs 100 \
        --batch_size 16 \
        --name ENet_baseline_penobscot \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 1472 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder Baseline_penobscot \
        --dataset 1 \
        --kernel 11 \
        --dropout 0 \
        --gpuID $GPU_ID 
    #ESPNet
        python ./train.py \
        --optimizer 0 \
        --model 7 \
        --epochs 100 \
        --batch_size 16 \
        --name ESPNet_baseline_penobscot \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 1472 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder Baseline_penobscot \
        --dataset 1 \
        --kernel 11 \
        --dropout 0 \
        --gpuID $GPU_ID 
    #ICNet
        python ./train.py \
        --optimizer 0 \
        --model 8 \
        --epochs 100 \
        --batch_size 16 \
        --name ICNet_baseline_penobscot \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 1472 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder Baseline_penobscot \
        --dataset 1 \
        --kernel 11 \
        --dropout 0 \
        --gpuID $GPU_ID 
    #EfficientNet B1
        python ./train.py \
        --optimizer 0 \
        --model 10 \
        --epochs 100 \
        --batch_size 16 \
        --name EfficientNetB1_baseline_penobscot \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 1472 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder Baseline_penobscot \
        --dataset 1 \
        --kernel 11 \
        --dropout 0 \
        --gpuID $GPU_ID 