#!/bin/bash
readonly GPU_ID=1
for Iteration in {1..5}; do
    #BridgeNet
        python ./train.py \
        --optimizer 0 \
        --model 3 \
        --epochs 100 \
        --batch_size 16 \
        --name BridgeNet \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 992 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder LRP_Penobscot_$Iteration \
        --dataset 1 \
        --kernel 11 \
        --dropout 0 \
        --gpuID $GPU_ID 

    #LWBAUNET 
        python ./train.py \
        --optimizer 0 \
        --model 11 \
        --epochs 100 \
        --batch_size 8 \
        --name LWBNA \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 992 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder LRP_Penobscot_$Iteration \
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
        --name Attention \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 992 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder LRP_Penobscot_$Iteration \
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
        --name UNet \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 992 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder LRP_Penobscot_$Iteration \
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
        --name UNet3+ \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 992 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder LRP_Penobscot_$Iteration \
        --dataset 1 \
        --kernel 11 \
        --dropout 0 \
        --gpuID $GPU_ID 
    #CFPNetM
        python ./train.py \
        --optimizer 0 \
        --model 4 \
        --epochs 100 \
        --batch_size 16 \
        --name CFPNetM \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 992 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder LRP_Penobscot_$Iteration \
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
        --name ENet \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 992 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder LRP_Penobscot_$Iteration \
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
        --name ESPNet \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 992 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder LRP_Penobscot_$Iteration \
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
        --name ICNet \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 992 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder LRP_Penobscot_$Iteration \
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
        --name EfficientNetB1 \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 992 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder LRP_Penobscot_$Iteration \
        --dataset 1 \
        --kernel 11 \
        --dropout 0 \
        --gpuID $GPU_ID 
done