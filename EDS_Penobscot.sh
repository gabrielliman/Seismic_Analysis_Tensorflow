#!/bin/bash
readonly GPU_ID=1
#Penobscot
for Iteration in {1..5}; do
    for slices in 5 15 25 35 45; do
        #UNet
        python ./train.py \
            --num_extra_train $slices \
            --optimizer 0 \
            --model 0 \
            --epochs 100 \
            --batch_size 16 \
            --name UNet_${slices} \
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 15 \
            --loss_function 0 \
            --folder "EDS_Penobscot_$Iteration" \
            --dataset 7 \
            --kernel 11 \
            --dropout 0 \
            --gpuID "$GPU_ID"
        #BridgeNet 
        python ./train.py \
            --num_extra_train $slices \
            --optimizer 0 \
            --model 3 \
            --epochs 100 \
            --batch_size 16 \
            --name BridgeNet_${slices}\
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 15 \
            --loss_function 0 \
            --folder "EDS_Penobscot_$Iteration" \
            --dataset 7 \
            --kernel 11 \
            --dropout 0 \
            --gpuID "$GPU_ID"
        #UNet3+ 
        python ./train.py \
            --num_extra_train $slices \
            --optimizer 0 \
            --model 1 \
            --epochs 100 \
            --batch_size 16 \
            --name UNet3+_${slices}\
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 15 \
            --loss_function 0 \
            --folder "EDS_Penobscot_$Iteration" \
            --dataset 7 \
            --kernel 11 \
            --dropout 0 \
            --gpuID "$GPU_ID"
        #Attention 
        python ./train.py \
            --num_extra_train $slices \
            --optimizer 0 \
            --model 2 \
            --epochs 100 \
            --batch_size 16 \
            --name Attention_${slices}\
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 15 \
            --loss_function 0 \
            --folder "EDS_Penobscot_$Iteration" \
            --dataset 7 \
            --kernel 11 \
            --dropout 0 \
            --gpuID "$GPU_ID"
        #ESPNet 
        python ./train.py \
            --num_extra_train $slices \
            --optimizer 0 \
            --model 7 \
            --epochs 100 \
            --batch_size 16 \
            --name ESPNet_${slices}\
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 15 \
            --loss_function 0 \
            --folder "EDS_Penobscot_$Iteration" \
            --dataset 7 \
            --kernel 11 \
            --dropout 0 \
            --gpuID "$GPU_ID"
        #ENet    
        python ./train.py \
            --num_extra_train $slices \
            --optimizer 0 \
            --model 6 \
            --epochs 100 \
            --batch_size 16 \
            --name ENet_${slices}\
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 15 \
            --loss_function 0 \
            --folder "EDS_Penobscot_$Iteration" \
            --dataset 7 \
            --kernel 11 \
            --dropout 0 \
            --gpuID "$GPU_ID"
        #ICNet 
        python ./train.py \
            --num_extra_train $slices \
            --optimizer 0 \
            --model 8 \
            --epochs 100 \
            --batch_size 16 \
            --name ICNet_${slices}\
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 15 \
            --loss_function 0 \
            --folder "EDS_Penobscot_$Iteration" \
            --dataset 7 \
            --kernel 11 \
            --dropout 0 \
            --gpuID "$GPU_ID"
        #CFPNetM 
        python ./train.py \
            --num_extra_train $slices \
            --optimizer 0 \
            --model 4 \
            --epochs 100 \
            --batch_size 16 \
            --name CFPNet_${slices}\
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 15 \
            --loss_function 0 \
            --folder "EDS_Penobscot_$Iteration" \
            --dataset 7 \
            --kernel 11 \
            --dropout 0 \
            --gpuID "$GPU_ID"
        #LWBNA 
        python ./train.py \
            --num_extra_train $slices \
            --optimizer 0 \
            --model 11 \
            --epochs 100 \
            --batch_size 16 \
            --name LWBNA_${slices}\
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 15 \
            --loss_function 0 \
            --folder "EDS_Penobscot_$Iteration" \
            --dataset 7 \
            --kernel 11 \
            --dropout 0 \
            --gpuID "$GPU_ID"
        #EfficientNetB1 
        python ./train.py \
            --num_extra_train $slices \
            --optimizer 0 \
            --model 10 \
            --epochs 100 \
            --batch_size 16 \
            --name EfficientNetB1_${slices}\
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 15 \
            --loss_function 0 \
            --folder "EDS_Penobscot_$Iteration" \
            --dataset 7 \
            --kernel 11 \
            --dropout 0 \
            --gpuID "$GPU_ID"
    done
done