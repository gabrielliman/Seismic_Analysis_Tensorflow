#!/bin/bash
readonly GPU_ID=1
#192x192 [2,8,32] 279x219 [2,8,32]
    #Attention
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 2 \
            --sizetrainx 192 \
            --sizetrainy 192 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 16 \
            --stride2 16 \
            --stridetest1 128 \
            --stridetest2 192 \
            --slice_shape1 1472 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder Attention_192x192_with_train_slices_penobscot \
            --dataset 8 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 2 \
            --sizetrainx 279 \
            --sizetrainy 219 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 16 \
            --stride2 16 \
            --stridetest1 128 \
            --stridetest2 192 \
            --slice_shape1 1472 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder Attention_279x219_with_train_slices_penobscot \
            --dataset 8 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
    #BridgeNet
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 3 \
            --sizetrainx 192 \
            --sizetrainy 192 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 16 \
            --stride2 16 \
            --stridetest1 128 \
            --stridetest2 192 \
            --slice_shape1 1472 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder BridgeNet_192x192_with_train_slices_penobscot \
            --dataset 8 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 3 \
            --sizetrainx 279 \
            --sizetrainy 219 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 16 \
            --stride2 16 \
            --stridetest1 128 \
            --stridetest2 192 \
            --slice_shape1 1472 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder BridgeNet_279x219_with_train_slices_penobscot \
            --dataset 8 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
    #UNet
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 0 \
            --sizetrainx 192 \
            --sizetrainy 192 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 16 \
            --stride2 16 \
            --stridetest1 128 \
            --stridetest2 192 \
            --slice_shape1 1472 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder UNet_192x192_with_train_slices_penobscot \
            --dataset 8 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 0 \
            --sizetrainx 279 \
            --sizetrainy 219 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 16 \
            --stride2 16 \
            --stridetest1 128 \
            --stridetest2 192 \
            --slice_shape1 1472 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder UNet_279x219_with_train_slices_penobscot \
            --dataset 8 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
    #Unet3+
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 1 \
            --sizetrainx 192 \
            --sizetrainy 192 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 16 \
            --stride2 16 \
            --stridetest1 128 \
            --stridetest2 192 \
            --slice_shape1 1024 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder UNet3+_192x192_with_train_slices_penobscot \
            --dataset 8 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 1 \
            --sizetrainx 279 \
            --sizetrainy 219 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 16 \
            --stride2 16 \
            --stridetest1 128 \
            --stridetest2 192 \
            --slice_shape1 1024 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder UNet3+_279x219_with_train_slices_penobscot \
            --dataset 8 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
    #LWBAUNET
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 11 \
            --sizetrainx 192 \
            --sizetrainy 192 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 16 \
            --stride2 16 \
            --stridetest1 128 \
            --stridetest2 192 \
            --slice_shape1 1024 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder LWBNA_192x192_with_train_slices_penobscot \
            --dataset 8 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 11 \
            --sizetrainx 279 \
            --sizetrainy 219 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 16 \
            --stride2 16 \
            --stridetest1 128 \
            --stridetest2 192 \
            --slice_shape1 1024 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder LWBNA_279x219_with_train_slices_penobscot \
            --dataset 8 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
    #CPFNetM
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 4 \
            --sizetrainx 192 \
            --sizetrainy 192 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 16 \
            --stride2 16 \
            --stridetest1 128 \
            --stridetest2 192 \
            --slice_shape1 1472 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder CPFNetM_192x192_with_train_slices_penobscot \
            --dataset 8 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 4 \
            --sizetrainx 279 \
            --sizetrainy 219 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 16 \
            --stride2 16 \
            --stridetest1 128 \
            --stridetest2 192 \
            --slice_shape1 1472 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder CPFNetM_279x219_with_train_slices_penobscot \
            --dataset 8 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
    #ENet
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 6 \
            --sizetrainx 192 \
            --sizetrainy 192 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 16 \
            --stride2 16 \
            --stridetest1 128 \
            --stridetest2 192 \
            --slice_shape1 1472 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder ENet_192x192_with_train_slices_penobscot \
            --dataset 8 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 6 \
            --sizetrainx 279 \
            --sizetrainy 219 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 16 \
            --stride2 16 \
            --stridetest1 128 \
            --stridetest2 192 \
            --slice_shape1 1472 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder ENet_279x219_with_train_slices_penobscot \
            --dataset 8 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
    #ESPNet
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 7 \
            --sizetrainx 192 \
            --sizetrainy 192 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 16 \
            --stride2 16 \
            --stridetest1 128 \
            --stridetest2 192 \
            --slice_shape1 1472 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder ESPNet_192x192_with_train_slices_penobscot \
            --dataset 8 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 7 \
            --sizetrainx 279 \
            --sizetrainy 219 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 16 \
            --stride2 16 \
            --stridetest1 128 \
            --stridetest2 192 \
            --slice_shape1 1472 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder ESPNet_279x219_with_train_slices_penobscot \
            --dataset 8 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
    #ICNet
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 8 \
            --sizetrainx 192 \
            --sizetrainy 192 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 16 \
            --stride2 16 \
            --stridetest1 128 \
            --stridetest2 192 \
            --slice_shape1 1472 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder ICNet_192x192_with_train_slices_penobscot \
            --dataset 8 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 8 \
            --sizetrainx 279 \
            --sizetrainy 219 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 16 \
            --stride2 16 \
            --stridetest1 128 \
            --stridetest2 192 \
            --slice_shape1 1472 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder ICNet_279x219_with_train_slices_penobscot \
            --dataset 8 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
    #EfficientNet B1
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 10 \
            --sizetrainx 192 \
            --sizetrainy 192 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 16 \
            --stride2 16 \
            --stridetest1 128 \
            --stridetest2 192 \
            --slice_shape1 1472 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder EfficientNetB1_192x192_with_train_slices_penobscot \
            --dataset 8 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 10 \
            --sizetrainx 279 \
            --sizetrainy 219 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 16 \
            --stride2 16 \
            --stridetest1 128 \
            --stridetest2 192 \
            --slice_shape1 1472 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder EfficientNetB1_279x219_with_train_slices_penobscot \
            --dataset 8 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done