#!/bin/bash
readonly GPU_ID=1
#Penobscot 192x192 279x219
for Iteration in {1..5}; do
        for sizex_sizey in "192x192" "279x219"; do
            sizex=${sizex_sizey%x*}
            sizey=${sizex_sizey#*x}
            #UNet
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 0 \
                --epochs 100 \
                --batch_size 16 \
                --name "UNet_$sizex_sizey" \
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder "RPRV_Penobscot_$Iteration" \
                --dataset 3 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
            #BridgeNet 
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 3 \
                --epochs 100 \
                --batch_size 16 \
                --name "BridgeNet_$sizex_sizey"\
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder "RPRV_Penobscot_$Iteration" \
                --dataset 3 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
            #UNet3+ 
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 1 \
                --epochs 100 \
                --batch_size 16 \
                --name "UNet3+_$sizex_sizey"\
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder "RPRV_Penobscot_$Iteration" \
                --dataset 3 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
            #Attention 
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 2 \
                --epochs 100 \
                --batch_size 16 \
                --name "Attention_$sizex_sizey"\
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder "RPRV_Penobscot_$Iteration" \
                --dataset 3 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
            #ESPNet 
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 7 \
                --epochs 100 \
                --batch_size 16 \
                --name "ESPNet_$sizex_sizey"\
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder "RPRV_Penobscot_$Iteration" \
                --dataset 3 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
            #ENet    
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 6 \
                --epochs 100 \
                --batch_size 16 \
                --name "ENet_$sizex_sizey"\
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder "RPRV_Penobscot_$Iteration" \
                --dataset 3 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
            #ICNet 
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 8 \
                --epochs 100 \
                --batch_size 16 \
                --name "ICNet_$sizex_sizey"\
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder "RPRV_Penobscot_$Iteration" \
                --dataset 3 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
            #CFPNetM 
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 4 \
                --epochs 100 \
                --batch_size 16 \
                --name "CFPNet_$sizex_sizey"\
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder "RPRV_Penobscot_$Iteration" \
                --dataset 3 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
            #LWBNA 
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 11 \
                --epochs 100 \
                --batch_size 16 \
                --name "LWBNA_$sizex_sizey"\
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder "RPRV_Penobscot_$Iteration" \
                --dataset 3 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
            #EfficientNetB1 
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 10 \
                --epochs 100 \
                --batch_size 16 \
                --name "EfficientNetB1_$sizex_sizey"\
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder "RPRV_Penobscot_$Iteration" \
                --dataset 3 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
        done
done