#!/bin/bash
readonly GPU_ID=1
#LRP DONE 
    #BridgeNet DONE
        python ./train.py \
        --optimizer 0 \
        --model 3 \
        --epochs 100 \
        --batch_size 16 \
        --name BridgeNet_baseline \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 992 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder BridgeNet_baseline \
        --dataset 0 \
        --kernel 11 \
        --dropout 0 \
        --gpuID $GPU_ID

#RPRV DONE
    #BridgeNet  DONE
        for sizex_sizey in "192x192" "235x216" "278x240" "321x264"; do
            sizex=${sizex_sizey%x*}
            sizey=${sizex_sizey#*x}
            
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 3 \
                --epochs 100 \
                --batch_size 16 \
                --name "${sizex}_${sizey}" \
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder BridgeNet_baseline_limited_end \
                --dataset 3 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
        done

    #UNet3+  DONE
        for sizex_sizey in "321x264"; do
            sizex=${sizex_sizey%x*}
            sizey=${sizex_sizey#*x}
            
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 1 \
                --epochs 100 \
                --batch_size 16 \
                --name "${sizex}_${sizey}" \
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder unet3+_baseline_limited_end \
                --dataset 3 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
        done

    #ESPNet  DONE
        for sizex_sizey in "192x192" "235x216" "278x240" "321x264"; do
            sizex=${sizex_sizey%x*}
            sizey=${sizex_sizey#*x}
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 7 \
                --epochs 100 \
                --batch_size 16 \
                --name "${sizex}_${sizey}" \
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder ESPNet_baseline_limited_end \
                --dataset 3 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
        done

    #ENet  DONE
        for sizex_sizey in "192x192" "235x216" "278x240" "321x264"; do
            sizex=${sizex_sizey%x*}
            sizey=${sizex_sizey#*x}
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 6 \
                --epochs 100 \
                --batch_size 16 \
                --name "${sizex}_${sizey}" \
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder ENet_baseline_limited_end \
                --dataset 3 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
        done

    #ICNet  DONE
        for sizex_sizey in "192x192" "235x216" "278x240" "321x264"; do
            sizex=${sizex_sizey%x*}
            sizey=${sizex_sizey#*x}
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 8 \
                --epochs 100 \
                --batch_size 16 \
                --name "${sizex}_${sizey}" \
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder ICNet_baseline_limited_end \
                --dataset 3 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
        done

    #CPFNetM  DONE
        for sizex_sizey in "192x192" "235x216" "278x240" "321x264"; do
            sizex=${sizex_sizey%x*}
            sizey=${sizex_sizey#*x}
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 4 \
                --epochs 100 \
                --batch_size 16 \
                --name "${sizex}_${sizey}" \
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder CPFNetM_baseline_limited_end \
                --dataset 3 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
        done

    #LWBNA DONE
        for sizex_sizey in "192x192" "235x216" "278x240" "321x264"; do
            sizex=${sizex_sizey%x*}
            sizey=${sizex_sizey#*x}
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 11 \
                --epochs 100 \
                --batch_size 16 \
                --name "${sizex}_${sizey}" \
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder LWBNA_baseline_limited_end \
                --dataset 3 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
        done
    
    #EfficientNetB1 DONE
        for sizex_sizey in "192x192" "235x216" "278x240" "321x264"; do
            sizex=${sizex_sizey%x*}
            sizey=${sizex_sizey#*x}
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 10 \
                --epochs 100 \
                --batch_size 16 \
                --name "${sizex}_${sizey}" \
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder EfficientNetB1_baseline_limited_end \
                --dataset 3 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
        done
#RPSE
    #BridgeNet DONE
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
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder BridgeNet_baseline_192x192_with_train_slices \
            --dataset 4 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 3 \
            --sizetrainx 321 \
            --sizetrainy 264 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder BridgeNet_baseline_321x264_with_train_slices \
            --dataset 4 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done

    #UNet DONE
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 0 \
            --sizetrainx 321 \
            --sizetrainy 264 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder UNet_baseline_321x264_with_train_slices \
            --dataset 4 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
    #UNet3+ DONE
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 1 \
            --sizetrainx 321 \
            --sizetrainy 264 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder UNet3+_baseline_321x264_with_train_slices \
            --dataset 4 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
    #Attention DONE
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 2 \
            --sizetrainx 321 \
            --sizetrainy 264 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder Attention_baseline_321x264_with_train_slices \
            --dataset 4 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
    #ESPNet DONE
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 7 \
            --sizetrainx 321 \
            --sizetrainy 264 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder ESPNet_baseline_321x264_with_train_slices \
            --dataset 4 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
    #ENet DONE
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 6 \
            --sizetrainx 321 \
            --sizetrainy 264 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder ENet_baseline_321x264_with_train_slices \
            --dataset 4 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
    #ICNet DONE
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 8 \
            --sizetrainx 321 \
            --sizetrainy 264 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder ICNet_baseline_321x264_with_train_slices \
            --dataset 4 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
    #CPFNetM DONE
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 4 \
            --sizetrainx 321 \
            --sizetrainy 264 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder CPFNetM_baseline_321x264_with_train_slices \
            --dataset 4 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
    #LWBNA
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
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder LWBNA_baseline_192x192_with_train_slices \
            --dataset 4 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 11 \
            --sizetrainx 321 \
            --sizetrainy 264 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder LWBNA_baseline_321x264_with_train_slices \
            --dataset 4 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
    #EfficientNetB1 DONE
        for num_extra_train in {2,8,32}; do
            python ./train.py \
            --num_extra_train $num_extra_train \
            --optimizer 0 \
            --model 10 \
            --sizetrainx 321 \
            --sizetrainy 264 \
            --epochs 100 \
            --batch_size 16 \
            --name "$num_extra_train"_slice \
            --stride1 128 \
            --stride2 64 \
            --stridetest1 128 \
            --stridetest2 64 \
            --slice_shape1 992 \
            --slice_shape2 192 \
            --delta 1e-4 \
            --patience 5 \
            --loss_function 0 \
            --folder EfficientNetB1_baseline_321x264_with_train_slices \
            --dataset 4 \
            --kernel 11 \
            --dropout 0 \
            --gpuID $GPU_ID
        done
#EDS
    #BridgeNet
        for num_extra_train in 5 15 25 35 45; do
            python ./train.py \
                --num_extra_train $num_extra_train \
                --optimizer 0 \
                --model 3 \
                --epochs 100 \
                --batch_size 16 \
                --name "$num_extra_train"_slice \
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder BridgeNet_baseline_only_train_slices \
                --dataset 5 \
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
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder LWBNA_baseline_only_train_slices \
                --dataset 5 \
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
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder CPFNetM_baseline_only_train_slices \
                --dataset 5 \
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
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder ENet_baseline_only_train_slices \
                --dataset 5 \
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
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder ESPNet_baseline_only_train_slices \
                --dataset 5 \
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
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder ICNet_baseline_only_train_slices \
                --dataset 5 \
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
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 992 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder EfficientNetB1_baseline_only_train_slices \
                --dataset 5 \
                --kernel 11 \
                --dropout 0 \
                --gpuID $GPU_ID
        done