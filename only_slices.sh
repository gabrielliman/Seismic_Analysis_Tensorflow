#!/bin/bash
readonly GPU_ID=1


#Attention baseline
# for num_extra_train in 5 15 25 35 45; do
for num_extra_train in 5 15 25 35 45; do
  python ./train.py \
    --num_extra_train $num_extra_train \
    --optimizer 0 \
    --model 2 \
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
    --folder attention_baseline_only_train_slices_penobscot \
    --dataset 9 \
    --kernel 11 \
    --dropout 0 \
    --gpuID $GPU_ID
done

#CPFNetM Parihaka
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

#CPFNetM Penobscot
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
    --folder CPFNetM_baseline_only_train_slices_penobscot \
    --dataset 9 \
    --kernel 11 \
    --dropout 0 \
    --gpuID $GPU_ID
done



#ENet Parihaka
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

#ENet Penobscot
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
    --folder ENet_baseline_only_train_slices_penobscot \
    --dataset 9 \
    --kernel 11 \
    --dropout 0 \
    --gpuID $GPU_ID
done


#ESPNet Parihaka
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

#ESPNet Penobscot
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
    --folder ESPNet_baseline_only_train_slices_penobscot \
    --dataset 9 \
    --kernel 11 \
    --dropout 0 \
    --gpuID $GPU_ID
done


#ICNet Parihaka
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

#ICNet Penobscot
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
    --folder ICNet_baseline_only_train_slices_penobscot \
    --dataset 9 \
    --kernel 11 \
    --dropout 0 \
    --gpuID $GPU_ID
done


#EfficientNetB1 Parihaka
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

#EfficientNetB1 Penobscot
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
    --folder EfficientNetB1_baseline_only_train_slices_penobscot \
    --dataset 9 \
    --kernel 11 \
    --dropout 0 \
    --gpuID $GPU_ID
done




#LWBAUnet Parihaka
for num_extra_train in 5 15 25 35 45; do
  python ./train.py \
    --num_extra_train $num_extra_train \
    --optimizer 0 \
    --model 11 \
    --epochs 100 \
    --batch_size 8 \
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
    --folder LWBAUnet_baseline_only_train_slices \
    --dataset 5 \
    --kernel 11 \
    --dropout 0 \
    --gpuID $GPU_ID
done

#LWBAUnet Penobscot
for num_extra_train in 5 15 25 35 45; do
  python ./train.py \
    --num_extra_train $num_extra_train \
    --optimizer 0 \
    --model 11 \
    --epochs 100 \
    --batch_size 8 \
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
    --folder LWBAUnet_baseline_only_train_slices_penobscot \
    --dataset 9 \
    --kernel 11 \
    --dropout 0 \
    --gpuID $GPU_ID
done



#BridgeNet Parihaka
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

#BridgeNet Penobscot
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
    --folder BridgeNet_baseline_only_train_slices_penobscot \
    --dataset 9 \
    --kernel 11 \
    --dropout 0 \
    --gpuID $GPU_ID
done


#UNet Penobscot
for num_extra_train in 5 15 25 35 45; do
  python ./train.py \
    --num_extra_train $num_extra_train \
    --optimizer 0 \
    --model 0 \
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
    --folder unet_baseline_only_train_slices_penobscot \
    --dataset 9 \
    --kernel 11 \
    --dropout 0 \
    --gpuID $GPU_ID
done

#UNet Penobscot
for num_extra_train in 5 15 25 35 45; do
  python ./train.py \
    --num_extra_train $num_extra_train \
    --optimizer 0 \
    --model 1 \
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
    --folder unet3+_baseline_only_train_slices_penobscot \
    --dataset 9 \
    --kernel 11 \
    --dropout 0 \
    --gpuID $GPU_ID
done

