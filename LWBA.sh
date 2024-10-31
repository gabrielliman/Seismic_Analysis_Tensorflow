#!/bin/bash
readonly GPU_ID=1

python ./train.py \
  --optimizer 0 \
  --model 8 \
  --epochs 100 \
  --batch_size 8 \
  --name LWBAUnet_baseline_penobscot \
  --stride1 128 \
  --stride2 64 \
  --stridetest1 128 \
  --stridetest2 64 \
  --slice_shape1 992 \
  --slice_shape2 192 \
  --delta 1e-4 \
  --patience 5 \
  --loss_function 0 \
  --folder LWBAUnet_baseline_penobscot \
  --dataset 1 \
  --kernel 11 \
  --dropout 0 \
  --gpuID $GPU_ID

for num_extra_train in {2,8,32}; do
python ./train.py \
  --num_extra_train $num_extra_train \
  --optimizer 0 \
  --model 8 \
  --sizetrainx 192 \
  --sizetrainy 192 \
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
  --folder LWBAUnet_baseline_192x192_with_train_slices_penobscot \
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
  --sizetrainx 192 \
  --sizetrainy 192 \
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
  --folder LWBAUnet_baseline_192x192_with_train_slices \
  --dataset 4 \
  --kernel 11 \
  --dropout 0 \
  --gpuID $GPU_ID
done