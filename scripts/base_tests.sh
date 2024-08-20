#!/bin/bash
readonly GPU_ID=1
# feitos

# #batch size=16 stride=128x64 otimizador=Adam delta=1e-4 patience=15 loss_function=Cross Entropy kernel_size=11 dropout=0 Dataset=parihaka_minha_divisao
# #parihaka
python ./train.py --optimizer 0 --model 0 --epochs 100 --batch_size 16 --name base_unet --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka --dataset 0 --kernel 11 --dropout 0

python ./train.py --optimizer 0 --model 1 --epochs 100 --batch_size 16 --name base_unet3+ --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka --dataset 0 --kernel 11 --dropout 0

python ./train.py --optimizer 0 --model 2 --epochs 100 --batch_size 16 --name base_attention --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka --dataset 0 --kernel 11 --dropout 0


# #penobscot

python ./train.py --optimizer 0 --model 0 --epochs 100 --batch_size 16 --name base_unet_1607 --stride1 256 --stride2 64 --stridetest1 256  --stridetest2 64 --slice_shape1 1024 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder penobscot --dataset 1 --kernel 11 --dropout 0 --gpuID $GPU_ID

python ./train.py --optimizer 0 --model 1 --epochs 100 --batch_size 16 --name base_unet3+_1607 --stride1 256 --stride2 64 --stridetest1 256  --stridetest2 64 --slice_shape1 1024 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder penobscot --dataset 1 --kernel 11 --dropout 0 --gpuID $GPU_ID

python ./train.py --optimizer 0 --model 2 --epochs 100 --batch_size 16 --name base_attention_1607 --stride1 256 --stride2 64 --stridetest1 256  --stridetest2 64 --slice_shape1 1024 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder penobscot --dataset 1 --kernel 11 --dropout 0 --gpuID $GPU_ID