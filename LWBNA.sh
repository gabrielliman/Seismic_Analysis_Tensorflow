#!/bin/bash
readonly GPU_ID=0
:'
#LRP_Parihaka
    python ./train.py \
        --optimizer 0 \
        --model 11 \
        --epochs 100 \
        --batch_size 16 \
        --name LWBNA_1 \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 992 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder Models_LRP_Parihaka \
        --dataset 0 \
        --kernel 11 \
        --dropout 0 \
        --gpuID $GPU_ID 
'
#LRP_Penobscot
    python ./train.py \
        --optimizer 0 \
        --model 11 \
        --epochs 100 \
        --batch_size 8 \
        --name LWBNA_1 \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 992 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 5 \
        --loss_function 0 \
        --folder Models_LRP_Penobscot \
        --dataset 1 \
        --kernel 11 \
        --dropout 0 \
        --gpuID $GPU_ID 

#RPRV_Parihaka
    python ./train.py \
        --test_pos "end" \
        --sizetrainx 192 \
        --sizetrainy 192 \
        --optimizer 0 \
        --model 11 \
        --epochs 100 \
        --batch_size 16 \
        --name LWBNA_1 \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 992 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 15 \
        --loss_function 0 \
        --folder Models_RPRV_192x192_Parihaka \
        --dataset 2 \
        --kernel 11 \
        --dropout 0 \
        --gpuID $GPU_ID
#RPRV_Penobscot
    python ./train.py \
        --test_pos "end" \
        --sizetrainx 192 \
        --sizetrainy 192 \
        --optimizer 0 \
        --model 11 \
        --epochs 100 \
        --batch_size 16 \
        --name LWBNA_1 \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 992 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 15 \
        --loss_function 0 \
        --folder Models_RPRV_192x192_Penobscot \
        --dataset 3 \
        --kernel 11 \
        --dropout 0 \
        --gpuID "$GPU_ID"
#RPEDS_Parihaka
    python ./train.py \
        --num_extra_train 8 \
        --optimizer 0 \
        --model 11 \
        --sizetrainx 192 \
        --sizetrainy 192 \
        --epochs 100 \
        --batch_size 16 \
        --name LWBNA_1 \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 192 \
        --slice_shape1 992 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 15 \
        --loss_function 0 \
        --folder Models_RPEDS_Parihaka \
        --dataset 4 \
        --kernel 11 \
        --dropout 0 \
        --gpuID 0
#RPEDS_Penobscot
    python ./train.py \
        --num_extra_train 8 \
        --optimizer 0 \
        --model 11 \
        --sizetrainx 192 \
        --sizetrainy 192 \
        --epochs 100 \
        --batch_size 16 \
        --name LWBNA_1 \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 192 \
        --slice_shape1 992 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 15 \
        --loss_function 0 \
        --folder Models_RPEDS_Penobscot \
        --dataset 5 \
        --kernel 11 \
        --dropout 0 \
        --gpuID $GPU_ID
#EDS_Parihaka
    python ./train.py \
        --num_extra_train 45 \
        --optimizer 0 \
        --model 11 \
        --epochs 100 \
        --batch_size 16 \
        --name LWBNA_1 \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 992 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 15 \
        --loss_function 0 \
        --folder "Models_EDS_Parihaka" \
        --dataset 6 \
        --kernel 11 \
        --dropout 0 \
        --gpuID $GPU_ID
#EDS_Penobscot
    python ./train.py \
        --num_extra_train 45 \
        --optimizer 0 \
        --model 11 \
        --epochs 100 \
        --batch_size 16 \
        --name LWBNA_1 \
        --stride1 128 \
        --stride2 64 \
        --stridetest1 128 \
        --stridetest2 64 \
        --slice_shape1 992 \
        --slice_shape2 192 \
        --delta 1e-4 \
        --patience 15 \
        --loss_function 0 \
        --folder "Models_EDS_Penobscot" \
        --dataset 7 \
        --kernel 11 \
        --dropout 0 \
        --gpuID $GPU_ID



#UNET3+ QUE BAGUNCEI
    python ./train.py \
        --num_extra_train 45 \
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
        --patience 15 \
        --loss_function 0 \
        --folder "Models_EDS_Parihaka" \
        --dataset 6 \
        --kernel 11 \
        --dropout 0 \
        --gpuID $GPU_ID