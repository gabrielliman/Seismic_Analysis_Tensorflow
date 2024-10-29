#!/bin/bash
readonly GPU_ID=0


#CPFNetM
# python ./train.py --optimizer 0 --model 4 --epochs 100 --batch_size 16 --name CPFNetM --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder new_networks_penobscot --dataset 1

#DCUNet ta dando falta de memoria
#python ./train.py --optimizer 0 --model 5 --epochs 100 --batch_size 16 --name DCUNet --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder new_networks_penobscot --dataset 1

#ENet
# python ./train.py --optimizer 0 --model 6 --epochs 100 --batch_size 16 --name ENet --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder new_networks_penobscot --dataset 1

#ESPNet
# python ./train.py --optimizer 0 --model 7 --epochs 100 --batch_size 16 --name ESPNet --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder new_networks_penobscot --dataset 1

#ICNet
# python ./train.py --optimizer 0 --model 8 --epochs 100 --batch_size 16 --name ICNet --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder new_networks_penobscot --dataset 1

#MultiResUnet ta faltando memoria
#python ./train.py --optimizer 0 --model 9 --epochs 100 --batch_size 16 --name MultiResUnet --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder new_networks_penobscot --dataset 1

#EfficientNet B1
#python ./train.py --optimizer 0 --model 10 --epochs 100 --batch_size 16 --name EfficientNetB1 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder new_networks --dataset 0 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --model 10 --epochs 100 --batch_size 16 --name EfficientNetB1 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder new_networks_penobscot --dataset 1 --gpuID $GPU_ID
