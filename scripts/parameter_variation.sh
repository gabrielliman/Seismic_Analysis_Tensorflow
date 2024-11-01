#!/bin/bash
readonly GPU_ID=1

#kernel size
#python ./train.py  --kernel 3 --optimizer 0 --model 2 --epochs 100 --batch_size 16 --name attention_kernel3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka_kernel --dataset 0 --dropout 0 --gpuID $GPU_ID
#python ./train.py  --kernel 5 --optimizer 0 --model 2 --epochs 100 --batch_size 16 --name attention_kernel5 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka_kernel --dataset 0 --dropout 0 --gpuID $GPU_ID
#python ./train.py  --kernel 7 --optimizer 0 --model 2 --epochs 100 --batch_size 16 --name attention_kernel7 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka_kernel --dataset 0 --dropout 0 --gpuID $GPU_ID
#python ./train.py  --kernel 9 --optimizer 0 --model 2 --epochs 100 --batch_size 16 --name attention_kernel9 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka_kernel --dataset 0 --dropout 0 --gpuID $GPU_ID

#batch_size
#python ./train.py --batch_size 2 --optimizer 0 --model 2 --epochs 100 --name attention_batch2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka_batch --dataset 0 --kernel 11 --dropout 0 --gpuID $GPU_ID
#python ./train.py --batch_size 4 --optimizer 0 --model 2 --epochs 100 --name attention_batch4 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka_batch --dataset 0 --kernel 11 --dropout 0 --gpuID $GPU_ID
#python ./train.py --batch_size 8 --optimizer 0 --model 2 --epochs 100 --name attention_batch8 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka_batch --dataset 0 --kernel 11 --dropout 0 --gpuID $GPU_ID


#Camadas
#python ./train.py --filters 1 --optimizer 0 --model 2 --epochs 100 --batch_size 16 --name attention_camada1 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka_camada --dataset 0 --kernel 11 --dropout 0 --gpuID $GPU_ID
#python ./train.py --filters 2 --optimizer 0 --model 2 --epochs 100 --batch_size 16 --name attention_camada2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka_camada --dataset 0 --kernel 11 --dropout 0 --gpuID $GPU_ID 
#python ./train.py --filters 3 --optimizer 0 --model 2 --epochs 100 --batch_size 16 --name attention_camada3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka_camada --dataset 0 --kernel 11 --dropout 0 --gpuID $GPU_ID
#python ./train.py --filters 4 --optimizer 0 --model 2 --epochs 100 --batch_size 16 --name attention_camada4 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka_camada --dataset 0 --kernel 11 --dropout 0 --gpuID $GPU_ID
#python ./train.py --filters 5 --optimizer 0 --model 2 --epochs 100 --batch_size 16 --name attention_camada5 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka_camada --dataset 0 --kernel 11 --dropout 0 --gpuID $GPU_ID

#Loss

#python ./train.py --loss_function 1 --gamma 2 --optimizer 0 --model 2 --epochs 100 --batch_size 16 --name attention_focal2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --folder parihaka_loss --dataset 0 --kernel 11 --dropout 0 --gpuID $GPU_ID
#python ./train.py --loss_function 1 --gamma 4 --optimizer 0 --model 2 --epochs 100 --batch_size 16 --name attention_focal4 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --folder parihaka_loss --dataset 0 --kernel 11 --dropout 0 --gpuID $GPU_ID
#python ./train.py --loss_function 1 --gamma 8 --optimizer 0 --model 2 --epochs 100 --batch_size 16 --name attention_focal8 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --folder parihaka_loss --dataset 0 --kernel 11 --dropout 0 --gpuID $GPU_ID


#Dropout

#python ./train.py --dropout 0.25 --optimizer 0 --model 2 --epochs 100 --batch_size 16 --name attention_dropout25 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka_dropout --dataset 0 --kernel 11
#python ./train.py --dropout 0.5 --optimizer 0 --model 2 --epochs 100 --batch_size 16 --name attention_dropout50 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka_dropout --dataset 0 --kernel 11
#python ./train.py --dropout 0.75 --optimizer 0 --model 2 --epochs 100 --batch_size 16 --name attention_dropout75 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka_dropout --dataset 0 --kernel 11
