#!/bin/bash
readonly GPU_ID=1

#python ./train.py --optimizer 0 --model 2 --epochs 100 --batch_size 4 --name parihaka_best100_articledivision --stride1 256 --stride2 128 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 5 --loss_function 1 --folder parihaka --dataset 2 --kernel 7 --dropout 0.5
python ./train.py --optimizer 0 --filters 6 --model 2 --epochs 100 --batch_size 4 --name parihaka_best_parameter_variation --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka --dataset 0 --kernel 9 --dropout 0.5 --gpuID $GPU_ID




python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --name parihaka_best100_mydivision --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder parihaka --dataset 0 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --name penobscot_best100_mydivision --stride1 256 --stride2 128 --slice_shape1 1024 --slice_shape2 192 --delta 1e-4 --patience 2 --loss_function 1 --folder penobscot --dataset 1 --kernel 7 --dropout 0.5
