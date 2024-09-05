#!/bin/bash
readonly GPU_ID=1

#192x192
#python ./train.py --num_extra_train 1 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 192 --sizetrainy 192 --name 192x192_1_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 3 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 192 --sizetrainy 192 --name 192x192_3_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 7 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 192 --sizetrainy 192 --name 192x192_7_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 15 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 192 --sizetrainy 192 --name 192x192_15_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 31 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 192 --sizetrainy 192 --name 192x192_31_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 63 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 192 --sizetrainy 192 --name 192x192_63_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#235x216
#python ./train.py --num_extra_train 1 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 235 --sizetrainy 216 --name 235x216_1_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 3 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 235 --sizetrainy 216 --name 235x216_3_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 7 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 235 --sizetrainy 216 --name 235x216_7_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 15 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 235 --sizetrainy 216 --name 235x216_15_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 31 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 235 --sizetrainy 216 --name 235x216_31_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 63 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 235 --sizetrainy 216 --name 235x216_63_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#278x240
#python ./train.py --num_extra_train 1 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 278 --sizetrainy 240 --name 278x240_1_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 3 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 278 --sizetrainy 240 --name 278x240_3_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 7 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 278 --sizetrainy 240 --name 278x240_7_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 15 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 278 --sizetrainy 240 --name 278x240_15_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 31 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 278 --sizetrainy 240 --name 278x240_31_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 63 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 278 --sizetrainy 240 --name 278x240_63_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#321x264
#python ./train.py --num_extra_train 1 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 321 --sizetrainy 264 --name 321x264_1_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 3 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 321 --sizetrainy 264 --name 321x264_3_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 7 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 321 --sizetrainy 264 --name 321x264_7_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 15 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 321 --sizetrainy 264 --name 321x264_15_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 31 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 321 --sizetrainy 264 --name 321x264_31_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 63 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 321 --sizetrainy 264 --name 321x264_63_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#364x288
#python ./train.py --num_extra_train 1 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 364 --sizetrainy 288 --name 364x288_1_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 3 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 364 --sizetrainy 288 --name 364x288_3_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 7 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 364 --sizetrainy 288 --name 364x288_7_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 15 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 364 --sizetrainy 288 --name 364x288_15_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 31 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 364 --sizetrainy 288 --name 364x288_31_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 63 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 364 --sizetrainy 288 --name 364x288_63_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#407x312
#python ./train.py --num_extra_train 1 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 407 --sizetrainy 312 --name 407x312_1_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 3 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 407 --sizetrainy 312 --name 407x312_3_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 7 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 407 --sizetrainy 312 --name 407x312_7_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 15 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 407 --sizetrainy 312 --name 407x312_15_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 31 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 407 --sizetrainy 312 --name 407x312_31_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 63 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 407 --sizetrainy 312 --name 407x312_63_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#450x336
#python ./train.py --num_extra_train 1 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 450 --sizetrainy 336 --name 450x336_1_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 3 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 450 --sizetrainy 336 --name 450x336_3_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 7 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 450 --sizetrainy 336 --name 450x336_7_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 15 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 450 --sizetrainy 336 --name 450x336_15_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 31 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 450 --sizetrainy 336 --name 450x336_31_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 63 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 450 --sizetrainy 336 --name 450x336_63_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#493x360
#python ./train.py --num_extra_train 1 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 493 --sizetrainy 360 --name 493x360_1_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 3 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 493 --sizetrainy 360 --name 493x360_3_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 7 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 493 --sizetrainy 360 --name 493x360_7_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 15 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 493 --sizetrainy 360 --name 493x360_15_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 31 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 493 --sizetrainy 360 --name 493x360_31_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 63 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 493 --sizetrainy 360 --name 493x360_63_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#536x384
#python ./train.py --num_extra_train 1 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 536 --sizetrainy 384 --name 536x384_1_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 3 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 536 --sizetrainy 384 --name 536x384_3_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 7 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 536 --sizetrainy 384 --name 536x384_7_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 15 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 536 --sizetrainy 384 --name 536x384_15_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 31 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 536 --sizetrainy 384 --name 536x384_31_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 63 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 536 --sizetrainy 384 --name 536x384_63_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID



#579x408
#python ./train.py --num_extra_train 1 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 579 --sizetrainy 408 --name 579x408_1_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 3 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 579 --sizetrainy 408 --name 579x408_3_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 7 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 579 --sizetrainy 408 --name 579x408_7_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --num_extra_train 15 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 579 --sizetrainy 408 --name 579x408_15_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --num_extra_train 31 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 579 --sizetrainy 408 --name 579x408_31_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --num_extra_train 63 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 579 --sizetrainy 408 --name 579x408_63_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#622x430


python ./train.py --num_extra_train 1 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 622 --sizetrainy 430 --name 622x430_1_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --num_extra_train 3 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 622 --sizetrainy 430 --name 622x430_3_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --num_extra_train 7 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 622 --sizetrainy 430 --name 622x430_7_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --num_extra_train 15 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 622 --sizetrainy 430 --name 622x430_15_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --num_extra_train 31 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 622 --sizetrainy 430 --name 622x430_31_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --num_extra_train 63 --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 622 --sizetrainy 430 --name 622x430_63_slice --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder train_slices_in_test_area --dataset 4 --kernel 7 --dropout 0.5 --gpuID $GPU_ID
