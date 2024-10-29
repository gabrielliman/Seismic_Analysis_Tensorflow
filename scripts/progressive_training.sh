#!/bin/bash
readonly GPU_ID=1

#192x192
#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 192 --sizetrainy 192 --test_pos "start" --name 192x192_start --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 192 --sizetrainy 192 --test_pos "mid" --name 192x192_mid --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 192 --sizetrainy 192 --test_pos "end" --name 192x192_end --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#235x216
#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 235 --sizetrainy 216 --test_pos "start" --name 235x216_start --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 235 --sizetrainy 216 --test_pos "mid" --name 235x216_mid --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 235 --sizetrainy 216 --test_pos "end" --name 235x216_end --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#278x240
#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 278 --sizetrainy 240 --test_pos "start" --name 278x240_start --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 278 --sizetrainy 240 --test_pos "mid" --name 278x240_mid --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 278 --sizetrainy 240 --test_pos "end" --name 278x240_end --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#321x264
#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 321 --sizetrainy 264 --test_pos "start" --name 321x264_start --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 321 --sizetrainy 264 --test_pos "mid" --name 321x264_mid --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 321 --sizetrainy 264 --test_pos "end" --name 321x264_end --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#364x288
#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 364 --sizetrainy 288 --test_pos "start" --name 364x288_start --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 364 --sizetrainy 288 --test_pos "mid" --name 364x288_mid --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 364 --sizetrainy 288 --test_pos "end" --name 364x288_end --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#407x312
#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 407 --sizetrainy 312 --test_pos "start" --name 407x312_start --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 407 --sizetrainy 312 --test_pos "mid" --name 407x312_mid --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 407 --sizetrainy 312 --test_pos "end" --name 407x312_end --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#450x336
#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 450 --sizetrainy 336 --test_pos "start" --name 450x336_start --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 450 --sizetrainy 336 --test_pos "mid" --name 450x336_mid --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 450 --sizetrainy 336 --test_pos "end" --name 450x336_end --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#493x360
#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 493 --sizetrainy 360 --test_pos "start" --name 493x360_start --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 493 --sizetrainy 360 --test_pos "mid" --name 493x360_mid --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 493 --sizetrainy 360 --test_pos "end" --name 493x360_end --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#536x384
#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 536 --sizetrainy 384 --test_pos "start" --name 536x384_start --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 536 --sizetrainy 384 --test_pos "mid" --name 536x384_mid --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 536 --sizetrainy 384 --test_pos "end" --name 536x384_end --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#579x408
#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 579 --sizetrainy 408 --test_pos "start" --name 579x408_start --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 579 --sizetrainy 408 --test_pos "mid" --name 579x408_mid --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 579 --sizetrainy 408 --test_pos "end" --name 579x408_end --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#622x430
#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 622 --sizetrainy 430 --name 622x430 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID




#Repeat everything for confidence interval

#192x192
#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 192 --sizetrainy 192 --test_pos "start" --name 192x192_start_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 192 --sizetrainy 192 --test_pos "mid" --name 192x192_mid_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 192 --sizetrainy 192 --test_pos "end" --name 192x192_end_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#235x216
#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 235 --sizetrainy 216 --test_pos "start" --name 235x216_start_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 235 --sizetrainy 216 --test_pos "mid" --name 235x216_mid_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 235 --sizetrainy 216 --test_pos "end" --name 235x216_end_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#278x240
#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 278 --sizetrainy 240 --test_pos "start" --name 278x240_start_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 278 --sizetrainy 240 --test_pos "mid" --name 278x240_mid_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 278 --sizetrainy 240 --test_pos "end" --name 278x240_end_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#321x264
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 321 --sizetrainy 264 --test_pos "start" --name 321x264_start_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 321 --sizetrainy 264 --test_pos "mid" --name 321x264_mid_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 321 --sizetrainy 264 --test_pos "end" --name 321x264_end_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#364x288
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 364 --sizetrainy 288 --test_pos "start" --name 364x288_start_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 364 --sizetrainy 288 --test_pos "mid" --name 364x288_mid_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 364 --sizetrainy 288 --test_pos "end" --name 364x288_end_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#407x312
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 407 --sizetrainy 312 --test_pos "start" --name 407x312_start_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 407 --sizetrainy 312 --test_pos "mid" --name 407x312_mid_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 407 --sizetrainy 312 --test_pos "end" --name 407x312_end_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#450x336
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 450 --sizetrainy 336 --test_pos "start" --name 450x336_start_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 450 --sizetrainy 336 --test_pos "mid" --name 450x336_mid_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 450 --sizetrainy 336 --test_pos "end" --name 450x336_end_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#493x360
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 493 --sizetrainy 360 --test_pos "start" --name 493x360_start_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 493 --sizetrainy 360 --test_pos "mid" --name 493x360_mid_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 493 --sizetrainy 360 --test_pos "end" --name 493x360_end_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#536x384
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 536 --sizetrainy 384 --test_pos "start" --name 536x384_start_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 536 --sizetrainy 384 --test_pos "mid" --name 536x384_mid_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 536 --sizetrainy 384 --test_pos "end" --name 536x384_end_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#579x408
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 579 --sizetrainy 408 --test_pos "start" --name 579x408_start_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 579 --sizetrainy 408 --test_pos "mid" --name 579x408_mid_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 579 --sizetrainy 408 --test_pos "end" --name 579x408_end_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#622x430
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 622 --sizetrainy 430 --name 622x430_2 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID



#Ultima repeticao


#192x192
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 192 --sizetrainy 192 --test_pos "start" --name 192x192_start_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 192 --sizetrainy 192 --test_pos "mid" --name 192x192_mid_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 192 --sizetrainy 192 --test_pos "end" --name 192x192_end_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#235x216
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 235 --sizetrainy 216 --test_pos "start" --name 235x216_start_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 235 --sizetrainy 216 --test_pos "mid" --name 235x216_mid_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 235 --sizetrainy 216 --test_pos "end" --name 235x216_end_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#278x240
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 278 --sizetrainy 240 --test_pos "start" --name 278x240_start_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 278 --sizetrainy 240 --test_pos "mid" --name 278x240_mid_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 278 --sizetrainy 240 --test_pos "end" --name 278x240_end_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#321x264
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 321 --sizetrainy 264 --test_pos "start" --name 321x264_start_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 321 --sizetrainy 264 --test_pos "mid" --name 321x264_mid_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 321 --sizetrainy 264 --test_pos "end" --name 321x264_end_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#364x288
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 364 --sizetrainy 288 --test_pos "start" --name 364x288_start_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 364 --sizetrainy 288 --test_pos "mid" --name 364x288_mid_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 364 --sizetrainy 288 --test_pos "end" --name 364x288_end_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#407x312
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 407 --sizetrainy 312 --test_pos "start" --name 407x312_start_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 407 --sizetrainy 312 --test_pos "mid" --name 407x312_mid_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 407 --sizetrainy 312 --test_pos "end" --name 407x312_end_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#450x336
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 450 --sizetrainy 336 --test_pos "start" --name 450x336_start_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 450 --sizetrainy 336 --test_pos "mid" --name 450x336_mid_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 450 --sizetrainy 336 --test_pos "end" --name 450x336_end_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#493x360
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 493 --sizetrainy 360 --test_pos "start" --name 493x360_start_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 493 --sizetrainy 360 --test_pos "mid" --name 493x360_mid_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 493 --sizetrainy 360 --test_pos "end" --name 493x360_end_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#536x384
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 536 --sizetrainy 384 --test_pos "start" --name 536x384_start_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 536 --sizetrainy 384 --test_pos "mid" --name 536x384_mid_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 536 --sizetrainy 384 --test_pos "end" --name 536x384_end_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#579x408
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 579 --sizetrainy 408 --test_pos "start" --name 579x408_start_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 579 --sizetrainy 408 --test_pos "mid" --name 579x408_mid_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 579 --sizetrainy 408 --test_pos "end" --name 579x408_end_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#622x430
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 622 --sizetrainy 430 --name 622x430_3 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_parihaka --dataset 3 --kernel 7 --dropout 0.5 --gpuID $GPU_ID
