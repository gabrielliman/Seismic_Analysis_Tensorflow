#!/bin/bash
readonly GPU_ID=1

#192x192
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 192 --sizetrainy 192 --test_pos "start" --name 192x192_start --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_2 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 192 --sizetrainy 192 --test_pos "mid" --name 192x192_mid --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_2 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 192 --sizetrainy 192 --test_pos "end" --name 192x192_end --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_2 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#224x212
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 224 --sizetrainy 212 --test_pos "start" --name 224x212_start --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_2 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 224 --sizetrainy 212 --test_pos "mid" --name 224x212_mid --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_2 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 224 --sizetrainy 212 --test_pos "end" --name 224x212_end --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_2 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#256x232
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 256 --sizetrainy 232 --test_pos "start" --name 256x232_start --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_2 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 256 --sizetrainy 232 --test_pos "mid" --name 256x232_mid --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_2 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 256 --sizetrainy 232 --test_pos "end" --name 256x232_end --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_2 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#288x252
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 288 --sizetrainy 252 --test_pos "start" --name 288x252_start --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_2 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 288 --sizetrainy 252 --test_pos "mid" --name 288x252_mid --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_2 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 288 --sizetrainy 252 --test_pos "end" --name 288x252_end --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_2 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#320x270
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 320 --sizetrainy 270 --test_pos "start" --name 320x270_start --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_2 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 320 --sizetrainy 270 --test_pos "mid" --name 320x270_mid --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_2 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 320 --sizetrainy 270 --test_pos "end" --name 320x270_end --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_2 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID






#192x192
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 192 --sizetrainy 192 --test_pos "start" --name 192x192_start --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_3 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 192 --sizetrainy 192 --test_pos "mid" --name 192x192_mid --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_3 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 192 --sizetrainy 192 --test_pos "end" --name 192x192_end --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_3 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#224x212
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 224 --sizetrainy 212 --test_pos "start" --name 224x212_start --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_3 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 224 --sizetrainy 212 --test_pos "mid" --name 224x212_mid --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_3 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 224 --sizetrainy 212 --test_pos "end" --name 224x212_end --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_3 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#256x232
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 256 --sizetrainy 232 --test_pos "start" --name 256x232_start --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_3 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 256 --sizetrainy 232 --test_pos "mid" --name 256x232_mid --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_3 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 256 --sizetrainy 232 --test_pos "end" --name 256x232_end --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_3 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#288x252
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 288 --sizetrainy 252 --test_pos "start" --name 288x252_start --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_3 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 288 --sizetrainy 252 --test_pos "mid" --name 288x252_mid --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_3 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 288 --sizetrainy 252 --test_pos "end" --name 288x252_end --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_3 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

#320x270
python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 320 --sizetrainy 270 --test_pos "start" --name 320x270_start --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_3 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 320 --sizetrainy 270 --test_pos "mid" --name 320x270_mid --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_3 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID

python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --sizetrainx 320 --sizetrainy 270 --test_pos "end" --name 320x270_end --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder training_size_variation_fixed_val_test_parihaka_3 --dataset 7 --kernel 7 --dropout 0.5 --gpuID $GPU_ID
