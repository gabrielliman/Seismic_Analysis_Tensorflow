#!/bin/bash
readonly GPU_ID=1
# feitos

# #batch size=16 stride=128x64 otimizador=Adam delta=1e-4 patience=15 loss_function=Cross Entropy kernel_size=11 dropout=0 Dataset=parihaka_minha_divisao
# #parihaka
# python ./train.py --optimizer 0 --model 0 --epochs 100 --batch_size 16 --name base_unet --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka --dataset 0 --kernel 11 --dropout 0

# python ./train.py --optimizer 0 --model 1 --epochs 100 --batch_size 16 --name base_unet3+ --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka --dataset 0 --kernel 11 --dropout 0

# python ./train.py --optimizer 0 --model 2 --epochs 100 --batch_size 16 --name base_attention --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder parihaka --dataset 0 --kernel 11 --dropout 0


# #penobscot

python ./train.py --optimizer 0 --model 0 --epochs 100 --batch_size 16 --name base_unet_1607 --stride1 256 --stride2 64 --slice_shape1 1024 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder penobscot --dataset 1 --kernel 11 --dropout 0 --gpuID $GPU_ID

python ./train.py --optimizer 0 --model 1 --epochs 100 --batch_size 16 --name base_unet3+_1607 --stride1 256 --stride2 64 --slice_shape1 1024 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder penobscot --dataset 1 --kernel 11 --dropout 0 --gpuID $GPU_ID

python ./train.py --optimizer 0 --model 2 --epochs 100 --batch_size 16 --name base_attention_1607 --stride1 256 --stride2 64 --slice_shape1 1024 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 0 --folder penobscot --dataset 1 --kernel 11 --dropout 0 --gpuID $GPU_ID

#melhor ate agora parihaka
#python ./train.py --optimizer 0 --gamma 3.6 --model 2 --epochs 100 --batch_size 4 --name parihaka_best100_mydivision --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --slice_shape1 992 --slice_shape2 192 --delta 1e-4 --patience 15 --loss_function 1 --folder parihaka --dataset 0 --kernel 7 --dropout 0.5


#FALTA FAZER
#vou rodar dnv o optimizador 

python ./train_bayes.py --name "penobscot_bayes_10_100_16jul" --optimizer 0 -s1 1024 -s2 192 --stride1 256 --stride2 64 --delta 1e-4 --patience 5 --loss_function 0 --folder "penobscot_bayes" --init_points 10 --num_iter 100 --epochs 100 --dataset 1 --gpuID $GPU_ID

while true; do
    # Tente executar seu comando aqui
    echo "Tentando executar o comando..."

    # Substitua "seu_comando" pelo comando real que você deseja executar
    python ./train_bayes.py --name "penobscot_bayes_10_100_16jul" --last_iter "penobscot_bayes_10_100_16jul" --optimizer 0 -s1 1024 -s2 192 --stride1 256 --stride2 64 --delta 1e-4 --patience 5 --loss_function 0 --folder "penobscot_bayes" --init_points 0 --num_iter 100 --epochs 100 --dataset 1 --gpuID $GPU_ID

    # Verifique o status de saída do comando
    if [ $? -eq 0 ]; then
        echo "Comando executado com sucesso!"
        break
    else
        echo "Ocorreu um erro ao executar o comando. Tentando novamente em 5 segundos..."
        sleep 5
    fi
done

#python ./train_bayes.py --name "parihaka_bayes_10_100_8jul" --last_iter "parihaka_bayes_10_100" --optimizer 0 -s1 992 -s2 192 --stride1 128 --stride2 64 --stridetest1 128  --stridetest2 64 --delta 1e-4 --patience 15 --loss_function 0 --folder "parihaka_bayes" --init_points 0 --num_iter 100 --epochs 100 --dataset 0
