#!/bin/bash
readonly GPU_ID=1

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