#Penobscot 192x192 221x201 250x210 279x219
    #UNet
        for sizex_sizey in "192x192" "221x201" "250x210" "279x219"; do
            sizex=${sizex_sizey%x*}
            sizey=${sizex_sizey#*x}
            
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 0 \
                --epochs 100 \
                --batch_size 16 \
                --name "${sizex}_${sizey}" \
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 1472 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder UNet_limited_end_penobscot \
                --dataset 10 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
        done
    #BridgeNet 
            for sizex_sizey in "192x192" "221x201" "250x210" "279x219"; do
            sizex=${sizex_sizey%x*}
            sizey=${sizex_sizey#*x}
            
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 3 \
                --epochs 100 \
                --batch_size 16 \
                --name "${sizex}_${sizey}" \
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 1472 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder BridgeNet_limited_end_penobscot \
                --dataset 10 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
        done
    #UNet3+ 
            for sizex_sizey in "192x192" "221x201" "250x210" "279x219"; do
            sizex=${sizex_sizey%x*}
            sizey=${sizex_sizey#*x}
            
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 1 \
                --epochs 100 \
                --batch_size 16 \
                --name "${sizex}_${sizey}" \
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 1472 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder UNet3+_limited_end_penobscot \
                --dataset 10 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
        done
    #Attention 
            for sizex_sizey in "192x192" "221x201" "250x210" "279x219"; do
            sizex=${sizex_sizey%x*}
            sizey=${sizex_sizey#*x}
            
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 2 \
                --epochs 100 \
                --batch_size 16 \
                --name "${sizex}_${sizey}" \
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 1472 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder Attention_limited_end_penobscot \
                --dataset 10 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
        done
    #ESPNet 
            for sizex_sizey in "192x192" "221x201" "250x210" "279x219"; do
            sizex=${sizex_sizey%x*}
            sizey=${sizex_sizey#*x}
            
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 7 \
                --epochs 100 \
                --batch_size 16 \
                --name "${sizex}_${sizey}" \
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 1472 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder ESPNet_limited_end_penobscot \
                --dataset 10 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
        done
    #ENet 
            for sizex_sizey in "192x192" "221x201" "250x210" "279x219"; do
            sizex=${sizex_sizey%x*}
            sizey=${sizex_sizey#*x}
            
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 6 \
                --epochs 100 \
                --batch_size 16 \
                --name "${sizex}_${sizey}" \
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 1472 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder ENet_limited_end_penobscot \
                --dataset 10 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
        done
    #ICNet 
            for sizex_sizey in "192x192" "221x201" "250x210" "279x219"; do
            sizex=${sizex_sizey%x*}
            sizey=${sizex_sizey#*x}
            
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 8 \
                --epochs 100 \
                --batch_size 16 \
                --name "${sizex}_${sizey}" \
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 1472 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder ICNet_limited_end_penobscot \
                --dataset 10 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
        done
    #CPFNetM 
            for sizex_sizey in "192x192" "221x201" "250x210" "279x219"; do
            sizex=${sizex_sizey%x*}
            sizey=${sizex_sizey#*x}
            
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 4 \
                --epochs 100 \
                --batch_size 16 \
                --name "${sizex}_${sizey}" \
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 1472 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder CPFNetM_limited_end_penobscot \
                --dataset 10 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
        done
    #LWBNA 
            for sizex_sizey in "192x192" "221x201" "250x210" "279x219"; do
            sizex=${sizex_sizey%x*}
            sizey=${sizex_sizey#*x}
            
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 11 \
                --epochs 100 \
                --batch_size 16 \
                --name "${sizex}_${sizey}" \
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 1472 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder LWBNA_limited_end_penobscot \
                --dataset 10 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
        done
    #EfficientNetB1 
            for sizex_sizey in "192x192" "221x201" "250x210" "279x219"; do
            sizex=${sizex_sizey%x*}
            sizey=${sizex_sizey#*x}
            
            python ./train.py \
                --test_pos "end" \
                --sizetrainx "$sizex" \
                --sizetrainy "$sizey" \
                --optimizer 0 \
                --model 10 \
                --epochs 100 \
                --batch_size 16 \
                --name "${sizex}_${sizey}" \
                --stride1 128 \
                --stride2 64 \
                --stridetest1 128 \
                --stridetest2 64 \
                --slice_shape1 1472 \
                --slice_shape2 192 \
                --delta 1e-4 \
                --patience 5 \
                --loss_function 0 \
                --folder EfficientNetB1_limited_end_penobscot \
                --dataset 10 \
                --kernel 11 \
                --dropout 0 \
                --gpuID "$GPU_ID"
        done