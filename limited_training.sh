#!/bin/bash
readonly GPU_ID=1


#Attention baseline

for sizex_sizey in "424x264" "453x273"; do
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
    --slice_shape1 992 \
    --slice_shape2 192 \
    --delta 1e-4 \
    --patience 5 \
    --loss_function 0 \
    --folder attention_baseline_limited_end_penobscot \
    --dataset 10 \
    --kernel 11 \
    --dropout 0 \
    --gpuID "$GPU_ID"
done

#UNet
for sizex_sizey in "192x192" "235x216" "278x240" "321x264" "364x288" "407x312" "450x336" "493x360" "536x384" "579x408"; do
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
    --slice_shape1 992 \
    --slice_shape2 192 \
    --delta 1e-4 \
    --patience 5 \
    --loss_function 0 \
    --folder unet_baseline_limited_end \
    --dataset 3 \
    --kernel 11 \
    --dropout 0 \
    --gpuID "$GPU_ID"
done

for sizex_sizey in "192x192" "221x201" "250x210" "279x219" "308x228" "337x237" "366x246" "395x255" "424x264" "453x273"; do
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
    --slice_shape1 992 \
    --slice_shape2 192 \
    --delta 1e-4 \
    --patience 5 \
    --loss_function 0 \
    --folder unet_baseline_limited_end_penobscot \
    --dataset 10 \
    --kernel 11 \
    --dropout 0 \
    --gpuID "$GPU_ID"
done

#Unet 3+

for sizex_sizey in "192x192" "235x216" "278x240" "321x264" "364x288" "407x312" "450x336" "493x360" "536x384" "579x408"; do
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
    --slice_shape1 992 \
    --slice_shape2 192 \
    --delta 1e-4 \
    --patience 5 \
    --loss_function 0 \
    --folder unet3+_baseline_limited_end \
    --dataset 3 \
    --kernel 11 \
    --dropout 0 \
    --gpuID "$GPU_ID"
done

for sizex_sizey in "192x192" "221x201" "250x210" "279x219" "308x228" "337x237" "366x246" "395x255" "424x264" "453x273"; do
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
    --slice_shape1 992 \
    --slice_shape2 192 \
    --delta 1e-4 \
    --patience 5 \
    --loss_function 0 \
    --folder unet3+_baseline_limited_end_penobscot \
    --dataset 10 \
    --kernel 11 \
    --dropout 0 \
    --gpuID "$GPU_ID"
done

#CPFNetM
for sizex_sizey in "192x192" "235x216" "278x240" "321x264" "364x288" "407x312" "450x336" "493x360" "536x384" "579x408"; do
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
    --slice_shape1 992 \
    --slice_shape2 192 \
    --delta 1e-4 \
    --patience 5 \
    --loss_function 0 \
    --folder CPFNetM_baseline_limited_end \
    --dataset 3 \
    --kernel 11 \
    --dropout 0 \
    --gpuID "$GPU_ID"
done

for sizex_sizey in "192x192" "221x201" "250x210" "279x219" "308x228" "337x237" "366x246" "395x255" "424x264" "453x273"; do
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
    --slice_shape1 992 \
    --slice_shape2 192 \
    --delta 1e-4 \
    --patience 5 \
    --loss_function 0 \
    --folder CPFNetM_baseline_limited_end_penobscot \
    --dataset 10 \
    --kernel 11 \
    --dropout 0 \
    --gpuID "$GPU_ID"
done

#ENet
for sizex_sizey in "192x192" "235x216" "278x240" "321x264" "364x288" "407x312" "450x336" "493x360" "536x384" "579x408"; do
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
    --slice_shape1 992 \
    --slice_shape2 192 \
    --delta 1e-4 \
    --patience 5 \
    --loss_function 0 \
    --folder ENet_baseline_limited_end \
    --dataset 3 \
    --kernel 11 \
    --dropout 0 \
    --gpuID "$GPU_ID"
done

for sizex_sizey in "192x192" "221x201" "250x210" "279x219" "308x228" "337x237" "366x246" "395x255" "424x264" "453x273"; do
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
    --slice_shape1 992 \
    --slice_shape2 192 \
    --delta 1e-4 \
    --patience 5 \
    --loss_function 0 \
    --folder ENet_baseline_limited_end_penobscot \
    --dataset 10 \
    --kernel 11 \
    --dropout 0 \
    --gpuID "$GPU_ID"
done

#ESPNet

for sizex_sizey in "192x192" "235x216" "278x240" "321x264" "364x288" "407x312" "450x336" "493x360" "536x384" "579x408"; do
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
    --slice_shape1 992 \
    --slice_shape2 192 \
    --delta 1e-4 \
    --patience 5 \
    --loss_function 0 \
    --folder ESPNet_baseline_limited_end \
    --dataset 3 \
    --kernel 11 \
    --dropout 0 \
    --gpuID "$GPU_ID"
done


for sizex_sizey in "192x192" "221x201" "250x210" "279x219" "308x228" "337x237" "366x246" "395x255" "424x264" "453x273"; do
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
    --slice_shape1 992 \
    --slice_shape2 192 \
    --delta 1e-4 \
    --patience 5 \
    --loss_function 0 \
    --folder ESPNet_baseline_limited_end_penobscot \
    --dataset 10 \
    --kernel 11 \
    --dropout 0 \
    --gpuID "$GPU_ID"
done

#ICNet

for sizex_sizey in "192x192" "235x216" "278x240" "321x264" "364x288" "407x312" "450x336" "493x360" "536x384" "579x408"; do
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
    --slice_shape1 992 \
    --slice_shape2 192 \
    --delta 1e-4 \
    --patience 5 \
    --loss_function 0 \
    --folder ICNet_baseline_limited_end \
    --dataset 3 \
    --kernel 11 \
    --dropout 0 \
    --gpuID "$GPU_ID"
done


for sizex_sizey in "192x192" "221x201" "250x210" "279x219" "308x228" "337x237" "366x246" "395x255" "424x264" "453x273"; do
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
    --slice_shape1 992 \
    --slice_shape2 192 \
    --delta 1e-4 \
    --patience 5 \
    --loss_function 0 \
    --folder ICNet_baseline_limited_end_penobscot \
    --dataset 10 \
    --kernel 11 \
    --dropout 0 \
    --gpuID "$GPU_ID"
done

#EfficientNetB1

for sizex_sizey in "192x192" "235x216" "278x240" "321x264" "364x288" "407x312" "450x336" "493x360" "536x384" "579x408"; do
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
    --slice_shape1 992 \
    --slice_shape2 192 \
    --delta 1e-4 \
    --patience 5 \
    --loss_function 0 \
    --folder EfficientNetB1_baseline_limited_end \
    --dataset 3 \
    --kernel 11 \
    --dropout 0 \
    --gpuID "$GPU_ID"
done


for sizex_sizey in "192x192" "221x201" "250x210" "279x219" "308x228" "337x237" "366x246" "395x255" "424x264" "453x273"; do
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
    --slice_shape1 992 \
    --slice_shape2 192 \
    --delta 1e-4 \
    --patience 5 \
    --loss_function 0 \
    --folder EfficientNetB1_baseline_limited_end_penobscot \
    --dataset 10 \
    --kernel 11 \
    --dropout 0 \
    --gpuID "$GPU_ID"
done

#LWBAUnet

for sizex_sizey in "192x192" "235x216" "278x240" "321x264" "364x288" "407x312" "450x336" "493x360" "536x384" "579x408"; do
  sizex=${sizex_sizey%x*}
  sizey=${sizex_sizey#*x}
  
  python ./train.py \
    --test_pos "end" \
    --sizetrainx "$sizex" \
    --sizetrainy "$sizey" \
    --optimizer 0 \
    --model 11 \
    --epochs 100 \
    --batch_size 8 \
    --name "${sizex}_${sizey}" \
    --stride1 128 \
    --stride2 64 \
    --stridetest1 128 \
    --stridetest2 64 \
    --slice_shape1 992 \
    --slice_shape2 192 \
    --delta 1e-4 \
    --patience 5 \
    --loss_function 0 \
    --folder LWBAUnet_baseline_limited_end \
    --dataset 3 \
    --kernel 11 \
    --dropout 0 \
    --gpuID "$GPU_ID"
done


for sizex_sizey in "192x192" "221x201" "250x210" "279x219" "308x228" "337x237" "366x246" "395x255" "424x264" "453x273"; do
  sizex=${sizex_sizey%x*}
  sizey=${sizex_sizey#*x}
  
  python ./train.py \
    --test_pos "end" \
    --sizetrainx "$sizex" \
    --sizetrainy "$sizey" \
    --optimizer 0 \
    --model 11 \
    --epochs 100 \
    --batch_size 8 \
    --name "${sizex}_${sizey}" \
    --stride1 128 \
    --stride2 64 \
    --stridetest1 128 \
    --stridetest2 64 \
    --slice_shape1 992 \
    --slice_shape2 192 \
    --delta 1e-4 \
    --patience 5 \
    --loss_function 0 \
    --folder LWBAUnet_baseline_limited_end_penobscot \
    --dataset 10 \
    --kernel 11 \
    --dropout 0 \
    --gpuID "$GPU_ID"
done

#BridgeNet

for sizex_sizey in "192x192" "235x216" "278x240" "321x264" "364x288" "407x312" "450x336" "493x360" "536x384" "579x408"; do
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
    --slice_shape1 992 \
    --slice_shape2 192 \
    --delta 1e-4 \
    --patience 5 \
    --loss_function 0 \
    --folder BridgeNet_baseline_limited_end \
    --dataset 3 \
    --kernel 11 \
    --dropout 0 \
    --gpuID "$GPU_ID"
done


for sizex_sizey in "192x192" "221x201" "250x210" "279x219" "308x228" "337x237" "366x246" "395x255" "424x264" "453x273"; do
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
    --slice_shape1 992 \
    --slice_shape2 192 \
    --delta 1e-4 \
    --patience 5 \
    --loss_function 0 \
    --folder BridgeNet_baseline_limited_end_penobscot \
    --dataset 10 \
    --kernel 11 \
    --dropout 0 \
    --gpuID "$GPU_ID"
done