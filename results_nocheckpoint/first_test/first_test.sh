#!/bin/sh
python train.py -m 2 --name "attentionAdam_G2" -o 0 -g 2 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 2 --name "attentionSGD_G2" -o 1 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 2 --name "attentionRMS_G2" -o 2 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 2 --name "attentionAdam_G4" -g 4 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 2 --name "attentionSGD_G4" -o 1 -g 4 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience  --loss_function 1 --folder "first_test"
python train.py -m 2 --name "attentionRMS_G4" -o 2 -g 4 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 2 --name "attentionSGD_G025" -o 1 -g 0.25 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 2 --name "attentionRMS_G025" -o 2 -g 0.25 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 2 --name "attentionAdam_G025" -g 0.25 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"

python train.py -m 1 --name "threeplusAdam_G2" -o 0 -g 2 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 1 --name "threeplusSGD_G2" -o 1 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 1 --name "threeplusRMS_G2" -o 2 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 1 --name "threeplusAdam_G4" -g 4 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 1 --name "threeplusSGD_G4" -o 1 -g 4 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 1 --name "threeplusRMS_G4" -o 2 -g 4 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 1 --name "threeplusAdam_G025" -g 0.25 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 1 --name "threeplusSGD_G025" -o 1 -g 0.25 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 1 --name "threeplusRMS_G025" -o 2 -g 0.25 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"

python train.py -m 0 --name "UnetAdam_G2(baseline)" -o 0 -g 2 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 0 --name "UnetSGD_G2" -o 1 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 0 --name "UnetRMS_G2" -o 2 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 0 --name "UnetAdam_G4" -g 4 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 0 --name "UnetSGD_G4" -o 1 -g 4 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 0 --name "UnetRMS_G4" -o 2 -g 4 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 0 --name "UnetSGD_G025" -o 1 -g 0.25 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 0 --name "UnetRMS_G025" -o 2 -g 0.25 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"
python train.py -m 0 --name "UnetAdam_G025" -g 0.25 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 10 --loss_function 1 --folder "first_test"