#!/bin/sh
# #test focal 100 1e-4

#test cross 100 1e-4
# echo "start of test cross 100 1e-4"

# python train.py -m 2 --name "attentionAdam" -o 0 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 0 --folder "100epochscross" 
# python train.py -m 2 --name "attentionSGD" -o 1 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 0 --folder "100epochscross" 
# python train.py -m 2 --name "attentionRMS" -o 2 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 0 --folder "100epochscross" 


# python train.py -m 1 --name "threeplusAdam" -o 0 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 0 --folder "100epochscross" 
# python train.py -m 1 --name "threeplusSGD" -o 1 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 0 --folder "100epochscross" 
# python train.py -m 1 --name "threeplusRMS" -o 2 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 0 --folder "100epochscross" 


# python train.py -m 0 --name "UnetAdam(baseline)" -o 0 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 0 --folder "100epochscross" 
# python train.py -m 0 --name "UnetSGD" -o 1 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 0 --folder "100epochscross" 
# python train.py -m 0 --name "UnetRMS" -o 2 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 0 --folder "100epochscross" 

# echo "end of test cross 100 1e-4"

# #test cross 200 1e-5
# echo "start of test cross 200 1e-5"

# python train.py -m 2 --name "attentionAdam" -o 0 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 0 --folder "200epochscross" 
# python train.py -m 2 --name "attentionSGD" -o 1 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 0 --folder "200epochscross" 
# python train.py -m 2 --name "attentionRMS" -o 2 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 0 --folder "200epochscross" 


# python train.py -m 1 --name "threeplusAdam" -o 0 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 0 --folder "200epochscross" 
# python train.py -m 1 --name "threeplusSGD" -o 1 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 0 --folder "200epochscross" 
# python train.py -m 1 --name "threeplusRMS" -o 2 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 0 --folder "200epochscross" 


# python train.py -m 0 --name "UnetAdam(baseline)" -o 0 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 0 --folder "200epochscross" 
# python train.py -m 0 --name "UnetSGD" -o 1 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 0 --folder "200epochscross" 
# python train.py -m 0 --name "UnetRMS" -o 2 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 0 --folder "200epochscross" 

# echo "end of test cross 200 1e-5"
# #test focal 200 1e-5
# echo "start of test focal 200 1e-5"

# python train.py -m 2 --name "attentionAdam_G2" -o 0 -g 2 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 2 --name "attentionSGD_G2" -o 1 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 2 --name "attentionRMS_G2" -o 2 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 2 --name "attentionAdam_G4" -g 4 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 2 --name "attentionSGD_G4" -o 1 -g 4 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 2 --name "attentionRMS_G4" -o 2 -g 4 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal"
# python train.py -m 2 --name "attentionSGD_G025" -o 1 -g 0.25 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 2 --name "attentionRMS_G025" -o 2 -g 0.25 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 2 --name "attentionAdam_G025" -g 0.25 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 

# python train.py -m 1 --name "threeplusAdam_G2" -o 0 -g 2 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 1 --name "threeplusSGD_G2" -o 1 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 1 --name "threeplusRMS_G2" -o 2 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 1 --name "threeplusAdam_G4" -g 4 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 1 --name "threeplusSGD_G4" -o 1 -g 4 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 1 --name "threeplusRMS_G4" -o 2 -g 4 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 1 --name "threeplusAdam_G025" -g 0.25 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 1 --name "threeplusSGD_G025" -o 1 -g 0.25 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 1 --name "threeplusRMS_G025" -o 2 -g 0.25 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 

# python train.py -m 0 --name "UnetAdam_G2(baseline)" -o 0 -g 2 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 0 --name "UnetSGD_G2" -o 1 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 0 --name "UnetRMS_G2" -o 2 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 0 --name "UnetAdam_G4" -g 4 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 0 --name "UnetSGD_G4" -o 1 -g 4 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 0 --name "UnetRMS_G4" -o 2 -g 4 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 0 --name "UnetSGD_G025" -o 1 -g 0.25 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 0 --name "UnetRMS_G025" -o 2 -g 0.25 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 
# python train.py -m 0 --name "UnetAdam_G025" -g 0.25 --epochs 200 -s2 192 --stridetrain 64 --delta 1e-5 --patience 30 --loss_function 1 --folder "200epochsfocal" 



#test focal 100 1e-4
echo "start of test focal 100 1e-4"

python train.py -m 2 --name "attentionAdam_G2" -o 0 -g 2 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 2 --name "attentionSGD_G2" -o 1 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 2 --name "attentionRMS_G2" -o 2 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 2 --name "attentionAdam_G4" -g 4 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 2 --name "attentionSGD_G4" -o 1 -g 4 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 2 --name "attentionRMS_G4" -o 2 -g 4 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 2 --name "attentionSGD_G025" -o 1 -g 0.25 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 2 --name "attentionRMS_G025" -o 2 -g 0.25 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 2 --name "attentionAdam_G025" -g 0.25 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"

python train.py -m 1 --name "threeplusAdam_G2" -o 0 -g 2 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 1 --name "threeplusSGD_G2" -o 1 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 1 --name "threeplusRMS_G2" -o 2 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 1 --name "threeplusAdam_G4" -g 4 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 1 --name "threeplusSGD_G4" -o 1 -g 4 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 1 --name "threeplusRMS_G4" -o 2 -g 4 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 1 --name "threeplusAdam_G025" -g 0.25 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 1 --name "threeplusSGD_G025" -o 1 -g 0.25 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 1 --name "threeplusRMS_G025" -o 2 -g 0.25 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"

python train.py -m 0 --name "UnetAdam_G2(baseline)" -o 0 -g 2 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 0 --name "UnetSGD_G2" -o 1 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 0 --name "UnetRMS_G2" -o 2 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 0 --name "UnetAdam_G4" -g 4 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 0 --name "UnetSGD_G4" -o 1 -g 4 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 0 --name "UnetRMS_G4" -o 2 -g 4 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 0 --name "UnetSGD_G025" -o 1 -g 0.25 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 0 --name "UnetRMS_G025" -o 2 -g 0.25 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"
python train.py -m 0 --name "UnetAdam_G025" -g 0.25 --epochs 100 -s2 192 --stridetrain 64 --delta 1e-4 --patience 10 --loss_function 1 --folder "100epochsfocal"

echo "end of test focal 100 1e-4"