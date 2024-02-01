import argparse
import tensorflow as tf
import numpy as np
import os
from models.attention import Attention_unet
from models.unet import Unet
from models.unet3plus import Unet_3plus
from models.newmodel import unetmodel
from focal_loss import SparseCategoricalFocalLoss
#from utils.datapreparationlocal import my_division_data
from utils.datapreparation import my_division_data
from utils.prediction import make_prediction
# from models.bridgenet import BridgeNet_1
import matplotlib.pyplot as plt
from utils.datapreparation_aerialimagery import aerial_patches
from utils.datapreparation_forest import forest_patches
from utils.datapreparation_drone import drone_patches

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

#conda run -n seismic_tf python ./teste_rapido.py -m 3 --name "0102_transfer_aerial_teste" -o 0 -g 3.6 -s1 992 -s2 192 --stridetrain 100 --delta 1e-5 --patience 10 --loss_function 1 --folder "0102_aerial_transfer_results_teste" --epochs 100 --dataset 4 -b 4 -k 3 --weights_path "./checkpoints/0102_aerial_transfer_results/checkpoint_0102_transfer_aerial.h5" --dropout 0.5 --pre_classes 6 --classes 6


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--optimizer', '-o', metavar='O', type=int, default=0, help="Choose optimizer, 0: Adam, 1: SGD, 2: RMS")
    parser.add_argument('--gamma', '-g', metavar='G', type=float, default=2, help="Gamma for Sparce Categorical Focal Loss, deve ser um float")
    parser.add_argument('--model', '-m', metavar='M', type=int, default=0, help="Choose Segmentation Model, 0: Unet, 1: Unet 3 Plus, 2: Attention UNet")
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Limit of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--name', '-n', type=str, default="default", help='Model name for saving')
    parser.add_argument('--stridetrain', type=int, default=32, help="Stride in second dimension for train images")
    parser.add_argument('--slice_shape1', '-s1',dest='slice_shape1', metavar='S', type=int, default=992, help='Shape 1 of the image slices')
    parser.add_argument('--slice_shape2', '-s2',dest='slice_shape2', metavar='S', type=int, default=576, help='Shape 2 of the image slices')
    parser.add_argument('--delta', '-d', type=float, default=1e-4, help="Delta for call back function")
    parser.add_argument('--patience', '-p', dest='patience', metavar='P', type=int, default=10, help="Patience for callback function")
    parser.add_argument('--loss_function', '-l', dest='loss_function', metavar='L', type=int, default=0, help="Choose loss function, 0= Cross Entropy, 1= Focal Loss")
    parser.add_argument('--folder', '-f', type=str, default="default_folder", help='Name of the folder where the results will be saved')
    parser.add_argument('--init_points', type=int, default=0, help="number of init points on bayes optimizer")
    parser.add_argument('--num_iter', type=int, default=0, help="number of iterations on bayes optimizer")
    parser.add_argument('--weights_path', type=str, default="", help="path to checkpoints of weights you want to load")
    parser.add_argument('--dataset', type=int, default=0, help="which dataset will be used, 0=seismic, 1=aerial imagery")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout rate")
    parser.add_argument('--kernel', '-k', type=int, default=3, help="kernel size")
    parser.add_argument('--pre_classes', type=int, default=6, help="number of classes of pre trained")
    parser.add_argument('--classes', type=int, default=6, help="number of classes to be trained")





    return parser.parse_args()

if __name__ == '__main__':
  
  #Creation of image slices based on arguments
  args= get_args()
  slice_shape1=args.slice_shape1
  slice_shape2=args.slice_shape2
  num_classes=args.pre_classes
  stride1=16
  strideval2=100
  stridetest2=100
  if(args.dataset==0):
    train_image,train_label, test_image, test_label, val_image, val_label=my_division_data(shape=(slice_shape1,slice_shape2), stridetrain=(stride1,args.stridetrain), strideval=(stride1,strideval2), stridetest=(stride1,stridetest2))
  elif(args.dataset==1):
    train_image,train_label, test_image, test_label, val_image, val_label=aerial_patches("/scratch/nuneslima/aerial_imagery/dataset", (slice_shape1,slice_shape2), (stride1,args.stridetrain),train=70, val=15, test=15)
  elif(args.dataset==2):
    train_image,train_label, test_image, test_label, val_image, val_label=forest_patches("/scratch/nuneslima/forest/Forest Segmented/Forest Segmented",train=70, val=15, test=15)
  elif(args.dataset==3):
    train_image,train_label, test_image, test_label, val_image, val_label=drone_patches("/scratch/nuneslima/drone",(slice_shape1,slice_shape2), (400,args.stridetrain),train=70, val=15, test=15)
  if(args.dataset==4):
    train_image,train_label, test_image, test_label, val_image, val_label=my_division_data(shape=(slice_shape1,slice_shape2), stridetrain=(stride1,args.stridetrain), strideval=(stride1,strideval2), stridetest=(stride1,stridetest2))
    train_image=train_image[:int(train_image.shape[0]/2)]
    train_label=train_label[:int(train_label.shape[0]/2)]
    test_image=test_image[:int(test_image.shape[0]/2)]
    test_label=test_label[:int(test_image.shape[0]/2)]

    # val_image=val_image[:100]
    # val_label=val_label[:100]
    

  # train_image=train_image[:100]
  # train_label=train_label[:100]
  #test_image=test_image[:100]
  #test_label=test_label[:100]
  # val_image=val_image[:100]
  # val_label=val_label[:100]
  
  
  #Definition of Models
  if(args.model==0):
    model = Unet(tam_entrada=(slice_shape1, slice_shape2, 1), num_filtros=[16, 32, 64, 128], classes=num_classes)
  elif(args.model==3):
    model = Attention_unet(tam_entrada=(slice_shape1, slice_shape2, 1), num_filtros=[16, 32, 64, 128, 256], classes=num_classes, dropout_rate=args.dropout, kernel_size=args.kernel)
  elif(args.model==2):
    model = Attention_unet(tam_entrada=(slice_shape1, slice_shape2, 1), num_filtros=[16, 32, 64, 128, 256, 512], classes=num_classes, dropout_rate=args.dropout, kernel_size=args.kernel)


  if(args.weights_path!=""):
    # model.load_weights(args.weights_path)
    print("\n\n LOADED WEIGHTS \n\n")
    new_model=Sequential()

    # for layer in model.layers[:-1]:
    #     new_model.add(layer)
    # # Add a new Dense layer with the specified number of classes
    # new_model.add(Dense(args.classes, activation='softmax'))
    # new_model.layers[-1].trainable = True  # You can set this to False if you want to freeze the last Dense layer

    new_model.add(model)
    new_model.add(Dense(args.classes, activation='softmax'))
    new_model.load_weights(args.weights_path)

    new_model.layers[0].trainable = True
    model= new_model
    num_classes=args.classes




  checkpoint_filepath = './checkpoints/'+args.folder+'/checkpoint_'+args.name+'.h5'

  if not os.path.exists('./checkpoints'):
     os.makedirs('./checkpoints')

  if not os.path.exists('./checkpoints/'+args.folder):
     os.makedirs('./checkpoints/'+args.folder)


  #Callback function   
  callbacks = [
      tf.keras.callbacks.EarlyStopping(
          # Stop training when `val_loss` is no longer improving
          monitor="val_loss",
          min_delta=args.delta,
          patience=args.patience,
          verbose=1,
      ),
      tf.keras.callbacks.ModelCheckpoint(
          filepath=checkpoint_filepath,
          save_weights_only=True,
          monitor= 'val_unet3plus_output_sup2_activation_acc' if args.model == 1 else "val_acc",
          mode='max',
          save_best_only=True
      )
  ]
  #Definition of Optimizers
  if(args.optimizer==0):
      opt=tf.keras.optimizers.Adam(learning_rate=1e-4)
      opt_name="Adam"
  elif(args.optimizer==1):
      opt=tf.keras.optimizers.SGD()
      opt_name="SGD"
  elif(args.optimizer==2):
      opt=tf.keras.optimizers.RMSprop()
      opt_name="RMS"

  #Definition of Loss Function
  if(args.loss_function==0):
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    loss_name="Sparce Categorical Cross Entropy"
  else:
    loss=SparseCategoricalFocalLoss(gamma=args.gamma, from_logits=True)
    loss_name="Sparce Categorical Focal Loss, Gamma: " + str(args.gamma)

  

  #Model Compilation and Training
  model.compile(optimizer=opt,
                      loss=loss,
                    metrics=['acc'])
  

#   history = model.fit(train_image, train_label, batch_size=args.batch_size, epochs=args.epochs,
#                           callbacks=callbacks,
#                           validation_data=(val_image, val_label))     
  

  # #The best epoch is saved 
  # model.load_weights(checkpoint_filepath)
  # model.save_weights(checkpoint_filepath)

  if not os.path.exists('./results/'+args.folder):
    os.makedirs('./results/'+args.folder)

  if not os.path.exists('./results/'+args.folder+'/graphs'):
    os.makedirs('./results/'+args.folder+'/graphs')
  
  if not os.path.exists('./results/'+args.folder+'/tables'):
      os.makedirs('./results/'+args.folder+'/tables')
  
  #Creation of training graphs with Loss and Accuracy, of Validation and Training, by Epoch
#   if args.model==1:
#     fig, axis = plt.subplots(1, 2, figsize=(20, 5))
#     axis[0].plot(history.history["unet3plus_output_final_activation_loss"], color='r', label = 'train loss')
#     axis[0].plot(history.history["val_unet3plus_output_final_activation_loss"], color='b', label = 'val loss')
#     axis[0].set_title('Loss Comparison')
#     axis[0].legend()
#     axis[1].plot(history.history["unet3plus_output_final_activation_acc"], color='r', label = 'train acc')
#     axis[1].plot(history.history["val_unet3plus_output_final_activation_acc"], color='b', label = 'val acc')
#     axis[1].set_title('Accuracy Comparison')
#     axis[1].legend()
#     plt.grid(False)
#   else:
#     fig, axis = plt.subplots(1, 2, figsize=(20, 5))
#     axis[0].plot(history.history["loss"], color='r', label='train loss')
#     axis[0].plot(history.history["val_loss"], color='b', label='val loss')
#     axis[0].set_title('Loss Comparison')
#     axis[0].legend()
#     axis[1].plot(history.history["acc"], color='r', label='train acc')
#     axis[1].plot(history.history["val_acc"], color='b', label='val acc')
#     axis[1].set_title('Accuracy Comparison')
#     axis[1].legend()
#     plt.grid(False)
#   fig.savefig("results/"+args.folder+"/graphs/graph_"+args.name+".png")

# model.save("/scratch/nuneslima/models/tensorflow/"+args.name+".h5")

  #Creation of Table with Test info and a summary of the Model
  make_prediction(args.name,args.folder,model, test_image, test_label, num_classes=num_classes)
  f = open("results/"+args.folder+"/tables/table_"+args.name+".txt", "a")
  model_info="\n\nModel: "+str(model.name)+"\nSlices: "+ str(slice_shape1)+"x"+str(slice_shape2)+"\nEpochs: "+str(args.epochs) + "\nDelta: "+ str(args.delta) + "\nPatience: " + str(args.patience)+ "\nBatch size: " + str(args.batch_size) + "\nOtimizador: " +str(opt_name) + "\nFunção de Perda: "+ str(loss_name)
  f.write(model_info)
  stride_info="\n\nStride Train: "+str(stride1)+"x"+str(args.stridetrain)+"\nStride Validation: "+str(stride1)+"x"+str(strideval2)+"\nStride Test: "+str(stride1)+"x"+str(stridetest2)
  f.write(stride_info)
  f.close()
