import argparse
import tensorflow as tf
import numpy as np
import os
from models.attention import Attention_unet
from models.unet import Unet
from models.unet3plus import Unet_3plus
from focal_loss import SparseCategoricalFocalLoss
from utils.datapreparation import my_division_data
from utils.prediction import make_prediction
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--optimizer', '-o', metavar='O', type=int, default=0, help="Optimizer")
    parser.add_argument('--gamma', '-g', metavar='G', type=float, default=2, help="Gamma for focal loss")
    parser.add_argument('--model', '-m', metavar='M', type=int, default=0, help="which model")
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--name', '-n', type=str, default="teste", help='Model name for saving')
    parser.add_argument('--stridetrain', type=int, default=32)
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--slice_shape1', '-s1',dest='slice_shape1', metavar='S', type=int,default=992, help='Shape 1 of the slices used in training and validation')
    parser.add_argument('--slice_shape2', '-s2',dest='slice_shape2', metavar='S', type=int,default=576, help='Shape 2 of the slices used in training and validation')
    parser.add_argument('--delta', '-d', type=float, default=1e-4, help="Delta for call back function")
    parser.add_argument('--patience', '-p', dest='patience', metavar='P', type=int, default=10, help="Patience for callback function")
    parser.add_argument('--loss_function', '-l', dest='loss_function', metavar='L', type=int, default=0, help="Choose loss function, 0= Cross Entropy, 1= Focal Loss")
    parser.add_argument('--folder', '-f', type=str, default="test_folder", help='Name of the folder where the results will be saved')
    return parser.parse_args()

if __name__ == '__main__':
  args= get_args()
  slice_shape1=args.slice_shape1
  slice_shape2=args.slice_shape2
  num_classes=6
  stride1=230
  strideval2=16
  stridetest2=100
  train_image,train_label, test_image, test_label, val_image, val_label=my_division_data(shape=(slice_shape1,slice_shape2), stridetrain=(stride1,args.stridetrain), strideval=(stride1,strideval2), stridetest=(stride1,stridetest2))
  if(args.model==0):
    model = Unet(tam_entrada=(slice_shape1, slice_shape2, 1), num_filtros=[16, 32, 64, 128, 256, 512], classes=num_classes)
  elif(args.model==1):
    model = Unet_3plus(input_size=(slice_shape1, slice_shape2, 1), n_filters=[16, 32, 64, 128, 256], classes=num_classes)
  elif(args.model==2):
      model = Attention_unet(tam_entrada=(slice_shape1, slice_shape2, 1), num_filtros=[16, 32, 64, 128, 256, 512], classes=num_classes)
  checkpoint_filepath = './checkpoints/'+args.folder+'/checkpoint_'+args.name
  
  if not os.path.exists('./checkpoints/'+args.folder):
     os.makedirs('./checkpoints/'+args.folder)
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
  if(args.optimizer==0):
     opt=tf.keras.optimizers.Adam()
     opt_name="Adam"
  elif(args.optimizer==1):
     opt=tf.keras.optimizers.SGD()
     opt_name="SGD"
  elif(args.optimizer==2):
     opt=tf.keras.optimizers.RMSprop()
     opt_name="RMS"

  if(args.loss_function==0):
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    loss_name="Sparce Categorical Cross Entropy"
  else:
    loss=SparseCategoricalFocalLoss(gamma=args.gamma, from_logits=True)
    loss_name="Sparce Categorical Focal Loss, Gamma: " + str(args.gamma)

  model.compile(optimizer=opt,
                      loss=loss,
                    metrics=['acc'])

  history = model.fit(train_image, train_label, batch_size=args.batch_size, epochs=args.epochs,
                          callbacks=callbacks,
                          validation_data=(val_image, val_label))
  
  model.load_weights(checkpoint_filepath)

  if not os.path.exists('./results/'+args.folder):
    os.makedirs('./results/'+args.folder)

  if not os.path.exists('./results/'+args.folder+'/graphs'):
    os.makedirs('./results/'+args.folder+'/graphs')
  
  if not os.path.exists('./results/'+args.folder+'/tables'):
        os.makedirs('./results/'+args.folder+'/tables')
  
  if args.model==1:
    fig, axis = plt.subplots(1, 2, figsize=(20, 5))
    axis[0].plot(history.history["unet3plus_output_final_activation_loss"], color='r', label = 'train loss')
    axis[0].plot(history.history["val_unet3plus_output_final_activation_loss"], color='b', label = 'val loss')
    axis[0].set_title('Loss Comparison')
    axis[0].legend()
    axis[1].plot(history.history["unet3plus_output_final_activation_acc"], color='r', label = 'train acc')
    axis[1].plot(history.history["val_unet3plus_output_final_activation_acc"], color='b', label = 'val acc')
    axis[1].set_title('Accuracy Comparison')
    axis[1].legend()
    plt.grid(False)
  else:
    fig, axis = plt.subplots(1, 2, figsize=(20, 5))
    axis[0].plot(history.history["loss"], color='r', label='train loss')
    axis[0].plot(history.history["val_loss"], color='b', label='val loss')
    axis[0].set_title('Loss Comparison')
    axis[0].legend()
    axis[1].plot(history.history["acc"], color='r', label='train acc')
    axis[1].plot(history.history["val_acc"], color='b', label='val acc')
    axis[1].set_title('Accuracy Comparison')
    axis[1].legend()
    plt.grid(False)
  
  fig.savefig("results/"+args.folder+"/graphs/graph_"+args.name+".png")


  # model.save("/scratch/nuneslima/models/tensorflow/"+args.name+".h5")
  make_prediction(args.name,args.folder,model, test_image, test_label)
  f = open("results/"+args.folder+"/tables/table_"+args.name+".txt", "a")
  model_info="\n\nModel: "+str(model.name)+"\nSlices: "+ str(slice_shape1)+"x"+str(slice_shape2)+"\nEpochs: "+str(args.epochs) + "\nDelta: "+ str(args.delta) + "\nPatience: " + str(args.patience)+ "\nBatch size: " + str(args.batch_size) + "\nOtimizador: " +str(opt_name) + "\nFunção de Perda: "+ str(loss_name)
  f.write(model_info)
  stride_info="\n\nStride Train: "+str(stride1)+"x"+str(args.stridetrain)+"\nStride Validation: "+str(stride1)+"x"+str(strideval2)+"\nStride Test: "+str(stride1)+"x"+str(stridetest2)
  f.write(stride_info)
  f.close()