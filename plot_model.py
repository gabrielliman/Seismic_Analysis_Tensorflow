import argparse
import tensorflow as tf
import numpy as np
import os
from models.attention import Attention_unet
from models.unet import Unet
from models.unet3plus import Unet_3plus
from tensorflow.keras.utils import plot_model
from focal_loss import SparseCategoricalFocalLoss
from utils.datapreparation import my_division_data, article_division_data, penobscot_data
from utils.prediction import make_prediction
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--optimizer', '-o', metavar='O', type=int, default=0, help="Choose optimizer, 0: Adam, 1: SGD, 2: RMS")
    parser.add_argument('--gamma', '-g', metavar='G', type=float, default=3.6, help="Gamma for Sparce Categorical Focal Loss, deve ser um float")
    parser.add_argument('--model', '-m', metavar='M', type=int, default=2, help="Choose Segmentation Model, 0: Unet, 1: Unet 3 Plus, 2: Attention UNet")
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Limit of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--name', '-n', type=str, default="default", help='Model name for saving')
    parser.add_argument('--stride1', type=int, default=32, help="Stride in first dimension for train images")
    parser.add_argument('--stride2', type=int, default=32, help="Stride in second dimension for train images")
    parser.add_argument('--stridetest1', type=int, default=32, help="Stride in first dimension for test images")
    parser.add_argument('--stridetest2', type=int, default=32, help="Stride in second dimension for test images")
    parser.add_argument('--slice_shape1', '-s1',dest='slice_shape1', metavar='S', type=int, default=992, help='Shape 1 of the image slices')
    parser.add_argument('--slice_shape2', '-s2',dest='slice_shape2', metavar='S', type=int, default=576, help='Shape 2 of the image slices')
    parser.add_argument('--delta', '-d', type=float, default=1e-4, help="Delta for call back function")
    parser.add_argument('--patience', '-p', dest='patience', metavar='P', type=int, default=10, help="Patience for callback function")
    parser.add_argument('--loss_function', '-l', dest='loss_function', metavar='L', type=int, default=0, help="Choose loss function, 0= Cross Entropy, 1= Focal Loss")
    parser.add_argument('--folder', '-f', type=str, default="default_folder", help='Name of the folder where the results will be saved')
    parser.add_argument('--dataset', type=int, default=0, help="0: Parihaka 1: Penobscot 2: Netherlands F3")
    parser.add_argument('--kernel', type=int, default=7, help="kernel size")
    parser.add_argument('--gpuID', type=int, default=1, help="gpu id")
    parser.add_argument('--filters', type=int, default=6, help="num_filters")
    parser.add_argument('--dropout', type=float, default=0, help="Delta for call back function")

    return parser.parse_args()

if __name__ == '__main__':
  
  #Creation of image slices based on arguments
  args= get_args()
  slice_shape1=args.slice_shape1
  slice_shape2=args.slice_shape2
  stride1=args.stride1
  stride2=args.stride2
  stridetest1=args.stridetest1
  stridetest2=args.stridetest2
  if(args.dataset==0):
      num_classes=6
      train_image,train_label, test_image, test_label, val_image, val_label=my_division_data(shape=(slice_shape1,slice_shape2), stridetrain=(stride1,stride2), strideval=(stride1,stride2), stridetest=(stridetest1,stridetest2))
  elif(args.dataset==1):
      num_classes=8
      train_image,train_label, test_image, test_label, val_image, val_label=penobscot_data(shape=(slice_shape1,slice_shape2), stridetrain=(stride1,stride2), strideval=(stride1,stride2), stridetest=(stridetest1,stridetest2))
  elif(args.dataset==2):
      num_classes=6
      train_image,train_label, test_image, test_label, val_image, val_label=article_division_data(shape=(slice_shape1,slice_shape2), strideval=(230,14), stridetrain=(stride1,stride2))

  


  if args.gpuID == -1:
      os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
      print('CPU is used.')
  elif args.gpuID == 0:
      os.environ["CUDA_VISIBLE_DEVICES"] = "0"
      print('GPU device ' + str(args.gpuID) + ' is used.')
  elif args.gpuID == 1:
      os.environ["CUDA_VISIBLE_DEVICES"] = "1"
      print('GPU device ' + str(args.gpuID) + ' is used.')

  filters=[]
  for i in range(0,int(args.filters)):
      filters.append(2**(4+i))


  #Definition of Models
  if(args.model==0):
    model = Unet(tam_entrada=(slice_shape1, slice_shape2, 1), num_filtros=filters, classes=num_classes)
  elif(args.model==1):
    model = Unet_3plus(tam_entrada=(slice_shape1, slice_shape2, 1), n_filters=filters, classes=num_classes)
  elif(args.model==2):
    model = Attention_unet(tam_entrada=(slice_shape1, slice_shape2, 1), num_filtros=filters, classes=num_classes,kernel_size=args.kernel,dropout_rate=args.dropout)

  checkpoint_filepath = './checkpoints/'+args.folder+'/checkpoint_'+args.name +'.weights.h5'

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


  plot_model(model, to_file='unet_model.png', show_shapes=True, show_layer_names=True)