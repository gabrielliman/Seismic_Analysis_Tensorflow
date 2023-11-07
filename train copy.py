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
from functools import partial
from bayes_opt import BayesianOptimization




def fit_with(model, input_shape, verbose, dropout2_rate, learning_rate, train_image, train_label, test_image, test_label, val_image, checkpoint_filepath):
    callbacks = [
      tf.keras.callbacks.EarlyStopping(
          # Stop training when `val_loss` is no longer improving
          monitor="val_loss",
          min_delta=1e-4,
          patience=5,
          verbose=1,
      ),
      tf.keras.callbacks.ModelCheckpoint(
          filepath=checkpoint_filepath,
          save_weights_only=True,
          monitor= "val_acc",
          mode='max',
          save_best_only=True
      )]

    # Train the model for a specified number of epochs.
    optimizer= Adam(lr = learning_rate)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(), 
                  loss=SparseCategoricalFocalLoss(gamma=4, from_logits=True), 
                  metrics = ['accuracy'])


    history = model.fit(train_image, train_label, batch_size=16, epochs=5,
                        callbacks=callbacks,
                        validation_data=(val_image, val_label))

    # Evaluate the model with the eval dataset.
    score = model.evaluate(test_image, test_label, steps = 10, verbose=1)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('\n')

    # Return the accuracy.

    return score[1]


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
    return parser.parse_args()

if __name__ == '__main__':
  
  #Creation of image slices based on arguments
  args= get_args()
  slice_shape1=args.slice_shape1
  slice_shape2=args.slice_shape2
  num_classes=6
  stride1=16
  strideval2=16
  stridetest2=16
  train_image,train_label, test_image, test_label, val_image, val_label=my_division_data(shape=(slice_shape1,slice_shape2), stridetrain=(stride1,args.stridetrain), strideval=(stride1,strideval2), stridetest=(stride1,stridetest2))
  
  #Definition of Models
  model = Unet(tam_entrada=(slice_shape1, slice_shape2, 1), num_filtros=[16, 32, 64, 128, 256, 512], classes=num_classes)

  checkpoint_filepath = './checkpoints/'+args.folder+'/checkpoint_'+args.name

  if not os.path.exists('./checkpoints/'+args.folder):
     os.makedirs('./checkpoints/'+args.folder)

  input_shape=(992,192,1)
  verbose=1
  fit_with_partial = partial(fit_with, input_shape, verbose)

  bounds = {'learning_rate'           :(1e-4, 1e-2),
          'dropout2_rate':(0.05, 0.5),
          'batch_size'   :(1, 4.001),
          'num_filters'  :(1, 4.001),
          'kernel_size'  :(2, 4.001)}
          

  bounds_2 = {
            'learning_rate'           : (1e-4, 1e-2)}



  optimizer = BayesianOptimization(
      f            = fit_with_partial,
      pbounds      = bounds_2,
      verbose      = 1,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
      random_state = 1
  )

  optimizer.maximize(init_points = 10, n_iter = 2,)

  for i, res in enumerate(optimizer.res):
      print("Iteration {}: \n\t{}".format(i, res))

  print(optimizer.max)






#  #The best epoch is saved 
#   model.load_weights(checkpoint_filepath)

#   if not os.path.exists('./results/'+args.folder):
#     os.makedirs('./results/'+args.folder)

#   if not os.path.exists('./results/'+args.folder+'/graphs'):
#     os.makedirs('./results/'+args.folder+'/graphs')
  
#   if not os.path.exists('./results/'+args.folder+'/tables'):
#         os.makedirs('./results/'+args.folder+'/tables')
  
  #Creation of training graphs with Loss and Accuracy, of Validation and Training, by Epoch
  # if args.model==1:
  #   fig, axis = plt.subplots(1, 2, figsize=(20, 5))
  #   axis[0].plot(history.history["unet3plus_output_final_activation_loss"], color='r', label = 'train loss')
  #   axis[0].plot(history.history["val_unet3plus_output_final_activation_loss"], color='b', label = 'val loss')
  #   axis[0].set_title('Loss Comparison')
  #   axis[0].legend()
  #   axis[1].plot(history.history["unet3plus_output_final_activation_acc"], color='r', label = 'train acc')
  #   axis[1].plot(history.history["val_unet3plus_output_final_activation_acc"], color='b', label = 'val acc')
  #   axis[1].set_title('Accuracy Comparison')
  #   axis[1].legend()
  #   plt.grid(False)
  # else:
  #   fig, axis = plt.subplots(1, 2, figsize=(20, 5))
  #   axis[0].plot(history.history["loss"], color='r', label='train loss')
  #   axis[0].plot(history.history["val_loss"], color='b', label='val loss')
  #   axis[0].set_title('Loss Comparison')
  #   axis[0].legend()
  #   axis[1].plot(history.history["acc"], color='r', label='train acc')
  #   axis[1].plot(history.history["val_acc"], color='b', label='val acc')
  #   axis[1].set_title('Accuracy Comparison')
  #   axis[1].legend()
  #   plt.grid(False)
  # fig.savefig("results/"+args.folder+"/graphs/graph_"+args.name+".png")

# model.save("/scratch/nuneslima/models/tensorflow/"+args.name+".h5")

  #Creation of Table with Test info and a summary of the Model
  # make_prediction(args.name,areval_dsgs.folder,model, test_image, test_label)
  # f = open("results/"+args.folder+"/tables/table_"+args.name+".txt", "a")
  # model_info="\n\nModel: "+str(model.name)+"\nSlices: "+ str(slice_shape1)+"x"+str(slice_shape2)+"\nEpochs: "+str(args.epochs) + "\nDelta: "+ str(args.delta) + "\nPatience: " + str(args.patience)+ "\nBatch size: " + str(args.batch_size) + "\nOtimizador: " +str(opt_name) + "\nFunção de Perda: "+ str(loss_name)
  # f.write(model_info)
  # stride_info="\n\nStride Train: "+str(stride1)+"x"+str(args.stridetrain)+"\nStride Validation: "+str(stride1)+"x"+str(strideval2)+"\nStride Test: "+str(stride1)+"x"+str(stridetest2)
  # f.write(stride_info)
  # f.close()