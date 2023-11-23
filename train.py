import argparse
import tensorflow as tf
import numpy as np
import os
from models.attention import Attention_unet
from models.unet import Unet
from models.unet3plus import Unet_3plus
from focal_loss import SparseCategoricalFocalLoss
from utils.datapreparation import my_division_data
from utils.prediction import make_prediction, seisfacies_predict, calculate_class_info, calculate_macro_f1_score
# from models.bridgenet import BridgeNet_1
import matplotlib.pyplot as plt
from functools import partial
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

def train_opt(model,callbacks,test_image,test_label,train_image, train_label, val_image, val_label,epochs, gamma, lr, batch_size, loss_function=1,optimizer=0):
    print(gamma)
    print(batch_size)
    print(lr)
    #Definition of Optimizers
    if(optimizer==0):
        opt=tf.keras.optimizers.Adam(learning_rate=lr)
        opt_name="Adam"
    elif(optimizer==1):
        opt=tf.keras.optimizers.SGD(learning_rate=lr)
        opt_name="SGD"
    elif(optimizer==2):
        opt=tf.keras.optimizers.RMSprop(learning_rate=lr)
        opt_name="RMS"

    #Definition of Loss Function
    if(loss_function==0):
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
      loss_name="Sparce Categorical Cross Entropy"
    else:
      loss=SparseCategoricalFocalLoss(gamma=gamma, from_logits=True)
      loss_name="Sparce Categorical Focal Loss, Gamma: " + str(gamma)

    #Model Compilation and Training
    model.compile(optimizer=opt,
                        loss=loss,
                      metrics=['acc'])

    history=model.fit(x=train_image, y=train_label, batch_size=int(batch_size), epochs=epochs,
                            callbacks=callbacks,
                            validation_data=(val_image, val_label))
    predicted_label = seisfacies_predict(model,test_image)
    class_info, micro_f1=calculate_class_info(model, test_image, test_label, 6, predicted_label)
    macro_f1, class_f1=calculate_macro_f1_score(class_info)
    f = open("bayes_opt/first_test.txt", "a")
    f.write(f"TESTE COM GAMMA = {gamma}, learning_rate = {lr}, batch_size = {batch_size}")
    f.write('Test F1: '+ str(round(macro_f1,3)))
    f.write('\nTest accuracy: ' + str(round(micro_f1,3)))
    f.write('\n\n')
    return macro_f1

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
    parser.add_argument('--bayes_opt', type=bool, default=False, help='Activates the use of bayesian optimizer')
    parser.add_argument('--init_points', type=int, default=0, help="number of init points on bayes optimizer")
    parser.add_argument('--num_iter', type=int, default=0, help="number of iterations on bayes optimizer")
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
  if(args.model==0):
    model = Unet(tam_entrada=(slice_shape1, slice_shape2, 1), num_filtros=[16, 32, 64, 128, 256, 512], classes=num_classes)
  elif(args.model==1):
    model = Unet_3plus(tam_entrada=(slice_shape1, slice_shape2, 1), n_filters=[16, 32, 64, 128, 256], classes=num_classes)
  elif(args.model==2):
      model = Attention_unet(tam_entrada=(slice_shape1, slice_shape2, 1), num_filtros=[16, 32, 64, 128, 256, 512], classes=num_classes)
  #NAO FUNCIONA
  # elif(args.model==3):
  #     model = BridgeNet_1()

  checkpoint_filepath = './checkpoints/'+args.folder+'/checkpoint_'+args.name

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
  if(not args.bayes_opt):
    if not os.path.exists('./bayes_opt'):
     os.makedirs('./bayes_opt')
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

    history = model.fit(train_image, train_label, batch_size=args.batch_size, epochs=args.epochs,
                            callbacks=callbacks,
                            validation_data=(val_image, val_label))     
  #The best epoch is saved 
    model.load_weights(checkpoint_filepath)

    if not os.path.exists('./results/'+args.folder):
      os.makedirs('./results/'+args.folder)

    if not os.path.exists('./results/'+args.folder+'/graphs'):
      os.makedirs('./results/'+args.folder+'/graphs')
    
    if not os.path.exists('./results/'+args.folder+'/tables'):
          os.makedirs('./results/'+args.folder+'/tables')
    
    #Creation of training graphs with Loss and Accuracy, of Validation and Training, by Epoch
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

    #Creation of Table with Test info and a summary of the Model
    make_prediction(args.name,args.folder,model, test_image, test_label)
    f = open("results/"+args.folder+"/tables/table_"+args.name+".txt", "a")
    model_info="\n\nModel: "+str(model.name)+"\nSlices: "+ str(slice_shape1)+"x"+str(slice_shape2)+"\nEpochs: "+str(args.epochs) + "\nDelta: "+ str(args.delta) + "\nPatience: " + str(args.patience)+ "\nBatch size: " + str(args.batch_size) + "\nOtimizador: " +str(opt_name) + "\nFunção de Perda: "+ str(loss_name)
    f.write(model_info)
    stride_info="\n\nStride Train: "+str(stride1)+"x"+str(args.stridetrain)+"\nStride Validation: "+str(stride1)+"x"+str(strideval2)+"\nStride Test: "+str(stride1)+"x"+str(stridetest2)
    f.write(stride_info)
    f.close()

  else:
    fit_with_partial = partial(train_opt,model,callbacks,test_image,test_label,train_image, train_label, val_image, val_label, args.epochs)

    bounds = {
      'gamma'        :(0.1, 10),
      'lr'           :(1e-4, 1e-2),
      'batch_size'   :(1, 16.001)}
    bayes_optimizer = BayesianOptimization(
      f            = fit_with_partial,
      pbounds      = bounds,
      verbose      = 1,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
      random_state = 1)
    
    logger = JSONLogger(path="./bayes_opt/logs.log")
    
    bayes_optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    f = open("bayes_opt/first_test.txt", "w")
    f.close()
    bayes_optimizer.maximize(init_points = args.init_points, n_iter = args.num_iter,)
    f = open("bayes_opt/best_result.txt", "w")
    for i, res in enumerate(bayes_optimizer.res):
        f.write(f"Iteration {i}: \n\t{res}")

    f.write(bayes_optimizer.max)
    f.close()