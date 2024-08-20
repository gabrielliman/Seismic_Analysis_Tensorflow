import argparse
import tensorflow as tf
import numpy as np
import os
from models.attention import Attention_unet
from focal_loss import SparseCategoricalFocalLoss
from utils.datapreparation import my_division_data, penobscot_data
from utils.generaluse import closest_odd_number
from utils.prediction import seisfacies_predict, calculate_class_info, calculate_macro_f1_score
import matplotlib.pyplot as plt
from functools import partial
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs


def train_opt(name,num_classes,slice_shape1,slice_shape2, callbacks,test_image,test_label,train_image, train_label, val_image, val_label,epochs,checkpoint_filepath,num_filters,dropout_rate, kernel_size, gamma, lr, batch_size, loss_function=1,optimizer=0):

    filters=[]
    for i in range(0,int(num_filters)):
       filters.append(2**(4+i))
    model=Attention_unet(tam_entrada=(slice_shape1,slice_shape2,1),num_filtros=filters,classes=num_classes, kernel_size=closest_odd_number(kernel_size), dropout_rate=dropout_rate)
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

    print(f"Test with gamma = {gamma}, learning_rate = {lr}, batch_size = {batch_size}, kernel_size = {kernel_size}, filters = {filters}, dropout rate = {dropout_rate}")

    #Model Compilation and Training
    model.compile(optimizer=opt,
                        loss=loss,
                      metrics=['acc'])

    history=model.fit(x=train_image, y=train_label, batch_size=int(batch_size), epochs=epochs,
                            callbacks=callbacks,
                            validation_data=(val_image, val_label))


    if os.path.exists(checkpoint_filepath):
        model.load_weights(checkpoint_filepath)
    
    predicted_label = seisfacies_predict(model,test_image)
    class_info, micro_f1=calculate_class_info(model, test_image, test_label, num_classes, predicted_label)
    macro_f1, class_f1=calculate_macro_f1_score(class_info, num_classes)
    with open("./bayes_opt/train_logs/"+str(name)+"_test.txt", "a") as f:
        f.write(f"Test with gamma = {gamma}, learning_rate = {lr}, batch_size = {batch_size}, kernel_size = {kernel_size}, filters = {filters}, dropout rate = {dropout_rate}")
        f.write('\nTest F1: '+ str(round(macro_f1,5)))
        f.write('\nTest accuracy: ' + str(round(micro_f1,5)))
        f.write('\n\n')
    if os.path.exists(checkpoint_filepath):
        os.remove(checkpoint_filepath)
    return macro_f1


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--optimizer', '-o', metavar='O', type=int, default=0, help="Choose optimizer, 0: Adam, 1: SGD, 2: RMS")
    parser.add_argument('--model', '-m', metavar='M', type=int, default=0, help="Choose Segmentation Model, 0: Unet, 1: Unet 3 Plus, 2: Attention UNet")
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Limit of epochs')
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
    parser.add_argument('--init_points', type=int, default=0, help="number of init points on bayes optimizer")
    parser.add_argument('--num_iter', type=int, default=0, help="number of iterations on bayes optimizer")
    parser.add_argument('--last_iter', type=str, default="", help="name of the file with last point on bayes optimizer")
    parser.add_argument('--dataset', type=int, default=0, help="0: Parihaka 1: Penobscot 2: Netherlands F3")
    parser.add_argument('--gpuID', type=int, default=1, help="gpu id")

    return parser.parse_args()

if __name__ == '__main__':
    
    #Creation of image slices based on arguments
    args= get_args()
    slice_shape1=args.slice_shape1
    slice_shape2=args.slice_shape2
    num_classes=6
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


    if args.gpuID == -1:
      os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
      print('CPU is used.')
    elif args.gpuID == 0:
      os.environ["CUDA_VISIBLE_DEVICES"] = "0"
      print('GPU device ' + str(args.gpuID) + ' is used.')
    elif args.gpuID == 1:
      os.environ["CUDA_VISIBLE_DEVICES"] = "1"
      print('GPU device ' + str(args.gpuID) + ' is used.')



    checkpoint_filepath = './checkpoints/'+args.folder+'/checkpoint_'+args.name+'.h5'

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    if not os.path.exists('./checkpoints/'+args.folder):
       os.makedirs('./checkpoints/'+args.folder)
    
    #cleaning file
    log_directory = "./bayes_opt/train_logs/"
    os.makedirs(log_directory, exist_ok=True)  
    with open("./bayes_opt/train_logs/"+str(args.name)+"_test.txt", "w") as f:
        f.close()



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
            monitor= 'val_unet3plus_output_sup2_activation_acc' if args.model == 3 else "val_acc",
            mode='max',
            save_best_only=True
        )
    ]

    fit_with_partial = partial(train_opt,args.name,num_classes,slice_shape1,slice_shape2,callbacks,test_image,test_label,train_image, train_label, val_image, val_label, args.epochs,checkpoint_filepath)
    #bounds definition
    bounds = {
        'num_filters'  :(3, 6.4),
        'gamma'        :(2, 10),
        'lr'           :(1e-4, 1e-2),
        'batch_size'   :(4, 20.001),
        'kernel_size'  :(2.99, 12.0),
        'dropout_rate' :(0.0,0.7)}
    
    bayes_optimizer = BayesianOptimization(
        f            = fit_with_partial,
        pbounds      = bounds,
        verbose      = 1,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        allow_duplicate_points=True)


    #loading previous logs
    if(args.last_iter==""):
        if os.path.exists("./bayes_opt/"+str(args.name)+"logs.log.json"):
            load_logs(bayes_optimizer, logs=["./bayes_opt/"+str(args.name)+"logs.log.json"])
            print("\n\n\nNew optimizer is now aware of {} points.\n\n\n".format(len(bayes_optimizer.space)))
    else:
        if os.path.exists("./bayes_opt/"+str(args.last_iter)+"logs.log.json"):
            load_logs(bayes_optimizer, logs=["./bayes_opt/"+str(args.last_iter)+"logs.log.json"])
            print("\n\n\nNew optimizer is now aware of {} points.\n\n\n".format(len(bayes_optimizer.space)))


    if os.path.exists(checkpoint_filepath):
        os.remove(checkpoint_filepath)


    #saving logs of the optimization
    logger = JSONLogger(path="./bayes_opt/"+str(args.name)+"logs.log")
    bayes_optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)


    ##i will run with 20,10 and 10,20 for now
    bayes_optimizer.maximize(init_points = args.init_points, n_iter = args.num_iter,)

    #saving results
    with open("./bayes_opt/train_logs/"+str(args.name)+"_results.txt", "a") as f:
        f.write("\n")
        for i, res in enumerate(bayes_optimizer.res):
            f.write(f"Iteration {i}: \n\t{res}")

        f.write(str(bayes_optimizer.max))
        f.close()
