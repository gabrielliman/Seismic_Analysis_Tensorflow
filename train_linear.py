import argparse
import tensorflow as tf
import numpy as np
import os
from models.simplecnn import simple_cnn
from sklearn.metrics import f1_score
# from focal_loss import SparseCategoricalFocalLoss
from utils.datapreparation import my_division_data, linear_data
# from utils.prediction import make_prediction
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def data_generator(images, labels, batch_size):
    # Reshape the images to have an additional dimension
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
    
    generator = ImageDataGenerator()  # No rescaling or augmentation
    generator = generator.flow(images, labels, batch_size=batch_size, shuffle=True)
    return generator


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--optimizer', '-o', metavar='O', type=int, default=0, help="Choose optimizer, 0: Adam, 1: SGD, 2: RMS")
    parser.add_argument('--gamma', '-g', metavar='G', type=float, default=2, help="Gamma for Sparce Categorical Focal Loss, deve ser um float")
    parser.add_argument('--model', '-m', metavar='M', type=int, default=0, help="Choose Segmentation Model, 0: Unet, 1: Unet 3 Plus, 2: Attention UNet")
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Limit of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=256, help='Batch size')
    parser.add_argument('--name', '-n', type=str, default="default", help='Model name for saving')
    parser.add_argument('--stridetrain', type=int, default=32, help="Stride in second dimension for train images")
    parser.add_argument('--slice_shape1', '-s1',dest='slice_shape1', metavar='S', type=int, default=50, help='Shape 1 of the image slices')
    parser.add_argument('--slice_shape2', '-s2',dest='slice_shape2', metavar='S', type=int, default=50, help='Shape 2 of the image slices')
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
    train_image,train_label, test_image, test_label, val_image, val_label=linear_data(shape=(slice_shape1,slice_shape2), stride=(50,50))
    train_image=train_image[:30000]
    train_label=train_label[:30000]
    test_image=test_image[:30000]
    test_label=test_label[:30000]
    val_image=val_image[:30000]
    val_label=val_label[:30000]
    train_image = train_image.reshape(train_image.shape[0], slice_shape1, slice_shape2, 1)
    model= simple_cnn(tam_entrada=(slice_shape1, slice_shape2, 1), num_classes=6)






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
    #Definition of Optimizers
    if(args.optimizer==0):
        opt=tf.keras.optimizers.Adam(learning_rate=1e-4)
        opt_name="Adam"
    # elif(args.optimizer==1):
    #     opt=tf.keras.optimizers.SGD()
    #     opt_name="SGD"
    # elif(args.optimizer==2):
    #     opt=tf.keras.optimizers.RMSprop()
    #     opt_name="RMS"

    #Definition of Loss Function
    if(args.loss_function==0):
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        loss_name="Sparce Categorical Cross Entropy"
    # else:
    #     loss=SparseCategoricalFocalLoss(gamma=args.gamma, from_logits=True)
    #     loss_name="Sparce Categorical Focal Loss, Gamma: " + str(args.gamma)

    #Model Compilation and Training
    model.compile(optimizer=opt,
                        loss=loss,
                        metrics=['acc'])

    batch_size = args.batch_size
    train_data_generator = data_generator(train_image, train_label, batch_size)
    val_data_generator = data_generator(val_image, val_label, batch_size)


    steps_per_epoch = len(train_image) // batch_size
    if(steps_per_epoch>20000): steps_per_epoch=20000
    validation_steps = len(val_image) // batch_size



    history = model.fit(train_data_generator, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
                    callbacks=callbacks, validation_data=val_data_generator, validation_steps=validation_steps) 
    


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
    print("################################\n")
    print(model.evaluate(test_image, test_label))

    predicted_labels=model.predict(test_image)
    f1 = f1_score(test_label, predicted_labels, average='weighted')  # Choose the appropriate average parameter

    print(f'Weighted F1 Score: {f1:.4f}')

    # model.save("/scratch/nuneslima/models/tensorflow/"+args.name+".h5")

    # #Creation of Table with Test info and a summary of the Model
    # make_prediction(args.name,args.folder,model, test_image, test_label)
    # f = open("results/"+args.folder+"/tables/table_"+args.name+".txt", "a")
    # model_info="\n\nModel: "+str(model.name)+"\nSlices: "+ str(slice_shape1)+"x"+str(slice_shape2)+"\nEpochs: "+str(args.epochs) + "\nDelta: "+ str(args.delta) + "\nPatience: " + str(args.patience)+ "\nBatch size: " + str(args.batch_size) + "\nOtimizador: " +str(opt_name) + "\nFunção de Perda: "+ str(loss_name)
    # f.write(model_info)
    # stride_info="\n\nStride Train: "+str(stride1)+"x"+str(args.stridetrain)+"\nStride Validation: "+str(stride1)+"x"+str(strideval2)+"\nStride Test: "+str(stride1)+"x"+str(stridetest2)
    # f.write(stride_info)
    # f.close()