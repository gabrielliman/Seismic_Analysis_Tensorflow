import argparse
import tensorflow as tf
import os
from utils.generaluse import scale_all_data
from models.attention import Attention_unet
from models.unet import Unet
from models.unet3plus import Unet_3plus
from models.bridgenet import FlexibleBridgeNet
from models.efficientNetB1 import EfficientNetB1
from focal_loss import SparseCategoricalFocalLoss
from utils.datapreparation import *
from utils.prediction import make_prediction
import matplotlib.pyplot as plt
import lwbna_unet as lwba


from network.CFPNetM import CFPNetM
from network.ENet import ENet
from network.ESPNet import ESPNet
from network.ICNet import ICNet


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
    parser.add_argument('--sizetrainx', type=int, default=192, help="size of x dimension of training for progressive training")
    parser.add_argument('--sizetrainy', type=int, default=192, help="size of x dimension of training for progressive training")
    parser.add_argument('--dropout', type=float, default=0, help="Delta for call back function")
    parser.add_argument('--test_pos', type=str, default="end", help='position of the test data relative to the data not used for training, "start,mid or end"')
    parser.add_argument('--num_extra_train', type=int, default=1, help="Number of extra slices classified to improve training")

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
        train_image,train_label, test_image, test_label, val_image, val_label=LRP_Parihaka(shape=(slice_shape1,slice_shape2), stridetrain=(stride1,stride2), strideval=(stride1,stride2), stridetest=(stridetest1,stridetest2))
    elif(args.dataset==1):
        num_classes=8
        train_image,train_label, test_image, test_label, val_image, val_label=LRP_Penobscot(shape=(slice_shape1,slice_shape2), stridetrain=(stride1,stride2), strideval=(stride1,stride2), stridetest=(stridetest1,stridetest2))
    elif(args.dataset==2):
        num_classes=6
        train_image,train_label, test_image, test_label, val_image, val_label=RPRV_Parihaka(shape=(slice_shape1,slice_shape2), stridetrain=(stride1,stride2), strideval=(stride1,stride2), stridetest=(stridetest1,stridetest2), sizetrain_x=args.sizetrainx, sizetrain_y=args.sizetrainy, test_pos=args.test_pos)
    elif (args.dataset==3):
        num_classes=8
        train_image,train_label, test_image, test_label, val_image, val_label=RPRV_Penobscot(shape=(slice_shape1,slice_shape2), stridetrain=(stride1,stride2), strideval=(stride1,stride2), stridetest=(stridetest1,stridetest2), sizetrain_x=args.sizetrainx, sizetrain_y=args.sizetrainy, test_pos=args.test_pos)
    elif(args.dataset==4):
        num_classes=6
        train_image,train_label, test_image, test_label, val_image, val_label=RPEDS_Parihaka(shape=(slice_shape1,slice_shape2), stridetrain=(stride1,stride2), strideval=(stride1,stride2), stridetest=(stridetest1,stridetest2), sizetrain_x=args.sizetrainx, sizetrain_y=args.sizetrainy, num_extra_train=args.num_extra_train)
    elif(args.dataset==5):
        num_classes=8
        train_image,train_label, test_image, test_label, val_image, val_label=RPEDS_Penobscot(shape=(slice_shape1,slice_shape2), stridetrain=(stride1,stride2), strideval=(stride1,stride2), stridetest=(stridetest1,stridetest2), sizetrain_x=args.sizetrainx, sizetrain_y=args.sizetrainy, num_extra_train=args.num_extra_train)
    elif(args.dataset==6):
        num_classes=6
        train_image,train_label, test_image, test_label, val_image, val_label=EDS_Parihaka(shape=(slice_shape1,slice_shape2), stridetrain=(stride1,stride2), strideval=(stride1,stride2), stridetest=(stridetest1,stridetest2), num_train=args.num_extra_train)
    elif (args.dataset==7):
        num_classes=8
        train_image,train_label, test_image, test_label, val_image, val_label=EDS_Penobscot(shape=(slice_shape1,slice_shape2), stridetrain=(stride1,stride2), strideval=(stride1,stride2), stridetest=(stridetest1,stridetest2), num_train=args.num_extra_train)

    print(train_image.shape)

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

    print(train_image.shape)
    #Definition of Models
    if(args.model== 0):
        model = Unet(tam_entrada=(slice_shape1, slice_shape2, 1), num_filtros=filters, classes=num_classes,kernel_size=args.kernel,dropout_rate=args.dropout)
    elif(args.model== 1):
        model = Unet_3plus(tam_entrada=(slice_shape1, slice_shape2, 1), n_filters=filters[:-1], classes=num_classes,kernel_size=args.kernel,dropout_rate=args.dropout)
    elif(args.model== 2):
        model = Attention_unet(tam_entrada=(slice_shape1, slice_shape2, 1), num_filtros=filters, classes=num_classes,kernel_size=args.kernel,dropout_rate=args.dropout)
    elif(args.model== 3):
        model = FlexibleBridgeNet(input_size=(slice_shape1,slice_shape2,1),up_down_times=5, Y_channels=num_classes, kernel_size=args.kernel,
                                kernels_all=[16, 32, 64, 128, 256, 512][0:6], conv2act_repeat=2, res_case=0,
                                res_number=0)
    elif(args.model== 4):
        model = CFPNetM(slice_shape1, slice_shape2, 1, num_classes)


    elif(args.model== 6):
        model = ENet(slice_shape1, slice_shape2, 1, num_classes)

    elif(args.model== 7):
        model = ESPNet(slice_shape1, slice_shape2, 1, num_classes)

    elif(args.model== 8):
        model = ICNet(slice_shape1, slice_shape2, 1, num_classes)

        
    elif(args.model== 10):
        model = EfficientNetB1(slice_shape1,slice_shape2, 1, num_classes)

    elif(args.model== 11):
        model = lwba.LWBNAUnet(
                n_classes=num_classes, 
                filters=64, 
                depth=4, 
                midblock_steps=4, 
                dropout_rate=0.3, 
                name="lwbna_unet"
            )
        model.build(input_shape=(args.batch_size,slice_shape1, slice_shape2,1))
        train_image, test_image, val_image = scale_all_data(
            (train_image, test_image, val_image))
        

    checkpoint_filepath = '/scratch/nunes/base_checkpoints/'+args.folder+'/checkpoint_'+args.name +'.weights.h5'

    if not os.path.exists('/scratch/nunes/base_checkpoints'):
        os.makedirs('/scratch/nunes/base_checkpoints')

    if not os.path.exists('/scratch/nunes/base_checkpoints/'+args.folder):
        os.makedirs('/scratch/nunes/base_checkpoints/'+args.folder)

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

    #model.save("/scratch/nunes/seismic_models/"+args.folder+"_"+args.name+".keras")

    #Creation of Table with Test info and a summary of the Model
    make_prediction(args.name,args.folder,model, test_image[:100], test_label[:100], num_classes)
    f = open("results/"+args.folder+"/tables/table_"+args.name+".txt", "a")
    model_info="\n\nModel: "+str(model.name)+"\nSlices: "+ str(slice_shape1)+"x"+str(slice_shape2)+"\nEpochs: "+str(args.epochs) + "\nDelta: "+ str(args.delta) + "\nPatience: " + str(args.patience)+ "\nBatch size: " + str(args.batch_size) + "\nOtimizador: " +str(opt_name) + "\nFunção de Perda: "+ str(loss_name)
    f.write(model_info)
    stride_info="\n\nStride Train: "+str(stride1)+"x"+str(stride2)+"\nStride Validation: "+str(stride1)+"x"+str(stride2)+"\nStride Test: "+str(stridetest1)+"x"+str(stridetest2)
    f.write(stride_info)
    f.close()