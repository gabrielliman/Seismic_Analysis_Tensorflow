from pyseismic_local.AI_ML_DL_functions import *
from pyseismic_local.date_time_functions import *
from pyseismic_local.AI_SeismicFaciesClassification import *
from pyseismic_local.models_networks_ANNs.FlexibleBridgeNet_Keras import *
from pyseismic_local.public_functions import *

import argparse
import time

import numpy as np

from keras.models import load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.optimizers import Adam

################################################################################
data_time_str = data_time_str_def()

X_channels = 1
np.random.seed(seed=0)  # for reproducibility
random.seed(0)
# Params
parser = argparse.ArgumentParser()
# Common
parser.add_argument('--gpuID', default=0, type=int, help='gpuID')
parser.add_argument('--training_data_disk', nargs='?', type=str, default='D', help='training_data_disk')
parser.add_argument('--models_disk', nargs='?', type=str, default='F', help='models_disk')
parser.add_argument('--training_number', default=10000, type=int, help='training_number')
parser.add_argument('--epochs', default=3, type=int, help='epochs')
parser.add_argument('--batch_size', default=16, type=int, help='batch_size')

parser.add_argument('--loss_used', default=5, type=int, help='loss_used')
parser.add_argument('--lr', default=0.000100, type=float, help='lr')
parser.add_argument('--kernel_size', default=11, type=int, help='kernel_size')
parser.add_argument('--BridgeNet_used', default=5, type=int, help='BridgeNet_used')

# Mostly unchanged
parser.add_argument('--nD', default=2, type=int, help='nD')
parser.add_argument('--continue_train', action='store_true', help='continue_train')
parser.add_argument('--save_every', default=1, type=int, help='save_every')
parser.add_argument('--patch_rows', default=992, type=int, help='patch_rows')
parser.add_argument('--patch_cols', default=576, type=int, help='patch_cols')
parser.add_argument('--stride', default=[15, 5], type=int, help='stride', nargs='*')
parser.add_argument('--Y_channels', default=1, type=int, help='Y_channels')
parser.add_argument('--Y_channels_model', default=1, type=int, help='Y_channels_model')
parser.add_argument('--X_normal', default=5522.086, type=float, help='X_normal')
parser.add_argument('--plot_show', action='store_true', help='plot_show')
parser.add_argument('--kernels_all', default=[16, 32, 64, 128, 256, 512], type=int, nargs='*')
parser.add_argument('--conv2act_repeat', default=2, type=int, help='conv2act_repeat')
parser.add_argument('--reproduce', default=1, type=int, help='reproduce')
parser.add_argument('--res_case', default=0, type=int, help='res_case')
parser.add_argument('--res_number', default=0, type=int, help='res_number')




parser.add_argument('--optimizer', '-o', metavar='O', type=int, default=0, help="Choose optimizer, 0: Adam, 1: SGD, 2: RMS")
parser.add_argument('--name', '-n', type=str, default="default", help='Model name for saving')
parser.add_argument('--stride1', type=int, default=32, help="Stride in first dimension for train images")
parser.add_argument('--stride2', type=int, default=32, help="Stride in second dimension for train images")
parser.add_argument('--stridetest1', type=int, default=32, help="Stride in first dimension for test images")
parser.add_argument('--stridetest2', type=int, default=32, help="Stride in second dimension for test images")
parser.add_argument('--slice_shape1', '-s1',dest='slice_shape1', metavar='S', type=int, default=992, help='Shape 1 of the image slices')
parser.add_argument('--slice_shape2', '-s2',dest='slice_shape2', metavar='S', type=int, default=576, help='Shape 2 of the image slices')
parser.add_argument('--delta', '-d', type=float, default=1e-4, help="Delta for call back function")
parser.add_argument('--patience', '-p', dest='patience', metavar='P', type=int, default=10, help="Patience for callback function")
parser.add_argument('--folder', '-f', type=str, default="default_folder", help='Name of the folder where the results will be saved')
parser.add_argument('--dataset', type=int, default=0, help="0: Parihaka 1: Penobscot")
parser.add_argument('--sizetrainx', type=int, default=192, help="size of x dimension of training for progressive training")
parser.add_argument('--sizetrainy', type=int, default=192, help="size of x dimension of training for progressive training")
parser.add_argument('--test_pos', type=str, default="end", help='position of the test data relative to the data not used for training, "start,mid or end"')
parser.add_argument('--num_extra_train', type=int, default=1, help="Number of extra slices classified to improve training")

args = parser.parse_args()
print('args.loss_used: ' + str(args.loss_used))
if args.loss_used == 5:
    Y_1 = 1
    args.act_last = 3
    args.Y_channels_model = 6
else:
    Y_1 = 0
print('args.act_last: ' + str(args.act_last))
print('args.Y_channels_model: ' + str(args.Y_channels_model))
################################################################################
import os

if args.gpuID == -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('CPU is used.')
elif args.gpuID == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('GPU device ' + str(args.gpuID) + ' is used.')
elif args.gpuID == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('GPU device ' + str(args.gpuID) + ' is used.')

################################################################################
disk_ID_data = args.training_data_disk
disk_ID_model = args.models_disk

module_name = '/scratch/nunes/data_parihaka_article'
module_data_dir = module_name
module_model_dir = module_name
if not os.path.exists(module_model_dir):
    os.mkdir(module_model_dir)
training_data_dir = module_data_dir + '/' + 'training_data_' + str(int(args.patch_rows)) + 'x' + str(int(args.patch_cols)) + '_stride' + str(args.stride[0]) + '_' + str(
    args.stride[1]) + '_Ychannels' + str(args.Y_channels)
training_data_path = training_data_dir + '/'
print(training_data_path)
validation_data_dir = module_data_dir + '/' + 'validation_data_' + str(int(args.patch_rows)) + 'x' + str(int(args.patch_cols)) + '_Ychannels' + str(args.Y_channels)
validation_data_path = validation_data_dir + '/'

models_dir = module_model_dir + '/' + 'models'
if not os.path.exists(models_dir):
    os.mkdir(models_dir)


model_log_dir = models_dir #+ '/' + model_folder_name

if not os.path.exists(model_log_dir):
    os.mkdir(model_log_dir)


def lr_scheduler(epoch):
    initial_lr = args.lr
    lr = initial_lr

    log('current learning rate is %2.11f' % lr)
    return lr


if __name__ == '__main__':
    start_time = time.time()
    model = FlexibleBridgeNet(input_size=(992,192,1),up_down_times=args.BridgeNet_used, Y_channels=args.Y_channels_model, kernel_size=args.kernel_size,
                              kernels_all=args.kernels_all[0:(args.BridgeNet_used + 1)], conv2act_repeat=args.conv2act_repeat, res_case=args.res_case,
                              res_number=args.res_number)

    print('kernels_all: ' + str(args.kernels_all[0:(args.BridgeNet_used + 1)]))

    # Regression Loss Function
    import tensorflow as tf
    if args.loss_used == 1:
        loss_used = 'mean_squared_error'
    elif args.loss_used == 2:
        loss_used = 'mean_absolute_error'
    elif args.loss_used == 3:
        loss_used = 'mean_absolute_percentage_error'
    elif args.loss_used == 4:
        loss_used = 'mean_squared_logarithmic_error'
    # Multi-class Classification Loss
    elif args.loss_used == 5:
        loss_used = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    elif args.loss_used == 6:
        loss_used = 'categorical_crossentropy'


    model.compile(optimizer=Adam(learning_rate=args.lr), loss=loss_used, metrics=['accuracy'])




    from myutils.prediction import make_prediction

    from myutils.datapreparation import my_division_data
    train_image,train_label, test_image, test_label, val_image, val_label=my_division_data(shape=(992,192), stridetrain=(128,64), strideval=(128,64), stridetest=(128,64))


    actual_training_number = (len(train_image) // args.batch_size) * args.batch_size
    print('actual_training_number: ' + str(actual_training_number))

    steps_per_epoch = (actual_training_number // args.batch_size)



    callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_log_dir, 'mydivision_model.keras'),
        save_weights_only=True,
        monitor= "accuracy",
        mode='max',
        save_best_only=True)]


    history = model.fit(train_image, train_label, steps_per_epoch=steps_per_epoch, batch_size=args.batch_size, epochs=args.epochs,
                        callbacks=callbacks,
                        validation_data=(val_image, val_label))     


    elapsed_time = time.time() - start_time
    elapsed_time_str = time2HMS(elapsed_time=elapsed_time)

    try:
        model.load_weights(os.path.join(model_log_dir, 'mydivision_model.keras'))
    finally:
    #model.summary()
        model.save("/scratch/nunes/seismic_models/"+"article_model"+"_"+"my_division"+".keras")


    with open(os.path.join(model_log_dir, data_time_str + '_elapsed_time_' + elapsed_time_str + '.txt'), "w") as f:
        f.write(str(elapsed_time) + ' seconds    = ' + elapsed_time_str)

    make_prediction("article_model_my_division","parihaka", model, test_image, test_label, num_classes=6)