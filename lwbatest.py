from utils.datapreparation import extract_patches
import lwbna_unet as unet
import numpy as np
import tensorflow as tf
import os
from utils.prediction import make_prediction

def scale_to_1(array):
    min_val = np.min(array)
    max_val = np.max(array)

    # Scale the array to the range [0, 255]
    scaled_array = ((array - min_val) / (max_val - min_val))

    # Round to integers
    scaled_array = scaled_array.astype(np.float64)

    return scaled_array

def my_division_data(shape=(992,192),stridetest=(128,64), strideval=(128,64), stridetrain=(128,64)):
    read_seis_data = np.load(
        '/scratch/nunes/seismic/data_train.npz', 
                allow_pickle=True, mmap_mode = 'r')
    # We read our labels
    read_labels = np.load(
        '/scratch/nunes/seismic/labels_train.npz',
                    allow_pickle=True, mmap_mode = 'r')

    # Inside the elements we pick what we are interesed in
    seis_data = read_seis_data['data']
    seis_data=scale_to_1(seis_data)
    labels = read_labels['labels']
    labels[labels==6] = 0

    testcrossline=seis_data[:,702:782,:]
    testinline=seis_data[:,:,510:590]
    testcrossline_label=labels[:,702:782,:]
    testinline_label=labels[:,:,510:590]


    #removing the test data the rest of our data has shape Z=1006 X=702 Y=510
    valcrossline=seis_data[:,622:702,:510]
    valinline=seis_data[:,:702,430:510]
    valcrossline_label=labels[:,622:702,:510]
    valinline_label=labels[:,:702,430:510]


    ##removing the validation data the rest of our data has shape Z=1006 X=622 Y=430
    traindata=seis_data[:,:622,:430]
    trainlabel=labels[:,:622,:430]


    #TRAINING
    trainpatches=[]
    trainlabels=[]
    for i in (range(traindata.shape[2])):
        trainpatches=trainpatches+extract_patches(traindata[:,:,i],(shape),(stridetrain))
        trainlabels=trainlabels+extract_patches(trainlabel[:,:,i],(shape),(stridetrain))
    for i in (range(traindata.shape[1])):
        trainpatches=trainpatches+extract_patches(traindata[:,i,:],(shape),(stridetrain))
        trainlabels=trainlabels+extract_patches(trainlabel[:,i,:],(shape),(stridetrain))
    trainslices=np.array(trainpatches)
    trainlabels=np.array(trainlabels)

    #VALIDATION
    valpatches=[]
    vallabels=[]
    for i in (range(valinline.shape[2])):
        valpatches=valpatches+extract_patches(valinline[:,:,i],(shape),(strideval))
        vallabels=vallabels+extract_patches(valinline_label[:,:,i],(shape),(strideval))
    for i in (range(valcrossline.shape[1])):
        valpatches=valpatches+extract_patches(valcrossline[:,i,:],(shape),(strideval))
        vallabels=vallabels+extract_patches(valcrossline_label[:,i,:],(shape),(strideval))
    valslices=np.array(valpatches)
    vallabels=np.array(vallabels)

    #TEST
    testpatches=[]
    testlabels=[]
    for i in (range(testinline.shape[2])):
        testpatches=testpatches+extract_patches(testinline[:,:,i],(shape),(stridetest))
        testlabels=testlabels+extract_patches(testinline_label[:,:,i],(shape),(stridetest))
    for i in (range(testcrossline.shape[1])):
        testpatches=testpatches+extract_patches(testcrossline[:,i,:],(shape),(stridetest))
        testlabels=testlabels+extract_patches(testcrossline_label[:,i,:],(shape),(stridetest))
    testslices=np.array(testpatches)
    testlabels=np.array(testlabels)

    return np.expand_dims(trainslices, axis=-1),np.expand_dims(trainlabels, axis=-1),np.expand_dims(testslices, axis=-1),np.expand_dims(testlabels, axis=-1),np.expand_dims(valslices, axis=-1),np.expand_dims(vallabels, axis=-1)


checkpoint_filepath = './checkpoints/'+"lwbna_unet"+'/checkpoint_'+"first_test" +'.weights.h5'

if not os.path.exists('./checkpoints'):
     os.makedirs('./checkpoints')

if not os.path.exists('./checkpoints/'+"lwbna_unet"):
     os.makedirs('./checkpoints/'+"lwbna_unet")
callbacks = [
    tf.keras.callbacks.EarlyStopping(
          # Stop training when `val_loss` is no longer improving
          monitor="val_loss",
          min_delta=1e-4,
          patience=15,
          verbose=1,
      ),
    tf.keras.callbacks.ModelCheckpoint(
          filepath=checkpoint_filepath,
          save_weights_only=True,
          monitor= "val_acc",
          mode='max',
          save_best_only=True
      )
  ]
  #Definition of Optimizers

opt=tf.keras.optimizers.Adam(learning_rate=1e-4)
opt_name="Adam"

  #Definition of Loss Function
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
loss_name="Sparce Categorical Cross Entropy"

trainslices, trainlabels, testslices, testlabels, valslices, vallabels=my_division_data(shape=(992,192), stridetrain=(128,64), strideval=(128,64), stridetest=(128,64))

my_unet = unet.LWBNAUnet(
    n_classes=6, 
    filters=64, 
    depth=4, 
    midblock_steps=4, 
    dropout_rate=0.3, 
    name="my_unet"
)

my_unet.build(input_shape=(16,992,192,1))

my_unet.compile(optimizer=opt,
                    loss=loss,
                    metrics=['acc'])

history = my_unet.fit(trainslices, trainlabels, batch_size=16, epochs=100,
                          callbacks=callbacks,
                          validation_data=(valslices, vallabels))

my_unet.load_weights(checkpoint_filepath)


if not os.path.exists('./results/'+"lwbna_unet"):
    os.makedirs('./results/'+"lwbna_unet")

if not os.path.exists('./results/'+"lwbna_unet"+'/graphs'):
    os.makedirs('./results/'+"lwbna_unet"+'/graphs')
  
if not os.path.exists('./results/'+"lwbna_unet"+'/tables'):
    os.makedirs('./results/'+"lwbna_unet"+'/tables')

make_prediction("my_unet","lwbna_unet",my_unet, testslices, testlabels, 6)