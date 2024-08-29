import numpy as np
import cv2
import h5py


def scale_to_256(array):
    min_val = np.min(array)
    max_val = np.max(array)

    # Scale the array to the range [0, 255]
    scaled_array = ((array - min_val) / (max_val - min_val)) * 255

    # Round to integers
    scaled_array = scaled_array.astype(np.uint8)

    return scaled_array

def extract_patches(input_array, patch_shape, stride):
    patches = []
    rows, cols = input_array.shape
    patch_rows, patch_cols = patch_shape
    row_stride, col_stride = stride
    
    for r in range(0, rows - patch_rows + 1, row_stride):
        for c in range(0, cols - patch_cols + 1, col_stride):
            patch = input_array[r:r + patch_rows, c:c + patch_cols]
            patches.append(patch)
    
    return patches





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
    seis_data=scale_to_256(seis_data)
    labels = read_labels['labels']
    labels[labels==6] = 0

    #fazendo uma pequena mudança porque eu acho q tava desperdiçando uns dados a toa (tava sem interseção entre crossline e inline no teste e validacao, vou colocar)
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

    return trainslices, trainlabels, testslices, testlabels, valslices, vallabels


def limited_training_data(shape=(992,192),stridetest=(128,64), strideval=(128,64), stridetrain=(128,64), sizetrain_x=192, sizetrain_y=192, test_pos='end'):
    read_seis_data = np.load(
        '/scratch/nunes/seismic/data_train.npz', 
                allow_pickle=True, mmap_mode = 'r')
    # We read our labels
    read_labels = np.load(
        '/scratch/nunes/seismic/labels_train.npz',
                    allow_pickle=True, mmap_mode = 'r')

    # Inside the elements we pick what we are interesed in
    seis_data = read_seis_data['data']
    seis_data=scale_to_256(seis_data)
    labels = read_labels['labels']
    labels[labels==6] = 0

    inicio_area_livre_x=sizetrain_x+80
    fim_area_livre_x=seis_data.shape[1]
    meio_area_livre_x=int((fim_area_livre_x-inicio_area_livre_x)/2 + inicio_area_livre_x)
    #ex:192+80=272
    inicio_area_livre_y=sizetrain_y+80
    fim_area_livre_y=seis_data.shape[2]
    meio_area_livre_y=int((fim_area_livre_y-inicio_area_livre_y)/2 + inicio_area_livre_y)



    if(test_pos=='end'):
        test_start_x=fim_area_livre_x-80
        test_end_x=fim_area_livre_x
        test_start_y=fim_area_livre_y-80
        test_end_y=fim_area_livre_y
    if(test_pos=='start'):
        test_start_x=inicio_area_livre_x
        test_end_x=inicio_area_livre_x+80
        test_start_y=inicio_area_livre_y
        test_end_y=inicio_area_livre_y+80
    if(test_pos=='mid'):
        test_start_x=meio_area_livre_x-40
        test_end_x=meio_area_livre_x+40
        test_start_y=meio_area_livre_y-40
        test_end_y=meio_area_livre_y+40
        print(test_start_x,test_end_x)
        print(test_start_y,test_end_y)

    testcrossline=seis_data[:,test_start_x:test_end_x,:]
    testinline=seis_data[:,:,test_start_y:test_end_y]
    testcrossline_label=labels[:,test_start_x:test_end_x,:]
    testinline_label=labels[:,:,test_start_y:test_end_y]


    #removing the test data the rest of our data has shape Z=1006 X=702 Y=510
    valcrossline=seis_data[:,sizetrain_x:sizetrain_x+80,:sizetrain_y+80]
    valinline=seis_data[:,:sizetrain_x+80,sizetrain_y:sizetrain_y+80]
    valcrossline_label=labels[:,sizetrain_x:sizetrain_x+80,:sizetrain_y+80]
    valinline_label=labels[:,:sizetrain_x+80,sizetrain_y:sizetrain_y+80]


    ##removing the validation data the rest of our data has shape Z=1006 X=622 Y=430
    traindata=seis_data[:,:sizetrain_x,:sizetrain_y]
    trainlabel=labels[:,:sizetrain_x,:sizetrain_y]


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

    return trainslices, trainlabels, testslices, testlabels, valslices, vallabels



def article_division_data(shape=(992,192),stridetest=(128,64), strideval=(128,64), stridetrain=(128,64)):
    read_seis_data = np.load('/scratch/nunes/seismic/data_train.npz', 
            allow_pickle=True, mmap_mode = 'r')
    # We read our labels
    read_labels = np.load('/scratch/nunes/seismic/labels_train.npz',
                    allow_pickle=True, mmap_mode = 'r')

    # Inside the elements we pick what we are interesed in
    seis_data = read_seis_data['data']
    labels = read_labels['labels']
    labels[labels==6] = 0


    valinline=np.append(seis_data[:,:,:7],seis_data[:,:,584:], axis=2)
    valinline_label=np.append(labels[:,:,:7],labels[:,:,584:], axis=2)
    valcrossline=np.append(seis_data[:,:45,:],seis_data[:,738:,:], axis=1)
    valcrossline_label=np.append(labels[:,:45,:],labels[:,738:,:], axis=1)
    traindata=seis_data[:,45:738,7:583]
    trainlabel=labels[:,45:738,7:583]

    trainpatches=[]
    trainlabels=[]
    for i in (range(traindata.shape[2])):
        trainpatches=trainpatches+extract_patches(traindata[:,:,i],(shape),(stridetrain))
        trainlabels=trainlabels+extract_patches(trainlabel[:,:,i],(shape),(stridetrain))
    for i in (range(traindata.shape[1])):
        trainpatches=trainpatches+extract_patches(traindata[:,i,:],(shape),(stridetrain))
        trainlabels=trainlabels+extract_patches(trainlabel[:,i,:],(shape),(stridetrain))

    valpatches=[]
    vallabels=[]
    for i in (range(valinline.shape[2])):
        valpatches=valpatches+extract_patches(valinline[:,:,i],(shape),(strideval))
        vallabels=vallabels+extract_patches(valinline_label[:,:,i],(shape),(strideval))
    for i in (range(valcrossline.shape[1])):
        valpatches=valpatches+extract_patches(valcrossline[:,i,:],(shape),(strideval))
        vallabels=vallabels+extract_patches(valcrossline_label[:,i,:],(shape),(strideval))
    
    trainslices=np.array(trainpatches)
    trainlabels=np.array(trainlabels)
    valslices=np.array(valpatches)
    vallabels=np.array(vallabels)

    testslices=valslices
    testlabels=vallabels
    return trainslices, trainlabels, testslices, testlabels, valslices, vallabels

def read_h5_file(file_path):
    f = h5py.File(file_path,'r')
    images=f['features']
    labels=f['label']
    return np.squeeze(np.array(images)), np.array(labels)

def penobscot_data(shape=(1024,192),stridetest=(256,64), strideval=(256,64), stridetrain=(256,64)):
    images, masks=read_h5_file('/scratch/nunes/seismic/penobscot.h5')
    images=scale_to_256(images)
    images=images.astype(np.uint8)



    #(601, 1501, 481)
    images = images.transpose((1, 0, 2))
    masks = masks.transpose((1, 0, 2))
    #(1501,601,401)

    testcrossline=images[:,541:601,:]
    testcrossline_label=masks[:,541:601,:]


    testinline=images[:,:601,341:401]
    testinline_label=masks[:,:601,341:401]


    #removing the test data the rest of our data has shape Z=1501 X=541 Y=341
    valcrossline=images[:,481:541,:341]
    valcrossline_label=masks[:,481:541,:341]

    valinline=images[:,:541,281:341]
    valinline_label=masks[:,:541,281:341]


    ##removing the validation data the rest of our data has shape Z=1006 X=481 Y=281
    traindata=images[:,:481,:281]
    trainlabel=masks[:,:481,:281]

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

    return trainslices, trainlabels, testslices, testlabels, valslices, vallabels