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

def get_limits(start, end, num_limits):
    step_size = (end - start) / (num_limits + 1)

    limits = []
    for i in range(1, num_limits + 1):
        limit = start + i * step_size
        limits.append(int(limit))

    return limits



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


def limited_training_data_fixed_test_val(shape=(992,192),stridetest=(128,64), strideval=(128,64), stridetrain=(128,64), sizetrain_x=192, sizetrain_y=192, test_pos='end',max_sizetrain_x=320,max_sizetrain_y=270):
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

    inicio_area_livre_x=max_sizetrain_x+80
    fim_area_livre_x=seis_data.shape[1]
    meio_area_livre_x=int((fim_area_livre_x-inicio_area_livre_x)/2 + inicio_area_livre_x)

    inicio_area_livre_y=max_sizetrain_y+80
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


    testcrossline=seis_data[:,test_start_x:test_end_x,:]
    testinline=seis_data[:,:,test_start_y:test_end_y]
    testcrossline_label=labels[:,test_start_x:test_end_x,:]
    testinline_label=labels[:,:,test_start_y:test_end_y]


    #removing the test data the rest of our data has shape Z=1006 X=702 Y=510
    valcrossline=seis_data[:,max_sizetrain_x:max_sizetrain_x+80,:max_sizetrain_y+80]
    valinline=seis_data[:,:max_sizetrain_x+80,max_sizetrain_y:max_sizetrain_y+80]
    valcrossline_label=labels[:,max_sizetrain_x:max_sizetrain_x+80,:max_sizetrain_y+80]
    valinline_label=labels[:,:max_sizetrain_x+80,max_sizetrain_y:max_sizetrain_y+80]


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


def slices_for_training(shape=(992,192),stridetest=(128,64), strideval=(128,64), stridetrain=(128,64),num_train=1):
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

    inicio_area_livre_x=0
    fim_area_livre_x=seis_data.shape[1]
    #ex:192+80=272
    inicio_area_livre_y=0
    fim_area_livre_y=seis_data.shape[2]



    #We start with our data with shape Z=1006 X=782 Y=590
    testcrossline=seis_data[:,inicio_area_livre_x:fim_area_livre_x,:]
    testinline=seis_data[:,:,inicio_area_livre_y:fim_area_livre_y]
    testcrossline_label=labels[:,inicio_area_livre_x:fim_area_livre_x,:]
    testinline_label=labels[:,:,inicio_area_livre_y:fim_area_livre_y]




    x_limits=get_limits(inicio_area_livre_x,fim_area_livre_x,num_train)
    y_limits=get_limits(inicio_area_livre_y,fim_area_livre_y,num_train)



    #TRAINING
    trainpatches=[]
    trainlabels=[]
    valpatches=[]
    vallabels=[]


    for pos_x, pos_y in zip(x_limits, y_limits):
        # Extract additional areas from test data
        extra_traincrossline = seis_data[:, pos_x:pos_x+1, :pos_y]
        extra_traininline = seis_data[:, :pos_x, pos_y:pos_y+1]
        extra_traincrossline_label = labels[:, pos_x:pos_x+1, :pos_y]
        extra_traininline_label = labels[:, :pos_x, pos_y:pos_y+1]

        for i in (range(extra_traininline.shape[2])):
            trainpatches=trainpatches+extract_patches(extra_traininline[:,:,i],(shape),(stridetrain))
            trainlabels=trainlabels+extract_patches(extra_traininline_label[:,:,i],(shape),(stridetrain))
        for i in (range(extra_traincrossline.shape[1])):
            trainpatches=trainpatches+extract_patches(extra_traincrossline[:,i,:],(shape),(stridetrain))
            trainlabels=trainlabels+extract_patches(extra_traincrossline_label[:,i,:],(shape),(stridetrain))

        extra_valcrossline = seis_data[:, pos_x+1:pos_x+2, :pos_y]
        extra_valinline = seis_data[:, :pos_x, pos_y+1:pos_y+2]
        extra_valcrossline_label = labels[:, pos_x+1:pos_x+2, :pos_y]
        extra_valinline_label = labels[:, :pos_x, pos_y+1:pos_y+2]

        for i in (range(extra_valinline.shape[2])):
            valpatches=valpatches+extract_patches(extra_valinline[:,:,i],(shape),(strideval))
            vallabels=vallabels+extract_patches(extra_valinline_label[:,:,i],(shape),(strideval))
        for i in (range(extra_valcrossline.shape[1])):
            valpatches=valpatches+extract_patches(extra_valcrossline[:,i,:],(shape),(strideval))
            vallabels=vallabels+extract_patches(extra_valcrossline_label[:,i,:],(shape),(strideval))
    
    testpatches=[]
    testlabels=[]
    for i in range(num_train+1):
        if(i==0):
            inicio_x=inicio_area_livre_x
            inicio_y=inicio_area_livre_y
            fim_x=x_limits[i]
            fim_y=y_limits[i]
        elif(i==num_train):
            inicio_x=x_limits[i-1]+2
            inicio_y=y_limits[i-1]+2
            fim_x=fim_area_livre_x
            fim_y=fim_area_livre_y
        else:
            inicio_x=x_limits[i-1]+2
            inicio_y=y_limits[i-1]+2
            fim_x=x_limits[i]
            fim_y=y_limits[i]

        testcrossline=seis_data[:,inicio_x:fim_x,:]
        testinline=seis_data[:,:,inicio_y:fim_y]
        testcrossline_label=labels[:,inicio_x:fim_x,:]
        testinline_label=labels[:,:,inicio_y:fim_y]
        for i in (range(testinline.shape[2])):
            testpatches=testpatches+extract_patches(testinline[:,:,i],(shape),(stridetest))
            testlabels=testlabels+extract_patches(testinline_label[:,:,i],(shape),(stridetest))
        for i in (range(testcrossline.shape[1])):
            testpatches=testpatches+extract_patches(testcrossline[:,i,:],(shape),(stridetest))
            testlabels=testlabels+extract_patches(testcrossline_label[:,i,:],(shape),(stridetest))


    trainslices=np.array(trainpatches)
    trainlabels=np.array(trainlabels)
    
    valslices=np.array(valpatches)
    vallabels=np.array(vallabels)

    testslices=np.array(testpatches)
    testlabels=np.array(testlabels)

    return trainslices, trainlabels, testslices, testlabels, valslices, vallabels


def smart_training_data(shape=(992,192),stridetest=(128,64), strideval=(128,64), stridetrain=(128,64), sizetrain_x=192, sizetrain_y=192,num_extra_train=1):
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
    #ex:192+80=272
    inicio_area_livre_y=sizetrain_y+80
    fim_area_livre_y=seis_data.shape[2]



    #We start with our data with shape Z=1006 X=782 Y=590
    testcrossline=seis_data[:,inicio_area_livre_x:fim_area_livre_x,:]
    testinline=seis_data[:,:,inicio_area_livre_y:fim_area_livre_y]
    testcrossline_label=labels[:,inicio_area_livre_x:fim_area_livre_x,:]
    testinline_label=labels[:,:,inicio_area_livre_y:fim_area_livre_y]


    #removing the test data the rest of our data has shape Z=1006 X=702 Y=510
    valcrossline=seis_data[:,sizetrain_x:sizetrain_x+80,:sizetrain_y+80]
    valinline=seis_data[:,:sizetrain_x+80,sizetrain_y:sizetrain_y+80]
    valcrossline_label=labels[:,sizetrain_x:sizetrain_x+80,:sizetrain_y+80]
    valinline_label=labels[:,:sizetrain_x+80,sizetrain_y:sizetrain_y+80]


    ##removing the validation data the rest of our data has shape Z=1006 X=622 Y=430
    traindata=seis_data[:,:sizetrain_x,:sizetrain_y]
    trainlabel=labels[:,:sizetrain_x,:sizetrain_y]


    x_limits=get_limits(inicio_area_livre_x,fim_area_livre_x,num_extra_train)
    y_limits=get_limits(inicio_area_livre_y,fim_area_livre_y,num_extra_train)



    #TRAINING
    trainpatches=[]
    trainlabels=[]
    for i in (range(traindata.shape[2])):
        trainpatches=trainpatches+extract_patches(traindata[:,:,i],(shape),(stridetrain))
        trainlabels=trainlabels+extract_patches(trainlabel[:,:,i],(shape),(stridetrain))
    for i in (range(traindata.shape[1])):
        trainpatches=trainpatches+extract_patches(traindata[:,i,:],(shape),(stridetrain))
        trainlabels=trainlabels+extract_patches(trainlabel[:,i,:],(shape),(stridetrain))

    for pos_x, pos_y in zip(x_limits, y_limits):
        # Extract additional areas from test data
        extra_traincrossline = seis_data[:, pos_x:pos_x+1, :pos_y]
        extra_traininline = seis_data[:, :pos_x, pos_y:pos_y+1]
        extra_traincrossline_label = labels[:, pos_x:pos_x+1, :pos_y]
        extra_traininline_label = labels[:, :pos_x, pos_y:pos_y+1]

        for i in (range(extra_traininline.shape[2])):
            trainpatches=trainpatches+extract_patches(extra_traininline[:,:,i],(shape),(stridetrain))
            trainlabels=trainlabels+extract_patches(extra_traininline_label[:,:,i],(shape),(stridetrain))
        for i in (range(extra_traincrossline.shape[1])):
            trainpatches=trainpatches+extract_patches(extra_traincrossline[:,i,:],(shape),(stridetrain))
            trainlabels=trainlabels+extract_patches(extra_traincrossline_label[:,i,:],(shape),(stridetrain))
    
    testpatches=[]
    testlabels=[]
    for i in range(num_extra_train+1):
        if(i==0):
            inicio_x=inicio_area_livre_x
            inicio_y=inicio_area_livre_y
            fim_x=x_limits[i]
            fim_y=y_limits[i]
        elif(i==num_extra_train):
            inicio_x=x_limits[i-1]+1
            inicio_y=y_limits[i-1]+1
            fim_x=fim_area_livre_x
            fim_y=fim_area_livre_y
        else:
            inicio_x=x_limits[i-1]+1
            inicio_y=y_limits[i-1]+1
            fim_x=x_limits[i]
            fim_y=y_limits[i]

        testcrossline=seis_data[:,inicio_x:fim_x,:]
        testinline=seis_data[:,:,inicio_y:fim_y]
        testcrossline_label=labels[:,inicio_x:fim_x,:]
        testinline_label=labels[:,:,inicio_y:fim_y]
        for i in (range(testinline.shape[2])):
            testpatches=testpatches+extract_patches(testinline[:,:,i],(shape),(stridetest))
            testlabels=testlabels+extract_patches(testinline_label[:,:,i],(shape),(stridetest))
        for i in (range(testcrossline.shape[1])):
            testpatches=testpatches+extract_patches(testcrossline[:,i,:],(shape),(stridetest))
            testlabels=testlabels+extract_patches(testcrossline_label[:,i,:],(shape),(stridetest))


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


    #(601, 1501, 401)
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


def penobscot_limited_data(shape=(1024,192),stridetest=(256,64), strideval=(256,64), stridetrain=(256,64), sizetrain_x=192,sizetrain_y=192, test_pos='end'):
    seis_data, labels=read_h5_file('/scratch/nunes/seismic/penobscot.h5')
    seis_data=scale_to_256(seis_data)
    seis_data=seis_data.astype(np.uint8)


    #(601, 1501, 481)
    seis_data = seis_data.transpose((1, 0, 2))
    labels = labels.transpose((1, 0, 2))
    #(1501,601,401)

    
    inicio_area_livre_x=sizetrain_x+60
    fim_area_livre_x=seis_data.shape[1]
    meio_area_livre_x=int((fim_area_livre_x-inicio_area_livre_x)/2 + inicio_area_livre_x)
    #ex:192+60=252
    inicio_area_livre_y=sizetrain_y+60
    fim_area_livre_y=seis_data.shape[2]
    meio_area_livre_y=int((fim_area_livre_y-inicio_area_livre_y)/2 + inicio_area_livre_y)

    if(test_pos=='end'):
        test_start_x=fim_area_livre_x-60
        test_end_x=fim_area_livre_x
        test_start_y=fim_area_livre_y-60
        test_end_y=fim_area_livre_y
    if(test_pos=='start'):
        test_start_x=inicio_area_livre_x
        test_end_x=inicio_area_livre_x+60
        test_start_y=inicio_area_livre_y
        test_end_y=inicio_area_livre_y+60
    if(test_pos=='mid'):
        test_start_x=meio_area_livre_x-30
        test_end_x=meio_area_livre_x+30
        test_start_y=meio_area_livre_y-30
        test_end_y=meio_area_livre_y+30

    testcrossline=seis_data[:,test_start_x:test_end_x,:]
    testinline=seis_data[:,:,test_start_y:test_end_y]
    testcrossline_label=labels[:,test_start_x:test_end_x,:]
    testinline_label=labels[:,:,test_start_y:test_end_y]

    valcrossline=seis_data[:,sizetrain_x:sizetrain_x+60,:sizetrain_y+60]
    valinline=seis_data[:,:sizetrain_x+60,sizetrain_y:sizetrain_y+60]
    valcrossline_label=labels[:,sizetrain_x:sizetrain_x+60,:sizetrain_y+60]
    valinline_label=labels[:,:sizetrain_x+60,sizetrain_y:sizetrain_y+60]

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

def penobscot_limited_training_data_fixed_test_val(shape=(1024,192),stridetest=(256,64), strideval=(256,64), stridetrain=(256,64), sizetrain_x=192, sizetrain_y=192, test_pos='end',max_sizetrain_x=320,max_sizetrain_y=270):
    seis_data, labels=read_h5_file('/scratch/nunes/seismic/penobscot.h5')
    seis_data=scale_to_256(seis_data)
    seis_data=seis_data.astype(np.uint8)


    #(601, 1501, 481)
    seis_data = seis_data.transpose((1, 0, 2))
    labels = labels.transpose((1, 0, 2))
    #(1501,601,401)

    inicio_area_livre_x=max_sizetrain_x+60
    fim_area_livre_x=seis_data.shape[1]
    meio_area_livre_x=int((fim_area_livre_x-inicio_area_livre_x)/2 + inicio_area_livre_x)

    inicio_area_livre_y=max_sizetrain_y+60
    fim_area_livre_y=seis_data.shape[2]
    meio_area_livre_y=int((fim_area_livre_y-inicio_area_livre_y)/2 + inicio_area_livre_y)


    if(test_pos=='end'):
        test_start_x=fim_area_livre_x-60
        test_end_x=fim_area_livre_x
        test_start_y=fim_area_livre_y-60
        test_end_y=fim_area_livre_y
    if(test_pos=='start'):
        test_start_x=inicio_area_livre_x
        test_end_x=inicio_area_livre_x+60
        test_start_y=inicio_area_livre_y
        test_end_y=inicio_area_livre_y+60
    if(test_pos=='mid'):
        test_start_x=meio_area_livre_x-30
        test_end_x=meio_area_livre_x+30
        test_start_y=meio_area_livre_y-30
        test_end_y=meio_area_livre_y+30

    testcrossline=seis_data[:,test_start_x:test_end_x,:]
    testinline=seis_data[:,:,test_start_y:test_end_y]
    testcrossline_label=labels[:,test_start_x:test_end_x,:]
    testinline_label=labels[:,:,test_start_y:test_end_y]

    valcrossline=seis_data[:,max_sizetrain_x:max_sizetrain_x+60,:max_sizetrain_y+60]
    valinline=seis_data[:,:max_sizetrain_x+60,max_sizetrain_y:max_sizetrain_y+60]
    valcrossline_label=labels[:,max_sizetrain_x:max_sizetrain_x+60,:max_sizetrain_y+60]
    valinline_label=labels[:,:max_sizetrain_x+60,max_sizetrain_y:max_sizetrain_y+60]

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

def penobscot_slices_for_training(shape=(1024,192),stridetest=(256,64), strideval=(256,64), stridetrain=(256,64),num_train=1):
    seis_data, labels=read_h5_file('/scratch/nunes/seismic/penobscot.h5')
    seis_data=scale_to_256(seis_data)
    seis_data=seis_data.astype(np.uint8)


    #(601, 1501, 481)
    seis_data = seis_data.transpose((1, 0, 2))
    labels = labels.transpose((1, 0, 2))
    #(1501,601,401)

    inicio_area_livre_x=0
    fim_area_livre_x=seis_data.shape[1]
    #ex:192+80=272
    inicio_area_livre_y=0
    fim_area_livre_y=seis_data.shape[2]



    #We start with our data with shape Z=1006 X=782 Y=590
    testcrossline=seis_data[:,inicio_area_livre_x:fim_area_livre_x,:]
    testinline=seis_data[:,:,inicio_area_livre_y:fim_area_livre_y]
    testcrossline_label=labels[:,inicio_area_livre_x:fim_area_livre_x,:]
    testinline_label=labels[:,:,inicio_area_livre_y:fim_area_livre_y]




    x_limits=get_limits(inicio_area_livre_x,fim_area_livre_x,num_train)
    y_limits=get_limits(inicio_area_livre_y,fim_area_livre_y,num_train)



    #TRAINING
    trainpatches=[]
    trainlabels=[]
    valpatches=[]
    vallabels=[]


    for pos_x, pos_y in zip(x_limits, y_limits):
        # Extract additional areas from test data
        extra_traincrossline = seis_data[:, pos_x:pos_x+1, :pos_y]
        extra_traininline = seis_data[:, :pos_x, pos_y:pos_y+1]
        extra_traincrossline_label = labels[:, pos_x:pos_x+1, :pos_y]
        extra_traininline_label = labels[:, :pos_x, pos_y:pos_y+1]

        for i in (range(extra_traininline.shape[2])):
            trainpatches=trainpatches+extract_patches(extra_traininline[:,:,i],(shape),(stridetrain))
            trainlabels=trainlabels+extract_patches(extra_traininline_label[:,:,i],(shape),(stridetrain))
        for i in (range(extra_traincrossline.shape[1])):
            trainpatches=trainpatches+extract_patches(extra_traincrossline[:,i,:],(shape),(stridetrain))
            trainlabels=trainlabels+extract_patches(extra_traincrossline_label[:,i,:],(shape),(stridetrain))

        extra_valcrossline = seis_data[:, pos_x+1:pos_x+2, :pos_y]
        extra_valinline = seis_data[:, :pos_x, pos_y+1:pos_y+2]
        extra_valcrossline_label = labels[:, pos_x+1:pos_x+2, :pos_y]
        extra_valinline_label = labels[:, :pos_x, pos_y+1:pos_y+2]

        for i in (range(extra_valinline.shape[2])):
            valpatches=valpatches+extract_patches(extra_valinline[:,:,i],(shape),(strideval))
            vallabels=vallabels+extract_patches(extra_valinline_label[:,:,i],(shape),(strideval))
        for i in (range(extra_valcrossline.shape[1])):
            valpatches=valpatches+extract_patches(extra_valcrossline[:,i,:],(shape),(strideval))
            vallabels=vallabels+extract_patches(extra_valcrossline_label[:,i,:],(shape),(strideval))
    
    testpatches=[]
    testlabels=[]
    for i in range(num_train+1):
        if(i==0):
            inicio_x=inicio_area_livre_x
            inicio_y=inicio_area_livre_y
            fim_x=x_limits[i]
            fim_y=y_limits[i]
        elif(i==num_train):
            inicio_x=x_limits[i-1]+2
            inicio_y=y_limits[i-1]+2
            fim_x=fim_area_livre_x
            fim_y=fim_area_livre_y
        else:
            inicio_x=x_limits[i-1]+2
            inicio_y=y_limits[i-1]+2
            fim_x=x_limits[i]
            fim_y=y_limits[i]

        testcrossline=seis_data[:,inicio_x:fim_x,:]
        testinline=seis_data[:,:,inicio_y:fim_y]
        testcrossline_label=labels[:,inicio_x:fim_x,:]
        testinline_label=labels[:,:,inicio_y:fim_y]
        for i in (range(testinline.shape[2])):
            testpatches=testpatches+extract_patches(testinline[:,:,i],(shape),(stridetest))
            testlabels=testlabels+extract_patches(testinline_label[:,:,i],(shape),(stridetest))
        for i in (range(testcrossline.shape[1])):
            testpatches=testpatches+extract_patches(testcrossline[:,i,:],(shape),(stridetest))
            testlabels=testlabels+extract_patches(testcrossline_label[:,i,:],(shape),(stridetest))


    trainslices=np.array(trainpatches)
    trainlabels=np.array(trainlabels)
    
    valslices=np.array(valpatches)
    vallabels=np.array(vallabels)

    testslices=np.array(testpatches)
    testlabels=np.array(testlabels)

    return trainslices, trainlabels, testslices, testlabels, valslices, vallabels

def penobscot_smart_training_data(shape=(1024,192),stridetest=(256,64), strideval=(256,64), stridetrain=(256,64), sizetrain_x=192, sizetrain_y=192,num_extra_train=1):
    seis_data, labels=read_h5_file('/scratch/nunes/seismic/penobscot.h5')
    seis_data=scale_to_256(seis_data)
    seis_data=seis_data.astype(np.uint8)


    #(601, 1501, 481)
    seis_data = seis_data.transpose((1, 0, 2))
    labels = labels.transpose((1, 0, 2))
    #(1501,601,401)

    inicio_area_livre_x=sizetrain_x+60
    fim_area_livre_x=seis_data.shape[1]
    #ex:192+80=272
    inicio_area_livre_y=sizetrain_y+60
    fim_area_livre_y=seis_data.shape[2]



    #We start with our data with shape Z=1006 X=782 Y=590
    testcrossline=seis_data[:,inicio_area_livre_x:fim_area_livre_x,:]
    testinline=seis_data[:,:,inicio_area_livre_y:fim_area_livre_y]
    testcrossline_label=labels[:,inicio_area_livre_x:fim_area_livre_x,:]
    testinline_label=labels[:,:,inicio_area_livre_y:fim_area_livre_y]


    #removing the test data the rest of our data has shape Z=1006 X=702 Y=510
    valcrossline=seis_data[:,sizetrain_x:sizetrain_x+60,:sizetrain_y+60]
    valinline=seis_data[:,:sizetrain_x+60,sizetrain_y:sizetrain_y+60]
    valcrossline_label=labels[:,sizetrain_x:sizetrain_x+60,:sizetrain_y+60]
    valinline_label=labels[:,:sizetrain_x+60,sizetrain_y:sizetrain_y+60]


    ##removing the validation data the rest of our data has shape Z=1006 X=622 Y=430
    traindata=seis_data[:,:sizetrain_x,:sizetrain_y]
    trainlabel=labels[:,:sizetrain_x,:sizetrain_y]


    x_limits=get_limits(inicio_area_livre_x,fim_area_livre_x,num_extra_train)
    y_limits=get_limits(inicio_area_livre_y,fim_area_livre_y,num_extra_train)



    #TRAINING
    trainpatches=[]
    trainlabels=[]
    for i in (range(traindata.shape[2])):
        trainpatches=trainpatches+extract_patches(traindata[:,:,i],(shape),(stridetrain))
        trainlabels=trainlabels+extract_patches(trainlabel[:,:,i],(shape),(stridetrain))
    for i in (range(traindata.shape[1])):
        trainpatches=trainpatches+extract_patches(traindata[:,i,:],(shape),(stridetrain))
        trainlabels=trainlabels+extract_patches(trainlabel[:,i,:],(shape),(stridetrain))

    for pos_x, pos_y in zip(x_limits, y_limits):
        # Extract additional areas from test data
        extra_traincrossline = seis_data[:, pos_x:pos_x+1, :pos_y]
        extra_traininline = seis_data[:, :pos_x, pos_y:pos_y+1]
        extra_traincrossline_label = labels[:, pos_x:pos_x+1, :pos_y]
        extra_traininline_label = labels[:, :pos_x, pos_y:pos_y+1]

        for i in (range(extra_traininline.shape[2])):
            trainpatches=trainpatches+extract_patches(extra_traininline[:,:,i],(shape),(stridetrain))
            trainlabels=trainlabels+extract_patches(extra_traininline_label[:,:,i],(shape),(stridetrain))
        for i in (range(extra_traincrossline.shape[1])):
            trainpatches=trainpatches+extract_patches(extra_traincrossline[:,i,:],(shape),(stridetrain))
            trainlabels=trainlabels+extract_patches(extra_traincrossline_label[:,i,:],(shape),(stridetrain))
    
    testpatches=[]
    testlabels=[]
    for i in range(num_extra_train+1):
        if(i==0):
            inicio_x=inicio_area_livre_x
            inicio_y=inicio_area_livre_y
            fim_x=x_limits[i]
            fim_y=y_limits[i]
        elif(i==num_extra_train):
            inicio_x=x_limits[i-1]+1
            inicio_y=y_limits[i-1]+1
            fim_x=fim_area_livre_x
            fim_y=fim_area_livre_y
        else:
            inicio_x=x_limits[i-1]+1
            inicio_y=y_limits[i-1]+1
            fim_x=x_limits[i]
            fim_y=y_limits[i]

        testcrossline=seis_data[:,inicio_x:fim_x,:]
        testinline=seis_data[:,:,inicio_y:fim_y]
        testcrossline_label=labels[:,inicio_x:fim_x,:]
        testinline_label=labels[:,:,inicio_y:fim_y]
        for i in (range(testinline.shape[2])):
            testpatches=testpatches+extract_patches(testinline[:,:,i],(shape),(stridetest))
            testlabels=testlabels+extract_patches(testinline_label[:,:,i],(shape),(stridetest))
        for i in (range(testcrossline.shape[1])):
            testpatches=testpatches+extract_patches(testcrossline[:,i,:],(shape),(stridetest))
            testlabels=testlabels+extract_patches(testcrossline_label[:,i,:],(shape),(stridetest))


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

    testslices=np.array(testpatches)
    testlabels=np.array(testlabels)

    return trainslices, trainlabels, testslices, testlabels, valslices, vallabels
