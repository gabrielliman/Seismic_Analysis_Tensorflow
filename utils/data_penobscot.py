import os
import numpy as np
import h5py
from PIL import Image

# def read_files(folder_path, mask_path):

#     tiff_files = [file for file in os.listdir(folder_path) if file.endswith('.tiff')]

#     images = []
#     masks = []
#     for file in tiff_files:
#         image = Image.open(os.path.join(folder_path, file))
#         images.append(np.array(image))
#         mask = Image.open(os.path.join(mask_path, os.path.splitext(file)[0] + '_mask.png'))
#         masks.append(np.array(mask))

#     return np.array(images), np.array(masks)

def read_h5_file(file_path="/home/grad/ccomp/21/nuneslima/Seismic-Analysis/penobscot/dataset.h5"):
    f = h5py.File(file_path,'r')
    images=f['features']
    labels=f['label']
    return np.squeeze(np.array(images)), np.array(labels)



def divide_into_patches(images, patch_h, patch_w, stride_h, stride_w):
    num_images, height, width= images.shape
    
    patches_per_dim_h = (height - patch_h) // stride_h + 1
    patches_per_dim_w = (width - patch_w) // stride_w + 1
    
    num_patches_per_image = patches_per_dim_h * patches_per_dim_w
    
    patches = np.zeros((num_images * num_patches_per_image, patch_h, patch_w), dtype=images.dtype)
    
    idx = 0
    for image in images:
        for h in range(0, height - patch_h + 1, stride_h):
            for w in range(0, width - patch_w + 1, stride_w):
                patch = image[h:h+patch_h, w:w+patch_w]
                patches[idx] = patch
                idx += 1
                
    return patches


def data_split(data, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
    num_samples = data.shape[0]
    
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    
    train_indices = np.arange(0, train_size)
    val_indices = np.arange(train_size, train_size+val_size)
    test_indices = np.arange(train_size+val_size, num_samples)
    
    train_set = data[train_indices]
    test_set = data[test_indices]
    val_set = data[val_indices]
    
    return train_set, test_set, val_set


def majority_class(images, masks, threshold_percentage=0.7):
    n, a, b = masks.shape
    majority_classes = []
    majority_images=[]

    for i in range(n):
        sample = masks[i]
        flattened_sample = sample.flatten()
        unique_classes, counts = np.unique(flattened_sample, return_counts=True)
        max_count = np.max(counts)
        total_count = np.sum(counts)
        if max_count / total_count >= threshold_percentage:
            majority_class_index = np.argmax(counts)
            majority_class = unique_classes[majority_class_index]
            majority_classes.append(majority_class)
            majority_images.append(images[i])


    return np.array(majority_images), np.array(majority_classes)




def penobscot_data_seg(patch_h, patch_w,stride_h, stride_w,train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
    #inlines, masks_in=read_files("/home/grad/ccomp/21/nuneslima/Datasets/Penobscot/inlines",'/home/grad/ccomp/21/nuneslima/Datasets/Penobscot/masks')
    images, masks=read_h5_file()
    images = ((images + 32767) / 65534) * 255
    images=images.astype(np.uint8)
    patch=divide_into_patches(images,patch_h, patch_w,stride_h, stride_w)
    patch_mask=divide_into_patches(masks,patch_h, patch_w,stride_h, stride_w)

    train,test,val=data_split(patch, train_ratio=train_ratio, test_ratio=test_ratio, val_ratio=val_ratio)
    mask_train,mask_test,mask_val=data_split(patch_mask, train_ratio=train_ratio, test_ratio=test_ratio, val_ratio=val_ratio)


    return train, mask_train, test, mask_test, val, mask_val


def penobscot_data(patch,stride):
    inlines, masks_in=read_h5_file("/home/grad/ccomp/21/nuneslima/Datasets/Penobscot/inlines",'/home/grad/ccomp/21/nuneslima/Datasets/Penobscot/masks')
    inlines = ((inlines + 32767) / 65534) * 255
    inlines=inlines.astype(np.uint8)
    #crosslines, masks_cross=read_files("/home/grad/ccomp/21/nuneslima/Datasets/Penobscot/crosslines",'/home/grad/ccomp/21/nuneslima/Datasets/Penobscot/masks')
    patch_inline=divide_into_patches(inlines,patch,stride)
    #patch_crossline=divide_into_patches(crosslines,patch,stride)
    patch_mask_in=divide_into_patches(masks_in,patch,stride)
    #patch_mask_cross=divide_into_patches(masks_cross,patch,stride)

    in_train,in_test,in_val=data_split(patch_inline)
    mask_in_train,mask_in_test,mask_in_val=data_split(patch_mask_in)
    # cross_train,cross_test,cross_val=data_split(patch_crossline)
    # mask_cross_train,mask_cross_test,mask_cross_val=data_split(patch_mask_cross)

    # trainX=np.append(in_train,cross_train, axis=0)
    # trainY=np.append(mask_in_train,mask_cross_train, axis=0)
    # testX=np.append(in_test,cross_test, axis=0)
    # testY=np.append(mask_in_test,mask_cross_test, axis=0)
    # valX=np.append(in_val,cross_val, axis=0)
    # valY=np.append(mask_in_val,mask_cross_val, axis=0)




    patch_train,train_labels = majority_class(in_train, mask_in_train, 0.7)
    patch_test,test_labels = majority_class(in_test, mask_in_test, 0.7)
    patch_val,val_labels = majority_class(in_val, mask_in_val, 0.7)


    return patch_train, train_labels, patch_test, test_labels, patch_val, val_labels

    #return trainX, majority_class(trainY), testX, majority_class(testY), valX, majority_class(valY)
