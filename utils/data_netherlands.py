import numpy as np

def read_files(folder_path='/home/grad/ccomp/21/nuneslima/Datasets/Netherlands/data'):
    train_label=np.load(folder_path+"/train/train_labels.npy")
    train_seismic=np.load(folder_path+"/train/train_seismic.npy") 
    train_seismic =((train_seismic+1)/2)*255
    train_seismic=train_seismic.astype(np.uint8)

    test1_label=np.load(folder_path+"/test_once/test1_labels.npy")
    test1_seismic=np.load(folder_path+"/test_once/test1_seismic.npy")    
    test1_seismic =((test1_seismic+1)/2)*255
    test1_seismic=test1_seismic.astype(np.uint8)

    test2_label=np.load(folder_path+"/test_once/test2_labels.npy")
    test2_label=np.transpose(test2_label,(1,0,2))
    test2_seismic=np.load(folder_path+"/test_once/test2_seismic.npy")
    test2_seismic=np.transpose(test2_seismic,(1,0,2))
    test2_seismic =((test2_seismic+1)/2)*255
    test2_seismic=test2_seismic.astype(np.uint8)


    return train_label, train_seismic, test1_label, test1_seismic, test2_label, test2_seismic


def divide_into_patches(images, patch_size, stride):
    num_images, height, width = images.shape
    
    patches_per_dim_h = (height - patch_size) // stride + 1
    patches_per_dim_w = (width - patch_size) // stride + 1
    
    num_patches_per_image = patches_per_dim_h * patches_per_dim_w
    
    patches = np.zeros((num_images * num_patches_per_image, patch_size, patch_size), dtype=images.dtype)
    
    idx = 0
    for image in images:
        for h in range(0, height - patch_size + 1, stride):
            for w in range(0, width - patch_size + 1, stride):
                patch = image[h:h+patch_size, w:w+patch_size]
                patches[idx] = patch
                idx += 1
                
    return patches



# def majority_class(array):
#     n, a, b = array.shape
#     majority_classes = []

#     for i in range(n):
#         sample = array[i]
#         flattened_sample = sample.flatten()
#         unique_classes, counts = np.unique(flattened_sample, return_counts=True)
#         majority_class_index = np.argmax(counts)
#         majority_class = unique_classes[majority_class_index]
#         majority_classes.append(majority_class)

#     return np.array(majority_classes)


def majority_class(images, masks, threshold_percentage=0.5):
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



def netherlands_data(patch,stride):
    train_label, train_seismic, test1_label, test1_seismic, test2_label, test2_seismic = read_files()
    patch_mask_train = divide_into_patches(train_label,patch,stride)
    patch_train = divide_into_patches(train_seismic,patch,stride)
    patch_mask_test = divide_into_patches(test1_label,patch,stride)
    patch_test = divide_into_patches(test1_seismic,patch,stride)
    patch_mask_val = divide_into_patches(test2_label,patch,stride)
    patch_val = divide_into_patches(test2_seismic,patch,stride)

    patch_train,train_labels = majority_class(patch_train, patch_mask_train, 0.7)
    patch_test,test_labels = majority_class(patch_test, patch_mask_test, 0.7)
    patch_val,val_labels = majority_class(patch_val, patch_mask_val, 0.7)


    return patch_train, train_labels, patch_test, test_labels, patch_val, val_labels



def netherlands_data_seg(patch,stride):
    train_label, train_seismic, test1_label, test1_seismic, test2_label, test2_seismic = read_files()
    patch_mask_train = divide_into_patches(train_label,patch,stride)
    patch_train = divide_into_patches(train_seismic,patch,stride)
    patch_mask_test = divide_into_patches(test1_label,patch,stride)
    patch_test = divide_into_patches(test1_seismic,patch,stride)
    patch_mask_val = divide_into_patches(test2_label,patch,stride)
    patch_val = divide_into_patches(test2_seismic,patch,stride)

    # patch_train,train_labels = majority_class(patch_train, patch_mask_train, 0.7)
    # patch_test,test_labels = majority_class(patch_test, patch_mask_test, 0.7)
    # patch_val,val_labels = majority_class(patch_val, patch_mask_val, 0.7)


    return patch_train, patch_mask_train, patch_test, patch_mask_test, patch_val, patch_mask_val