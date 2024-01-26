import os
import numpy as np
from PIL import Image


def load_images_from_tile(folder_path):
    image_data = []
    folder_path=folder_path+"/dataset/semantic_drone_dataset/original_images"
    i=0
    for filename in sorted(os.listdir(folder_path)):
        if(i>50):
            break
        image_path = os.path.join(folder_path, filename)
        img = Image.open(image_path)
        img_gray = img.convert('L')
        img_array = np.array(img_gray)
        image_data.append(img_array)
        i=i+1
    return image_data



def load_masks_from_tile(folder_path):
    image_data = []
    folder_path=folder_path+"/RGB_color_image_masks/RGB_color_image_masks"
    i=0
    for filename in sorted(os.listdir(folder_path)):
        if(i>50):
            break
        image_path = os.path.join(folder_path, filename)
        img = Image.open(image_path)
        img_gray = img.convert('L')
        img_array = np.array(img_gray)

        img_array[img_array==39]=1
        img_array[img_array==42]=2
        img_array[img_array==45]=3
        img_array[img_array==46]=4
        img_array[img_array==52]=5
        img_array[img_array==60]=6
        img_array[img_array==70]=7
        img_array[img_array==83]=8
        img_array[img_array==90]=9
        img_array[img_array==93]=10
        img_array[img_array==100]=11
        img_array[img_array==104]=12
        img_array[img_array==108]=13
        img_array[img_array==119]=14
        img_array[img_array==138]=15
        img_array[img_array==153]=16
        img_array[img_array==164]=17
        img_array[img_array==211]=18
        img_array[img_array==225]=19

        image_data.append(img_array)
        i=i+1
    return image_data


def extract_patches(images, patch_size, stride):
    """
    Extract patches from a list of grayscale images.

    Parameters:
    - images: List of 2D numpy arrays representing grayscale images.
    - patch_size: Tuple (height, width) specifying the size of the patches.
    - stride: Tuple (vertical_stride, horizontal_stride) specifying the stride between patches.

    Returns:
    - List of patches as 2D numpy arrays.
    """

    patch_height, patch_width = patch_size
    vertical_stride, horizontal_stride = stride

    all_patches = []

    for image in images:
        img_height, img_width = image.shape

        for y in range(0, img_height - patch_height + 1, vertical_stride):
            for x in range(0, img_width - patch_width + 1, horizontal_stride):
                patch = image[y:y+patch_height, x:x+patch_width]
                all_patches.append(patch)

    return np.stack(all_patches, axis=0)
    
def drone_patches(folder_path, patch_shape, stride,train=70, val=15, test=15):
    images=load_images_from_tile(folder_path)
    masks=load_masks_from_tile(folder_path)
    patches=extract_patches(images, patch_shape, stride)
    mask_patches=extract_patches(masks, patch_shape,stride)
    tam=int(patches.shape[0]/7)
    train_lim=int(tam*train/100)
    val_lim=int(tam*(train+val)/100)
    #train, test, val
    return patches[:train_lim,:,:], mask_patches[:train_lim,:,:],patches[val_lim:tam,:,:], mask_patches[val_lim:tam,:,:],patches[train_lim:val_lim,:,:], mask_patches[train_lim:val_lim,:,:],