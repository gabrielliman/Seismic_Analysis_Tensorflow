import os
import numpy as np
from PIL import Image


def load_images_from_tile(folder_path):
    image_data = []
    folder_path=folder_path+"/images"
    for filename in sorted(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, filename)
        img = Image.open(image_path)
        img_gray = img.convert('L')
        img_array = np.array(img_gray)
        image_data.append(img_array)

    return image_data


def load_masks_from_tile(folder_path):
    image_mask = []
    folder_path=folder_path+"/masks"
    for filename in sorted(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, filename)
        img = Image.open(image_path)
        img_gray = img.convert('L')
        img_array = np.array(img_gray)


        condition0=(img_array<=45)
        condition1=(img_array>45) & (img_array<=92)
        condition2=(img_array>92) & (img_array<=156)
        condition3=(img_array>156) & (img_array<=171)
        condition4=(img_array>171) & (img_array<=172)
        condition5=(img_array>173)

        img_array[condition0]=0
        img_array[condition1]=1
        img_array[condition2]=2
        img_array[condition3]=3
        img_array[condition4]=4
        img_array[condition5]=5
        
        image_mask.append(img_array)
    return image_mask
    

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


def aerial_patches(folder_path, patch_shape, stride,train=70, val=15, test=15):
    images=[] 
    masks=[]
    for i in range(1,9):
        img=load_images_from_tile(folder_path+"/Tile"+str(i))
        images=images+img
        msk=load_masks_from_tile(folder_path+"/Tile"+str(i))
        masks=masks+msk
    patches=extract_patches(images, patch_shape, stride)
    mask_patches=extract_patches(masks, patch_shape,stride)
    tam=int(patches.shape[0]/8)
    train_lim=int(tam*train/100)
    val_lim=int(tam*(train+val)/100)
    #train, test, val
    return patches[:train_lim,:,:], mask_patches[:train_lim,:,:],patches[val_lim:tam,:,:], mask_patches[val_lim:tam,:,:],patches[train_lim:val_lim,:,:], mask_patches[train_lim:val_lim,:,:],