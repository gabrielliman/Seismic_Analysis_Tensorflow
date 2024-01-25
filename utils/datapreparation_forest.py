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
    return np.stack(image_data, axis=0)

def load_masks_from_tile(folder_path):
    image_data = []
    folder_path=folder_path+"/masks"
    for filename in sorted(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, filename)
        img = Image.open(image_path)
        img_gray = img.convert('L')
        img_array = np.array(img_gray)

        condition0=(img_array<128)
        condition1=(img_array>128)

        img_array[condition0]=0
        img_array[condition1]=1

        image_data.append(img_array)
    return np.stack(image_data, axis=0)
    
def forest_patches(folder_path,train=70, val=15, test=15):
    patches=load_images_from_tile(folder_path)
    mask_patches=load_masks_from_tile(folder_path)
    tam=patches.shape[0]
    train_lim=int(tam*train/100)
    val_lim=int(tam*(train+val)/100)
    #train, test, val
    return patches[:train_lim,:,:], mask_patches[:train_lim,:,:],patches[val_lim:,:,:], mask_patches[val_lim:,:,:],patches[train_lim:val_lim,:,:], mask_patches[train_lim:val_lim,:,:],