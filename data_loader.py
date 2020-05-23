
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from skimage import io, transform
from PIL import Image, ImageOps
import os
import numpy as np

#os.listdir() method in python is used to get the list of all files and directories in the specified directory.
#The index() method searches an element in the list and returns its index.
#The open() function opens a file, and returns it as a file object.


class NucleiSeg(Dataset):
    def __init__(self, path='/home/naveen/v3/Cs_Project/Dataset/Train/Image/', transforms=None):
        self.path = path
        self.list = os.listdir(self.path)   
        
        self.transforms = transforms
        
    def __getitem__(self, index):
        # stuff
        image_path = '/home/naveen/v3/Cs_Project/Dataset/Train/Image/'
        mask_path = '/home/naveen/v3/Cs_Project/Dataset/Train/Mask/mask'
        image = Image.open(image_path+self.list[index])
        image = ImageOps.expand(image, border=12, fill=0) 
        image = image.convert('RGB')
        mask = Image.open(mask_path+self.list[index])
        mask = ImageOps.expand(mask, border=12, fill=0)
        mask = mask.convert('L')
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        return (image, mask)

    def __len__(self):
        return len(self.list) # of how many data(images?) you have

    
class NucleiSegVal(Dataset):
    def __init__(self, path='/home/naveen/v3/Cs_Project/Dataset/Val/Image/', transforms=None):
        self.path = path
        self.list = os.listdir(self.path)
        
        self.transforms = transforms
        
    def __getitem__(self, index):
        # stuff
        image_path = '/home/naveen/v3/Cs_Project/Dataset/Val/Image/'
        mask_path = '/home/naveen/v3/Cs_Project/Dataset/Val/Mask/mask'
        image_name = self.list[index]
        image = Image.open(image_path+self.list[index])
        image = ImageOps.expand(image, border=12, fill=0)
        image = image.convert('RGB')
        mask = Image.open(mask_path+self.list[index])
        mask = ImageOps.expand(mask, border=12, fill=0)
        mask = mask.convert('L')
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        return (image, mask, image_name )
        
    def __len__(self):
        return len(self.list)
        
class NucleiSegTest(Dataset):
    def __init__(self, path='/home/naveen/v3/Cs_Project/Dataset/Test/Image/', transforms=None):
        self.path = path
        self.list = os.listdir(self.path)
        
        self.transforms = transforms
        
    def __getitem__(self, index):
        # stuff
        image_path = '/home/naveen/v3/Cs_Project/Dataset/Test/Image/'
        mask_path = '/home/naveen/v3/Cs_Project/Dataset/Test/Mask/mask'
        image_name = self.list[index]
        image = Image.open(image_path+self.list[index])
        image = ImageOps.expand(image, border=12, fill=0)
        image = image.convert('RGB')
        mask = Image.open(mask_path+self.list[index])
        mask = ImageOps.expand(mask, border=12, fill=0)
        mask = mask.convert('L')
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        return (image, mask, image_name )

    def __len__(self):
        return len(self.list)
