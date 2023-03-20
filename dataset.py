import torch
import torchvision
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF

import numpy as np
import glob
from PIL import Image

class IdentityDataset(Dataset):
    def __init__(self, augment=True, train=True):
        self.augment = augment
        paths = glob.glob("../m_black_1/*.jpg")
        l = len(paths)
        if train:
            self.paths = paths[:int(l*0.8)]
        else:
            self.paths = paths[int(l*0.8):]
        
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.trainsforms_aug = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def __getitem__(self, index):
        img_path = self.paths[index]
        img = Image.open(img_path)
        
        if self.augment:
            img = self.trainsforms_aug(img)
            if torch.rand(1) > 0.5:
                img = TF.hflip(img)
            return img
        
        img = self.transforms(img)
        return img
    
    def __len__(self):
        return len(self.paths)