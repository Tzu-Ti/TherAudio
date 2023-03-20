import torch
import torchvision
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF

import numpy as np
import glob, os
from PIL import Image

class IdentityDataset(Dataset):
    def __init__(self, path, augment=True, train=True):
        self.augment = augment
        paths = glob.glob(os.path.join(path, "*.jpg"))
        l = len(paths)
        if train:
            self.paths = paths[:int(l*0.8)]
        else:
            self.paths = paths[int(l*0.8):]
        
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.aug = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            torchvision.transforms.RandomHorizontalFlip(p=1),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def __getitem__(self, index):
        img_path = self.paths[index]
        img = Image.open(img_path)
        
        if self.augment:
            if torch.rand(1) > 0.5:
                return self.aug(img)
            else:
                return self.transforms(img)

        return self.transforms(img)
    
    def __len__(self):
        return len(self.paths)