import numpy as np
import torch
from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset

class SignatureDataset(Dataset):
    def __init__(self, label_csv_dir=None, images_data_dir=None, transform=None) -> None:
        # used to prepare the labels and images path
        if label_csv_dir is None:
            self.dataframe = pd.read_csv(os.getcwd() + '/dataset/signature/train_data.csv')
        else: 
            self.dataframe = pd.read_csv(label_csv_dir)
        self.dataframe.columns =["image1", "image2", "label"]
        
        if images_data_dir is None:
            self.images_dir = os.getcwd() + '/dataset/signature/train'
        else:
            self.images_dir = images_data_dir
        self.transform = transform
    
    def __getitem__(self, index) -> None:
        # getting the image path
        image1_path=os.path.join(self.training_dir,self.train_df.iat[index, 0])
        image2_path=os.path.join(self.training_dir,self.train_df.iat[index, 1])
        
        # Loading the image
        img1 = Image.open(image1_path)
        img2 = Image.open(image2_path)
        img1 = img1.convert("L")
        img2 = img2.convert("L")
        
        # Apply image transformations
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.from_numpy(np.array([int(self.train_df.iat[index,2])], dtype=np.float32))
    
    def __len__(self):
        return len(self.train_df)
