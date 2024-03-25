# ./utils/dataset.py
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

class HeadGearDataset(Dataset):
    def __init__(self, mode, transform=None, target_transform=None):
        self.annotation_path = '/NasData/home/kmg/mlcl/MLCL_2023/data/headgear/headgear.csv'
        self.dataset_path = '/NasData/home/kmg/mlcl/MLCL_2023/data/headgear'

        self.img_labels = pd.read_csv(self.annotation_path)
        self.img_labels = self.img_labels[self.img_labels['data set'] == mode]
        
        # TODO: Define the attributes of this dataset
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        # TODO: Return the length of the dataset
        return len(self.img_labels)
        
    def __getitem__(self, idx):
        # TODO: Return the idx-th item of the dataset
        img_path = self.img_labels['filepaths'].iloc[idx]  # 'filepaths' column
        
        # TODO: path join
        img_path = os.path.join(self.dataset_path, img_path)
        image = self.load_image(img_path)
        
        # TODO: Return the idx-th item of the dataset
        label = self.img_labels['class id'].iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def load_image(self, path):
        with Image.open(path) as img:
            img.load()  # This forces the image file to be read into memory
            return img  # Return the PIL image directly
