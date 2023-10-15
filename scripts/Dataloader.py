import os
import json

import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from kornia.augmentation import *
import glob

class CocoAtribDataset(Dataset):
    def __init__(
        self, 
        path_to_images: str, 
        path_to_json: str, 
        keys_outputs: list,
        transform=None,
        categorical_type_to_int: dict=None,
    ):
        
        self.path_to_images = path_to_images
        self.transform = transform
        self.keys_outputs = keys_outputs

        with open(path_to_json, 'r') as file:
            json_data = json.load(file)
            
        self.annotations = json_data['annotations']

        if categorical_type_to_int:
            for key_o in categorical_type_to_int.keys():
                for annotation in self.annotations:
                    annotation['attributes'][key_o] = categorical_type_to_int[annotation['attributes'][key_o]]

        self.images_id_to_name = {image['id']: image['file_name']
                                    for image in json_data['images']}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        annotation = self.annotations[idx]
        bbox = np.array(annotation['bbox']).astype(int)

        path_to_image = os.path.join(self.path_to_images, self.images_id_to_name[annotation['image_id']])
        
        img = Image.open(path_to_image)  # RGB
        r, g, b = img.split()
        img = Image.merge("RGB", (b, g, r))  # BGR
        
        img = img.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))
        
        if self.transform:
            image_cut = self.transform(img)
        image_cut = image_cut[0][[2, 1, 0], :, :]
        
        output = [int(annotation['attributes'][key]) for key in self.keys_outputs]
        
        return image_cut, output

class CSVDataset(Dataset):
    def __init__(self,
                 path_to_images,
                 csv_path,
                 keys_outputs,
                 transform=None,
                 categorical_type_to_int: dict=None,
                 need_crop: bool=False) -> None:
        
        import pandas as pd

        self.path_to_images = path_to_images
        self.transform = transform
        self.keys_outputs = keys_outputs
        self.need_crop = need_crop

        df = pd.read_csv(csv_path)

        if categorical_type_to_int:
            for key_o in categorical_type_to_int.keys():
                df[key_o] = df[key_o].map(categorical_type_to_int[key_o])
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        annotation = self.data.iloc[idx].to_dict()
        # annotation = self.annotations[idx]
        if 'bbox' in annotation.keys():
            bbox = np.array(annotation['bbox']).astype(int)
        else:
            self.need_crop = False # just in case of forgetting to set it to False

        path_to_image = os.path.join(self.path_to_images, annotation['im_name'])
        
        img = Image.open(path_to_image)  # RGB
        r, g, b = img.split()
        img = Image.merge("RGB", (b, g, r))  # BGR
        if self.need_crop:
            img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        
        if self.transform:
            image_cut = self.transform(img)
        image_cut = image_cut[0][[2, 1, 0], :, :]
        
        output = [int(annotation[key]) for key in self.keys_outputs]
        
        return image_cut, output

class ImFolDataset(Dataset):
    def __init__(self,
                 path_to_images,
                 transform=None):
        self.path_to_images = path_to_images
        self.transform = transform

        self.files = glob.glob(path_to_images + '/**/*.jpg', recursive=True)
        self.mapping = dict((fol, id_) for id_, fol in enumerate(os.listdir(path_to_images)))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path_to_image = self.files[idx]
        
        img = Image.open(path_to_image)  # RGB
        r, g, b = img.split()
        img = Image.merge("RGB", (b, g, r))  # BGR
                
        if self.transform:
            image_cut = self.transform(img)
        image_cut = image_cut[0][[2, 1, 0], :, :]
        
        output = [self.mapping[os.path.basename(os.path.dirname(path_to_image))]]
        
        return image_cut, output