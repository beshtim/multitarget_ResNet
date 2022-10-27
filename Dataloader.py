import os
import json

import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from kornia.augmentation import *


class TrafficLightsDataset(Dataset):
    def __init__(
        self, 
        path_to_images: str, 
        path_to_json: str, 
        keys_outputs: list,
        transform=None,
        from_general_type_to_int: dict=None,
        from_type_to_int: dict=None
    ):
        
        self.path_to_images = path_to_images
        self.transform = transform
        self.keys_outputs = keys_outputs

        with open(path_to_json, 'r') as file:
            json_data = json.load(file)
            
        self.annotations = json_data['annotations']

        if from_general_type_to_int:
            for annotation in self.annotations:
                annotation['attributes']['general_type'] = from_general_type_to_int[annotation['attributes']['general_type']]
        
        if from_type_to_int:
            for annotation in self.annotations:
                annotation['attributes']['type'] = from_type_to_int[annotation['attributes']['type']]

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
