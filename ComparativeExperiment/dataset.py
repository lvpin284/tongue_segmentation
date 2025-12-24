import os
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2
import numpy as np

class TongueDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        
        path = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root_dir, path)
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
             raise FileNotFoundError(f"Image not found at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create mask
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for ann in coco_annotation:
            mask += coco.annToMask(ann)
        
        mask = np.clip(mask, 0, 1)
        
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
        return img, mask

    def __len__(self):
        return len(self.ids)
