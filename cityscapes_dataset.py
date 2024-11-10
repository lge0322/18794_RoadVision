import os
import json
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

def get_cityscapes_data(file_path, img_dir):
    """
    Load and process the Cityscapes JSON annotations and corresponding images.
    Returns: 
        boxes, labels, masks, image_id
    """
    with open(file_path, 'r') as f:
        annotation = json.load(f)
        
    # Initialize lists to store data
    boxes = []
    labels = []
    masks = []
    image_id = torch.tensor([0])  # For simplicity, use a constant ID for each image
    
    # For each object in the annotation
    for obj in annotation["objects"]:
        if obj["label"] == "traffic sign":
            label = 1  # Label for road signs (background will be 0)
        else:
            label = 0  # Background class
        
        # Get polygon points and convert to bounding box
        polygon = np.array(obj["polygon"])
        mask = np.zeros((annotation["imgHeight"], annotation["imgWidth"]), dtype=np.uint8)
        mask = cv2.fillPoly(mask, [polygon], 1)
        
        # Calculate the bounding box
        x_min = np.min(polygon[:, 0])
        x_max = np.max(polygon[:, 0])
        y_min = np.min(polygon[:, 1])
        y_max = np.max(polygon[:, 1])
        
        if x_max <= x_min or y_max <= y_min:
            continue 
        box = [x_min, y_min, x_max, y_max]
        
        boxes.append(box)
        labels.append(label)
        masks.append(mask)
        
    boxes = np.array(boxes) 
    labels = np.array(labels)
    masks = np.array(masks)
    
    # Convert to tensor
    boxes = torch.tensor(boxes)
    labels = torch.tensor(labels)
    masks = torch.tensor(masks)

    return boxes, labels, masks, image_id


class CityscapesMaskRCNN(Dataset):
    """
    Custom Dataset class for the Cityscapes dataset with only 'traffic sign' and 'background' classes.
    """
    def __init__(self, annotation_files, img_dir, transform=None):
        self.annotation_files = annotation_files
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.annotation_files)
    
    def __getitem__(self, idx):
        annotation_file = self.annotation_files[idx]
        
        # Extract the directory, base name, and create the corresponding image file name
        annotation_dir, annotation_filename = os.path.split(annotation_file)
        base_name = annotation_filename.split("_gt")[0]  # Get the part before "_gt"
        
        # Construct the expected image file name with `_leftImg8bit.png` suffix
        image_filename = f"{base_name}_leftImg8bit.png"
        
        # Determine the image directory based on the annotation directory structure
        image_dir = annotation_dir.replace("gtFine", "leftImg8bit").replace("gtCoarse", "leftImg8bit")
        image_path = os.path.join(image_dir, image_filename)
        
        # Load image and annotations
        image = Image.open(image_path).convert("RGB")
        boxes, labels, masks, image_id = get_cityscapes_data(annotation_file, self.img_dir)
        
        # Build the target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64)
        }

        if self.transform:
            image, target = self.transform(image, target)
        
        return image, target


def get_all_annotation_files(base_dir):
    """
    Recursively finds all annotation files in the dataset folder (gtFine or gtCoarse).
    """
    search_fine = os.path.join(base_dir, "gtFine", "**", "train", "**", "*_gtFine_polygons.json")
    search_coarse = os.path.join(base_dir, "gtCoarse", "**", "train", "**", "*_gtCoarse_polygons.json")
    
    files_fine = glob.glob(search_fine, recursive=True)
    files_coarse = glob.glob(search_coarse, recursive=True)
    
    # Concatenate fine and coarse
    return files_fine + files_coarse
