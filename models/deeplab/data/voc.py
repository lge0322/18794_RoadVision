from PIL.ImageFile import ImageFile
from utils import ext_transforms as et
import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np
import re

from PIL import Image


def voc_cmap(N=256, normalized=False):
    """
    Initializing color map for the Pascal VOC dataset

    Args:
        N: Number of colors to generate in the color map
        normalized: If True, colors are normalized to range 0 and 1
    """
    
    # Extract specific bit from a number
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    # If normalized, we store values from 0 to 255
    # If not normalized, we store float32

    dtype = 'float32' if normalized else 'uint8'

    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    # color map for each category
    cmap = voc_cmap(N=21)

    def generate_russian_dataset(self, image_set, root):
        base_dir = f"russian_dataset/{image_set}"
        voc_root = os.path.join(root, base_dir)

        # Dictionaries to store paths for images and masks with their base name as the key
        images = {}
        masks = {}
        # Looping through the training dataset
        for filename in os.listdir(voc_root):
            file_path = os.path.join(voc_root, filename)
            if filename.endswith(".jpg") and os.path.isfile(file_path):
                base_name = re.sub(r'_mask', '', filename.split('.')[0])  # Remove '_mask' if it exists
                if base_name.isnumeric():
                  if "_mask" in filename:
                      # It's a mask
                      masks[base_name] = file_path
                  else:
                      # It's an image
                      images[base_name] = file_path
        
        return masks, images
    
    def generate_coco_dataset(self, image_set, root):
        base_dir = f"coco/{image_set}"
        image_root = os.path.join(root, base_dir, "data")
        mask_root = os.path.join(root, base_dir, "masks")

        # Dictionaries to store paths for images and masks with their base name as the key
        images = {}
        masks = {}

        for filename in os.listdir(image_root):
            file_path = os.path.join(image_root, filename)
            base_name = filename.split('.')[0]
            if base_name.isnumeric() and base_name != "000000342051":
              if os.path.isfile(file_path):
                  images[base_name] = file_path
        for filename in os.listdir(mask_root):
            file_path = os.path.join(mask_root, filename)
            base_name = re.sub(r'mask_', '', filename.split('.')[0])
            if base_name.isnumeric() and base_name != "000000342051":
              if os.path.isfile(file_path):
                masks[base_name] = file_path
        
        return masks, images

    def __init__(self,
                 root,
                 image_set='train',
                 dataset='russian',
                 transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform

        self.image_set = image_set

        if dataset == 'russian':
            self.masks, self.images = self.generate_russian_dataset(image_set, self.root)
        
        elif dataset == "coco":
            self.masks, self.images = self.generate_coco_dataset(image_set, self.root)

        elif dataset == "both":
            russian_masks, russian_images = self.generate_russian_dataset(image_set, self.root)
            coco_masks, coco_images = self.generate_coco_dataset(image_set, self.root)

        else:
            raise Exception("Invalid dataset name!")

        # Sort and align the images and masks
        if dataset == "both":
            russian_images = dict(sorted(russian_images.items(), key=lambda x: int(x[0])))
            russian_masks = dict(sorted(russian_masks.items(), key=lambda x: int(x[0])))
            russian_images = list(russian_images.values())
            russian_masks = list(russian_masks.values())

            coco_images = dict(sorted(coco_images.items(), key=lambda x: int(x[0])))
            coco_masks = dict(sorted(coco_masks.items(), key=lambda x: int(x[0])))
            coco_images = list(coco_images.values())
            coco_masks = list(coco_masks.values())

            self.images = []
            self.masks = []
            min_length = min(len(russian_images), len(coco_images))
            for i in range(min_length):
                self.images.append(russian_images[i])
                self.images.append(coco_images[i])
                self.masks.append(russian_masks[i])
                self.masks.append(coco_masks[i])

        else:
            self.images = dict(sorted(self.images.items(), key=lambda x: int(x[0])))
            self.masks = dict(sorted(self.masks.items(), key=lambda x: int(x[0])))

            self.images = list(self.images.values())
            self.masks = list(self.masks.values())
            

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.

            What you should do
            1. read the image (jpg) as an PIL image in RGB format.
            2. read the mask (png) as a single-channel PIL image.
            3. perform the necessary transforms on image & mask.
        """

        image_path = self.images[index]
        mask_path = self.masks[index]

        # 1. Read the image (jpg) as an PIL image in RGB format
        image = Image.open(image_path)

        # 2. Read the mask as a single-channel PIL image
        mask = Image.open(mask_path)

        # 3. Perform the necessary transforms on image and mask
        # Apply the transformation if it's set
        try:
          transformed_image, transformed_mask = self.transform(image, mask)
        except:
          print("Image index: ", index)
        return transformed_image, transformed_mask

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image for visualization, using the color map"""
        # TODO Problem 1.1
        # =================================================

        mask = np.array(mask)
        # Getting the CMAP from 21 categories
        cmap = cls.cmap

        # Create the new decoded mask
        decoded_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

        # Loop through each pixel in the mask
        for i in range(mask.shape[0]):  # Iterate over height
            for j in range(mask.shape[1]):  # Iterate over width
                class_index = mask[i, j]
                if class_index < len(cmap):  # Check if class index is within bounds
                    decoded_mask[i, j] = cmap[class_index]

        return decoded_mask
