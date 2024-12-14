import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycocotools.coco import COCO 
import cv2
import json
import random
import os

training_folder = 'sign_dataset/train'
validation_folder = 'sign_dataset/val'


def get_images_metadata(data_type):
    """
    Function to get the images_metadata parsed from the via_region_data

    args:
        data_type [str]: "train" or "val" 
    returns:
        images_metadata [dict]: Returns the metadata parsed from via_region_data

    """
    images_metadata = dict()

    with open(f'sign_dataset/{data_type}/via_region_data.json') as annot:
        images_metadata = json.load(annot)

    return images_metadata


def segment_dataset(images_metadata, data_type="train"):
    """
    args:
        images_metadata: metadata of the images from via_regions
        data_type: "train" or "val", depending on which dataset i want to segment
    """

    for image_key, image_metadata in images_metadata.items():
        image_name = images_metadata[image_key]['filename']
        image_path = f"sign_dataset/{data_type}/{image_name}"
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image {image_name}. Skipping.")
            continue
        image_height, image_width = image.shape[0], image.shape[1]

        segmentation_mask = np.zeros(
            (image_height, image_width), dtype=np.uint8)

        # Process each region in the metadata
        image_regions = image_metadata['regions']
        for region_num, region in image_regions.items():
            shape_attributes = region['shape_attributes']
            shape = shape_attributes.get('name', '')

            if shape == 'polygon':
                all_points_x = shape_attributes['all_points_x']
                all_points_y = shape_attributes['all_points_y']
                assert len(all_points_x) == len(
                    all_points_y), "X and Y points must have the same length"

                # Create a list of (x, y) points for the polygon
                points = np.array([(x, y) for x, y in zip(
                    all_points_x, all_points_y)], dtype=np.int32)
                points = points.reshape((-1, 1, 2))  # Reshape for cv2.fillPoly

                # Fill the polygon on the mask
                cv2.fillPoly(segmentation_mask, [points], color=255)

            elif shape == 'ellipse':
                cx = shape_attributes['cx']
                cy = shape_attributes['cy']
                rx = shape_attributes['rx']
                ry = shape_attributes['ry']
                center = (int(cx), int(cy))
                axes = (int(rx), int(ry))

                # Fill the ellipse on the mask
                cv2.ellipse(segmentation_mask, center, axes, angle=0,
                            startAngle=0, endAngle=360, color=255, thickness=-1)

            elif shape == 'circle':
                cx = shape_attributes['cx']
                cy = shape_attributes['cy']
                r = shape_attributes['r']
                center = (int(cx), int(cy))
                radius = int(r)

                # Fill the circle on the mask
                cv2.circle(segmentation_mask, center,
                           radius, color=255, thickness=-1)

            elif shape == 'rect':
                x = shape_attributes['x']
                y = shape_attributes['y']
                width = shape_attributes['width']
                height = shape_attributes['height']

                top_left = (int(x), int(y))
                bottom_right = (int(x + width), int(y + height))

                cv2.rectangle(segmentation_mask, top_left,
                              bottom_right, color=255, thickness=-1)

        mask_path = f"sign_dataset/{data_type}/{image_name.split('.')[0]}_mask.jpg"
        cv2.imwrite(mask_path, segmentation_mask)
        print(f"Saved mask for {image_name} at {mask_path}")


def find_classes(images_metadata):
    """
    Function to parse the images_metadata and output the different classes

    args:
        images_metadata [dict]: Returns the metadata parsed from via_region_data

    returns:
        shapes [dict]: Dictionary with keys as shapes and value as an example 
                       of how it looks like
    """

    shapes = dict()

    for key, info in images_metadata.items():
        filename = info['filename']
        regions = info['regions']

        for region in regions:
            shape_attributes = regions[region]['shape_attributes']
            shape = shape_attributes['name']
            if shape not in shapes:
                shapes[shape] = shape_attributes

    return shapes


def get_masks(data_type):
    """
    Function to get the masks of the images from the images_metadata

    args:
        data_type [str]: "train" or "val" 
    """

    images_metadata = get_images_metadata(data_type)

    segment_dataset(images_metadata, data_type)

    breakpoint()


if __name__ == "__main__":

    get_masks(data_type="train")
    get_masks(data_type="val")
