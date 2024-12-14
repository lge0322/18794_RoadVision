# RoadVision
The aim of RoadVision is to build an intelligent system that can enhance driver safety and quick decision-making in complex environments.

# Data
The datasets used in this project include: CityScape, Russian Road Sign, and COCO-2017. Given that a lot of the dataset was not appropriately labeled with the type of traffic sign, we reduced this to a binary segmentation problem where the relevant classes are “background (0)” and “traffic sign (1)”.

Data processing code can be found in `data_processing`

## DeepLabV3
`Running_DeepLab.ipynb` provides a pipeline for running the DeepLabV3 training and evaluation. The demo also includes the OCR pipeline performed on the output segmentation mask.

## Mask R-CNN


## Maskformer
We experimented with Maskformer but it did not yield promising results. It also required too much memory to train which was unsupported by the EC2 instance.
