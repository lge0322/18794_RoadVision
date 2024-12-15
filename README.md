# RoadVision
The aim of RoadVision is to build an intelligent system that can enhance driver safety and quick decision-making in complex environments.

# Data
The datasets used in this project include: CityScape, Russian Road Sign, and COCO-2017. Given that a lot of the dataset was not appropriately labeled with the type of traffic sign, we reduced this to a binary segmentation problem where the relevant classes are “background (0)” and “traffic sign (1)”.

Data processing code can be found in `data_processing`

## DeepLabV3
### Training
Training can be run using the following command:
`python main.py --model=deeplabv3_resnet50 --dataset=coco --batch_size=8 --total_itrs=3000`

The `--dataset` flag can be used to specify which dataset to train on: Russian dataset (`russian`), COCO-2017 stop sign (`coco`) or the combination of both (`both`).

### Evaluation
Evaluation can be run using the following command, using `--save_val_results` will generate the visualizations of the model's predicted output.

`python main.py --test_only --save_val_results --model=deeplabv3_resnet50 --dataset=coco --ckpt=/content/drive/MyDrive/794_project/'Colab Notebooks'/checkpoints/latest_deeplabv3_resnet50_VOC_os16_lr0.pth`

### Visualization
`Running_DeepLab.ipynb` provides a pipeline for running the DeepLabV3 visuzalition. The demo also includes the OCR pipeline performed on the output segmentation mask.

## Mask R-CNN
### COCO-2017 Stop Sign Dataset
Training and evaluation of the Mask R-CNN model trained on the COCO-2017 stop sign dataset can be performed as follows:
- Training can be run using `train_maskrcnn_coco.py`
- Evaluation of the trained model can be run using `evaluate_model_coco.py`
- Evaluation of the trained model on out of distribution datasets can be run using `evaluate_model_coco_aux.py`

### Cityscape Dataset
Training and evaluation of the Mask R-CNN model trained on Cityscape dataset can be performed as follows:
- Training can be run using `train_maskrcnn_cityscape.py`
- Evaluation of the trained model can be run using `test_maskrcnn_cityscape.py`

### Visualization
`Running_MaskRCNN.ipynb` provides a pipeline for running the MaskRCNN visualization. The demo also includes the OCR pipeline performed on the output segmentation mask.

## Maskformer
We experimented with Maskformer but it did not yield promising results. It also required too much memory to train which was unsupported by the EC2 instance.
