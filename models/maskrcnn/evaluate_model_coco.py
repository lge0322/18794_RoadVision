import os
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

################################################################################
# Dataset Preparation
################################################################################

coco_val_images = "dataset/coco-2017/validation/data"
filtered_val_annotations = "dataset/coco-2017/validation/stop_labels.json"
trained_weights_path = "checkpoints/mask_rcnn_coco_20epochs_001_4batches.pth"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = CocoDetection(root=coco_val_images, annFile=filtered_val_annotations, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False, num_workers=4, collate_fn=lambda batch: tuple(zip(*batch)))

################################################################################
# Model Initialization
################################################################################

# 2 classes (background and stop sign)
model = models.detection.maskrcnn_resnet50_fpn(weights=models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
    model.roi_heads.box_predictor.cls_score.in_features, 2)
model.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(
    model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2)

################################################################################
# Load checkpoints
################################################################################

print(f"Loading checkpoints from {trained_weights_path}...")
model.load_state_dict(torch.load(trained_weights_path))

################################################################################
# Device setup
################################################################################

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

model.eval()

################################################################################
# Visualize the mask outputs on the original images
################################################################################

def visualize_masks(images, targets, outputs):
    plt.figure(figsize=(10, 25))
    print(len(images))
    # Display five images
    for i in range(5):
        image = images[i].cpu().numpy().transpose((1, 2, 0))  # (H, W, C)
        image = np.clip(image * 255, 0, 255).astype(np.uint8)  
        output = outputs[i]
        
        # Obtain actual and predicted masks
        threshold = 0.7
        masks_pred = (output['masks'] > threshold).squeeze(1)
        masks_true = targets[i]['masks'].squeeze(1)

        masks_pred = masks_pred[0].cpu().detach().numpy() 
        masks_true = masks_true[0].cpu().detach().numpy() 

        # True mask
        plt.subplot(5, 3, 3*i+1)
        plt.imshow(masks_true, cmap="gray")
        plt.title(f"True Mask for Image {i+1}")
        plt.axis("off")

        # Predicted mask
        plt.subplot(5, 3, 3*i+2)
        plt.imshow(masks_pred, cmap="gray")
        plt.title(f"Predicted Mask for Image {i+1}")
        plt.axis("off")

        # Overlay predicted masks on the original image using alpha blending
        mask_color = np.zeros_like(image) 
        mask_color[masks_pred == 1] = [0, 255, 0]  
        image = cv2.addWeighted(image, 0.5, mask_color, 0.5, 0)

        plt.subplot(5, 3, 3*i+3)
        plt.imshow(image)
        plt.title(f"Mask Overlay for Image {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"final_results/eval_combined_2.jpg")
    plt.show()
    plt.close()

################################################################################
# Convert COCO annotations into Mask R-CNN targets
################################################################################

def convert_coco_annotations(annotations, image_size):
    boxes = []
    labels = []
    masks = []

    for ann in annotations:
        # Bounding box
        bbox = ann['bbox']
        boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        
        # Convert polygon into masks
        mask = np.zeros(image_size, dtype=np.uint8)
        polygon = ann['segmentation'][0]
        polygon = np.array([(x, y) for x, y in zip(polygon[::2], polygon[1::2])], dtype=np.int32)
        cv2.fillPoly(mask, [polygon], color=255)
        mask = mask.astype(np.uint8)
        mask[mask == 255] = 1
        masks.append(mask)
        
        # Labels
        labels.append(1)
    
    target = {
        'boxes': torch.as_tensor(boxes, dtype=torch.float32),
        'labels': torch.as_tensor(labels, dtype=torch.int64),
        'masks': torch.as_tensor(masks, dtype=torch.uint8),
        'image_id': torch.tensor([annotations[0]['image_id']])
    }
    return target

################################################################################
# Evaluate on the validation dataset
################################################################################

with torch.no_grad():
    count = 0
    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        targets = [convert_coco_annotations(t, (img.size(1), img.size(2))) for t, img in zip(targets, images)]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)
        
        # Visualize the masks
        visualize_masks(images, targets, outputs)
        break
        
