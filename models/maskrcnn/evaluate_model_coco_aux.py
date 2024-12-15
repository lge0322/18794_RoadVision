import os
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from matplotlib.patches import Rectangle

################################################################################
# Dataset Preparation
################################################################################

coco_val_images = "dataset/aux_images/"
trained_weights_path = "checkpoints/mask_rcnn_coco_20epochs_001_4batches.pth"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Extract images manually
image_paths = [os.path.join(coco_val_images, fname) for fname in os.listdir(coco_val_images)]

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
# Visualize the mask outputs on the original images from aux_images
################################################################################

plt.figure(figsize=(10, 30))

# Display six images
for i in range(len(image_paths)):
    print(image_paths[i])
    image = Image.open(image_paths[i])
    img_tensor = transform(image)

    model.eval()
    img_tensor = img_tensor.to(device)
    output = model(img_tensor.unsqueeze(0))
    
    # Obtain predicted masks
    masks = output[0]['masks'].detach().cpu().numpy()
    scores = output[0]['scores'].detach().cpu().numpy()

    score_threshold = 0.5
    valid_indices = np.where(scores > score_threshold)[0]
    
    mask_vis = None
    
    if valid_indices.size > 0:
        # Use the mask with the highest score 
        best_index = valid_indices[0]  
        pred_mask = masks[best_index][0] 
        mask_vis = pred_mask
        
        # Find the bounding box
        rows_with_true = np.where(np.any(pred_mask, axis=1))[0]
        cols_with_true = np.where(np.any(pred_mask, axis=0))[0]

        if rows_with_true.size > 0 and cols_with_true.size > 0:
            min_row, max_row = rows_with_true[0], rows_with_true[-1]
            min_col, max_col = cols_with_true[0], cols_with_true[-1]
        else:
            # No non-zero values found
            min_row = max_row = min_col = max_col = None
        
        # Predicted mask
        ax = plt.subplot(6, 3, 3*i+1)
        ax.imshow(pred_mask, cmap="gray")
        ax.set_title(f"Predicted Mask for Image {i+1}")
        ax.axis("off")

        # Mask overlay
        ax = plt.subplot(6, 3, 3*i+2)
        ax.imshow(image)  # Show the original image
        ax.imshow(mask_vis, cmap='Greens', alpha=0.6)  
        ax.set_title("Mask Overlay")
        ax.axis("off")
        
        # Mask overlay with bounding box
        ax = plt.subplot(6, 3, 3*i+3)
        ax.imshow(image)
        ax.imshow(mask_vis, cmap='Greens', alpha=0.6)  
        if min_row is not None and min_col is not None:
            rect = Rectangle((min_col, min_row), max_col - min_col + 1, max_row - min_row + 1,
                            linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        ax.set_title("Mask Overlay with Bounding Box")
        ax.axis("off")
plt.tight_layout()
plt.savefig(f"final_results/eval_combined_aux.jpg")
plt.show()
plt.close()

        
