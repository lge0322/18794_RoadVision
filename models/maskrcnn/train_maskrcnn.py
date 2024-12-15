import os
import torch
from torchvision import models
from torch.utils.data import DataLoader
from torch import optim
from cityscapes_dataset import CityscapesMaskRCNN, get_all_annotation_files
import torchvision.transforms as T
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.spatial import ConvexHull
import numpy as np

dataset_base_dir = "dataset"
annotation_files = get_all_annotation_files(dataset_base_dir)
image_dir = os.path.join(dataset_base_dir, "leftImg8bit", "train")

# Initialize data
dataset = CityscapesMaskRCNN(annotation_files, image_dir)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Load the Mask R-CNN model
model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
model.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(
    model.roi_heads.mask_predictor.conv5_mask.in_channels,
    256,
    2
)

# Move model to device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Set up optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, 0.1)

# Training and validation metrics
num_epochs = 10
losses_list = []
accuracies_list = []
miou_list = []

# Helper function to calculate IoU
def calculate_iou(mask1, mask2):
    intersection = (mask1 & mask2).float().sum((1, 2))
    union = (mask1 | mask2).float().sum((1, 2))
    iou = intersection / union
    return iou.mean().item()

# Validation function
def validate_model(model, data_loader):
    model.eval()
    total_correct = 0
    total_instances = 0
    iou_sum = 0
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [T.ToTensor()(image).to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            
            for i, output in enumerate(outputs):
                preds = output['labels'].cpu()
                true_labels = targets[i]['labels'].cpu()
                masks_pred = (output['masks'] > 0.5).squeeze(1).cpu()  # Convert to binary
                masks_true = targets[i]['masks'].cpu()

                total_correct += (preds == true_labels).sum().item()
                total_instances += len(true_labels)
                
                # Calculate IoU for each mask
                iou_sum += calculate_iou(masks_pred, masks_true)

    accuracy = total_correct / total_instances if total_instances > 0 else 0
    mean_iou = iou_sum / num_batches if num_batches > 0 else 0
    return accuracy, mean_iou

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for images, targets in data_loader:
        images = [T.ToTensor()(image).to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()
    
    scheduler.step()
    epoch_loss /= len(data_loader)
    losses_list.append(epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

    # Validation step for accuracy and mIoU
    accuracy, mean_iou = validate_model(model, data_loader)
    accuracies_list.append(accuracy)
    miou_list.append(mean_iou)
    print(f"Validation Accuracy: {accuracy:.4f}, mIoU: {mean_iou:.4f}")

# Plotting the metrics
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, losses_list, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('loss_curve.png')

plt.figure(figsize=(10, 5))
plt.plot(epochs, accuracies_list, label='Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.savefig('accuracy_curve.png')

plt.figure(figsize=(10, 5))
plt.plot(epochs, miou_list, label='mIoU', color='blue')
plt.xlabel('Epoch')
plt.ylabel('mIoU')
plt.title('Mean IoU (mIoU)')
plt.legend()
plt.savefig('miou_curve.png')

# Save the model
torch.save(model.state_dict(), "mask_rcnn_cityscapes.pth")
