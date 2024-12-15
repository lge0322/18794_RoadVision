import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2

################################################################################
# Dataset Preparation
################################################################################

coco_train_images = "dataset/coco-2017/train/train_images"
coco_val_images = "dataset/coco-2017/validation/val_images"
filtered_train_annotations = "dataset/coco-2017/train/stop_labels_1700.json"
filtered_val_annotations = "dataset/coco-2017/validation/stop_labels_1700.json"
pretrained_weights_path = "mask_rcnn_cityscapes.pth"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CocoDetection(coco_train_images, filtered_train_annotations, transform)
val_dataset = CocoDetection(coco_val_images, filtered_val_annotations, transform)

# Tranpose batch of data using zip
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, 
                          collate_fn=lambda batch: tuple(zip(*batch))) 
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, 
                        collate_fn=lambda batch: tuple(zip(*batch)))

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

print(f"Loading checkpoints from {pretrained_weights_path}...")
model.load_state_dict(torch.load(pretrained_weights_path))

################################################################################
# Device setup
################################################################################

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

################################################################################
# Parameters
################################################################################

optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], 0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, 0.1)

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
# Validation Loop
################################################################################

num_epochs = 20
losses_list, accuracies_list, miou_list = [], [], []

def calculate_iou(mask1, mask2):
    mask1, mask2 = mask1.bool(), mask2.bool()
    intersection = (mask1 & mask2).float().sum()
    union = (mask1 | mask2).float().sum()
    return intersection / (union + 1e-6)

def validate_model(model, data_loader):
    model.eval()
    total_correct, total_instances = 0, 0
    iou_sum_mask, iou_sum_background = 0, 0
    num_masks, num_background = 0, 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [convert_coco_annotations(t, (img.size(1), img.size(2))) for t, img in zip(targets, images)]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            for i, output in enumerate(outputs):
                preds_labels = output['labels']
                true_labels = targets[i]['labels']
                total_correct += abs(preds_labels.size(0)-true_labels.size(0))/max(preds_labels.size(0),true_labels.size(0))
                total_instances += len(true_labels)

                masks_pred = (output['masks'] > 0.7).squeeze(1)
                masks_true = targets[i]['masks'].squeeze(1)
                
                # if False:
                #     masks_pred = masks_pred[0].cpu().detach().numpy() 
                #     masks_true = masks_true[0].cpu().detach().numpy() 

                #     # Visualize images
                #     plt.figure(figsize=(10, 5))

                #     # True mask
                #     plt.subplot(1, 2, 1)
                #     plt.title("True Mask")
                #     plt.imshow(masks_true, cmap="gray")
                #     plt.axis("off")

                #     # Predicted mask
                #     plt.subplot(1, 2, 2)
                #     plt.title("Predicted Mask")
                #     plt.imshow(masks_pred, cmap="gray")
                #     plt.axis("off")

                #     plt.tight_layout()
                #     plt.savefig("output_viz.png")

                for pred_mask, true_mask in zip(masks_pred, masks_true):
                    iou_sum_background += calculate_iou((pred_mask == 0), (true_mask == 0))
                    iou_sum_mask += calculate_iou((pred_mask == 1), (true_mask == 1))
                    num_masks += 1 
                    num_background += 1

    accuracy = total_correct / float(total_instances) if total_instances > 0 else 0
    mean_iou = (iou_sum_mask + iou_sum_background) / float(num_masks + num_background) if (num_masks + num_background) > 0 else 0
    return accuracy, mean_iou

################################################################################
# Training Loop
################################################################################

for epoch in tqdm(range(num_epochs)):
    model.train()
    epoch_loss = 0
    for images, targets in tqdm(train_loader, leave=False):
        images = [img.to(device) for img in images]
        # convert coco annotations into appropriate targets for maskrcnn
        targets = [convert_coco_annotations(t, (img.size(1), img.size(2))) for t, img in zip(targets, images)]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()
        
    scheduler.step()
    epoch_loss /= len(train_loader)
    losses_list.append(epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    accuracy, mean_iou = validate_model(model, val_loader)
    accuracies_list.append(accuracy)
    miou_list.append(mean_iou.cpu())
    print(f"Validation Accuracy: {accuracy}, mIoU: {mean_iou}")

################################################################################
# Plot loss, accuracy, and mIoU graphs
################################################################################

epochs = range(1, num_epochs + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, losses_list, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig(f'loss_curve_{num_epochs}epochs_001_8batches_final.png')

plt.figure(figsize=(10, 5))
plt.plot(epochs, accuracies_list, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.savefig(f'accuracy_curve_{num_epochs}epochs_001_8batches_final.png')

plt.figure(figsize=(10, 5))
plt.plot(epochs, miou_list, label='mIoU')
plt.xlabel('Epoch')
plt.ylabel('mIoU')
plt.title('Mean IoU (mIoU)')
plt.legend()
plt.savefig(f'miou_curve_{num_epochs}epochs_001_8batches_final.png')

# Save the model
torch.save(model.state_dict(), f"mask_rcnn_coco_{num_epochs}epochs_001_8batches_final.pth")
