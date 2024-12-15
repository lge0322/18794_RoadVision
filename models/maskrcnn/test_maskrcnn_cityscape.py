import torch
import matplotlib.pyplot as plt
from torchvision import models
from PIL import Image
from cityscapes_dataset import get_cityscapes_data

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = models.detection.maskrcnn_resnet50_fpn(pretrained=False)
model.load_state_dict(torch.load("mask_rcnn_cityscapes.pth"))
model.to(device)
model.eval()

image_path = "/path/to/test/image.png"
image = Image.open(image_path).convert("RGB")
annotation_file = "/path/to/test/annotation.json"
boxes, labels, masks, image_id = get_cityscapes_data(annotation_file, "/path/to/images")

# Perform inference
with torch.no_grad():
    prediction = model([image.to(device)])[0]

# Visualize the results
plt.imshow(image)
plt.imshow(prediction['masks'][0, 0].cpu().numpy(), cmap='gray', alpha=0.5)
plt.show()