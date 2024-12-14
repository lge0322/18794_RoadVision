import os
import json

import numpy as np
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import albumentations as A
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset, DatasetDict, Image
import evaluate
from transformers import (
    MaskFormerImageProcessor, 
    MaskFormerForInstanceSegmentation,
    Trainer
)

root = "/home/ubuntu/maskrcnn_training/dataset/coco-2017/"

class ImageSegmentationDataset(TorchDataset):
    """Image segmentation dataset."""

    def __init__(self, dataset, transform):
        """
        Args:
            dataset
        """
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        original_image = np.array(self.dataset[idx]['image'])
        original_segmentation_map = np.array(self.dataset[idx]['binary_label'])
        
        if self.transform != None:
            transformed = self.transform(image=original_image, mask=original_segmentation_map)
            image, segmentation_map = transformed['image'], transformed['mask']
        else:
            image, segmentation_map = original_image, original_segmentation_map
        # convert to C, H, W
        image = image.transpose(2,0,1)

        return image, segmentation_map, original_image, original_segmentation_map

def collate_fn(batch):
    inputs = list(zip(*batch))
    images = inputs[0]
    segmentation_maps = inputs[1]
    # this function pads the inputs to the same size,
    # and creates a pixel mask
    # actually padding isn't required here since we are cropping
    preprocessor = MaskFormerImageProcessor(ignore_index=0, reduce_labels=False, 
                                            do_resize=True, do_rescale=False, do_normalize=True)

    batch = preprocessor(
        images,
        segmentation_maps=segmentation_maps,
        return_tensors="pt",
    )

    batch["original_images"] = inputs[2]
    batch["original_segmentation_maps"] = inputs[3]
    
    return batch


def visualize_mask(labels, label_name, batch):
  print("Label:", label_name)
  idx = labels.index(label_name)

  visual_mask = (batch["mask_labels"][0][idx].bool().numpy() * 255).astype(np.uint8)
  return Image.fromarray(visual_mask)


def process_label(example):
    threshold = 255//2
    # Convert label image to numpy array
    label = np.array(example["label"])
    # Apply thresholding
    binary_label = np.where(label < threshold, 0, 1).astype(np.uint8)
    # Convert back to PIL Image
    binary_label_pil = PILImage.fromarray(binary_label)  # Scale to 0-255 for saving
    return {"binary_label": binary_label_pil}

def create_dataset(image_paths, label_paths):
    dataset = Dataset.from_dict({"image": sorted(image_paths),
                                "label": sorted(label_paths)})
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image())
    dataset = dataset.map(process_label)
    
    return dataset


def get_train_test_ds():
    # Parsing image paths
    image_paths_train = [os.path.join(root, "train/train_images", img_name) for img_name in os.listdir(os.path.join(root, "train/train_images"))]
    label_paths_train = [os.path.join(root, "train/masks", img_name) for img_name in os.listdir(os.path.join(root, "train/masks"))]
    image_paths_validation = [os.path.join(root, "validation/val_images", img_name) for img_name in os.listdir(os.path.join(root, "validation/val_images"))]
    label_paths_validation = [os.path.join(root, "validation/masks", img_name) for img_name in os.listdir(os.path.join(root, "validation/masks"))]

    print(len(image_paths_train), len(label_paths_train))
    print(len(image_paths_validation), len(label_paths_validation))
    print("Example image path:", image_paths_train[0])

    # step 1: create Dataset objects
    train_dataset = create_dataset(image_paths_train, label_paths_train)
    validation_dataset = create_dataset(image_paths_validation, label_paths_validation)

    # step 2: create DatasetDict
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        }
    )

    # shuffle + split dataset
    dataset = dataset.shuffle(seed=1)
    dataset = dataset["train"].train_test_split(test_size=0.2)
    train_ds = dataset["train"]
    test_ds = dataset["test"]

    return train_ds, test_ds

def get_labels():
    filename = "id2label.json"   
    with open(filename, "r") as file:
        id2label = json.load(file)
    id2label = {int(k):v for k,v in id2label.items()}

    print(id2label)

    return id2label
   

def create_torch_dataset(train_ds, test_ds):
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    # TODO: NORMALIZATION?

    train_transform = A.Compose([
        # A.LongestMaxSize(max_size=1333),
        A.SmallestMaxSize(max_size=512), 
        A.RandomCrop(width=512, height=512),
        A.HorizontalFlip(p=0.5),
        # A.Normalize(mean=MEAN, std=STD),
    ])

    test_transform = A.Compose([
        A.Resize(width=512, height=512),
        # A.Normalize(mean=MEAN, std=STD),
    ])

    train_dataset = ImageSegmentationDataset(train_ds, transform=train_transform)
    test_dataset = ImageSegmentationDataset(test_ds, transform=test_transform)

    return train_dataset, test_dataset

def evaluate_model(model, test_dataloader, device, metric, id2label):
    model.eval()
    preprocessor = MaskFormerImageProcessor(ignore_index=0, reduce_labels=False, 
                                            do_resize=True, do_rescale=False, do_normalize=True)
    for idx, batch in enumerate(tqdm(test_dataloader)):
        if idx > 5:
            break

        pixel_values = batch["pixel_values"]
        
        # Forward pass
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values.to(device))

        # get original images
        original_images = batch["original_images"]
        target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
        # predict segmentation maps
        predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs,
                                                                                    target_sizes=target_sizes)

        # get ground truth segmentation maps
        ground_truth_segmentation_maps = batch["original_segmentation_maps"]

        metric.add_batch(references=ground_truth_segmentation_maps, predictions=predicted_segmentation_maps)
  
    print("Mean IoU:", metric.compute(num_labels = len(id2label), ignore_index = 0)['mean_iou'])

def main():
    train_ds, test_ds = get_train_test_ds()
    id2label = get_labels()

    #####################
    # Look at one example image
    # example = train_ds[1]
    # image = example['image']
    # label = example['binary_label']
    # np.unique(label)
    # segmentation_map = np.array(example['binary_label'])
    # plt.figure(figsize=(8, 8))
    # plt.imshow(segmentation_map, cmap="gray")  # Display as a grayscale image
    # plt.title("Binary Segmentation Map")
    # plt.axis("off")
    # plt.show()
    # print(np.unique(segmentation_map))
    # # Create an overlay with color for the segmentation map
    # image_np = np.array(image)
    # overlay = np.zeros_like(image_np, dtype=np.uint8)  # Same size as the image_np
    # overlay[segmentation_map == 1] = [0, 255, 0]  # Color the mask areas (e.g., red)

    # # Blend the overlay with the original image_np
    # alpha = 0.5  # Transparency factor
    # blended = image_np * (1 - alpha) + overlay * alpha
    # blended = blended.astype(np.uint8)

    # # Plot the original image_np and overlay
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.imshow(image_np)  # Display the image
    # ax.imshow(segmentation_map, cmap='Blues', alpha=0.5)  # Overlay the mask with transparency
    # ax.axis("off")
    # ax.set_title("Image with Mask Overlay")
    # plt.show()
    ####################

    train_dataset, test_dataset = create_torch_dataset(train_ds, test_ds)

    # img, segmentation_map, _, _ = train_dataset[0]
    # print(img.shape)
    # print(segmentation_map.shape)

    # image_np = np.array(np.transpose(img, (1, 2, 0)) )
    # overlay = np.zeros_like(image_np, dtype=np.uint8)  # Same size as the image_np

    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.imshow(image_np)  # Display the image
    # ax.imshow(segmentation_map, cmap='Blues', alpha=0.5)  # Overlay the mask with transparency
    # ax.axis("off")
    # ax.set_title("Image with Mask Overlay")
    # plt.show()

    # Create a preprocessor
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Verify one batch
    batch = next(iter(train_dataloader))
    for k,v in batch.items():
        if isinstance(v, torch.Tensor):
            print(k,v.shape)
        else:
            print(k,v[0].shape)

    # visualize_mask(labels, "stop_sign", batch)
 
    # Replace the head of the pre-trained model
    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade",
                                                              id2label=id2label,
                                                              ignore_mismatched_sizes=True)

    print("Passing one batch...")
    # Compute initial loss
    outputs = model(batch["pixel_values"],
                    class_labels=batch["class_labels"],
                    mask_labels=batch["mask_labels"])
    print("Loss of one iteratin:", outputs.loss)

 
    # Train the model
    print("Training begins...")
    metric = evaluate.load("mean_iou")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    running_loss = 0.0
    num_samples = 0
    for epoch in range(100):
        print("Epoch:", epoch)
        model.train()
        for idx, batch in enumerate(tqdm(train_dataloader)):
            # Reset the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )

            # Backward propagation
            loss = outputs.loss
            loss.backward()

            batch_size = batch["pixel_values"].size(0)
            running_loss += loss.item()
            num_samples += batch_size

            if idx % 100 == 0:
                print("Loss:", running_loss/num_samples)

            # Optimization
            optimizer.step()
            if idx == 5: break

        break

    save_directory = "/trained_model"
    breakpoint()
    model.save_pretrained(save_directory)
    evaluate_model(model, test_dataloader, device, metric, id2label)

if __name__ == "__main__":
    main()

'''
def inference():
    # let's take the first test batch
    batch = next(iter(test_dataloader))
    for k,v in batch.items():
    if isinstance(v, torch.Tensor):
        print(k,v.shape)
    else:
        print(k,len(v))


    # forward pass
    with torch.no_grad():
    outputs = model(batch["pixel_values"].to(device))


    original_images = batch["original_images"]
    target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
    # predict segmentation maps
    predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)


    image = batch["original_images"][0]
    Image.fromarray(image)


    import numpy as np
    import matplotlib.pyplot as plt

    segmentation_map = predicted_segmentation_maps[0].cpu().numpy()

    color_segmentation_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label, color in enumerate(palette):
        color_segmentation_map[segmentation_map == label, :] = color
    # Convert to BGR
    ground_truth_color_seg = color_segmentation_map[..., ::-1]

    img = image * 0.5 + ground_truth_color_seg * 0.5
    img = img.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.show()

    
    # Compare to the ground truth:


    import numpy as np
    import matplotlib.pyplot as plt

    segmentation_map = batch["original_segmentation_maps"][0]

    color_segmentation_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label, color in enumerate(palette):
        color_segmentation_map[segmentation_map == label, :] = color
    # Convert to BGR
    ground_truth_color_seg = color_segmentation_map[..., ::-1]

    img = image * 0.5 + ground_truth_color_seg * 0.5
    img = img.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.show()
'''