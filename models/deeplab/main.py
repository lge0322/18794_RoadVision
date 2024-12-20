from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--dataset", type=str, default='sign')
    parser.add_argument("--data_root", type=str, default='datasets/data',
                        help="path to Dataset")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")
    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int,
                        default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=5e3,
                        help="epoch number (default: 5k)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=1000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=8,
                        help='batch size (default: 8)')
    parser.add_argument("--val_batch_size", type=int, default=8,
                        help='batch size for validation (default: 8)')
    parser.add_argument("--crop_size", type=int, default=513) 

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training",
                        action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy'], help="You may add different loss types here")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--vis_num_samples", type=int, default=5,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_train_val_dataset(opts):
    """ Dataset And Augmentation
    """

    train_transform = et.ExtCompose([
        # et.ExtResize([opts.crop_size, opts.crop_size]),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size),
                         pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    if opts.crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            # et.ExtResize([opts.crop_size, opts.crop_size]),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    train_dst = VOCSegmentation(
        root=opts.data_root, image_set='train', dataset=opts.dataset, transform=train_transform)
    val_dst = VOCSegmentation(
        root=opts.data_root, image_set='val', dataset=opts.dataset, transform=val_transform)

    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            
            images = images.to(device, dtype=torch.float32)

            # Apply the threshold to keep everything below 127 as 0 and above as 1
            labels = np.where(labels > 127, 1, 0).astype(np.uint8)
            labels = torch.from_numpy(labels).long() if isinstance(labels, np.ndarray) else labels

            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1,
                                                            2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(
                        target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save(
                        'results/%d_image.png' % img_id)
                    Image.fromarray(target).save(
                        'results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' %
                                img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def plot_mIoU_curve(mIoU_per_epoch, model_type):
    """
    Plot the graph of mean IoU per epoch.

    Args:
        mIoU_per_epoch (list): List of mean IoU values for each epoch.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(mIoU_per_epoch, marker='o', linestyle='-', color='b')
    plt.title(f"Mean IoU per Epoch for {model_type}")
    plt.xlabel("Epoch")
    plt.ylabel("Mean IoU")
    plt.grid(True)
    plt.savefig(f"figures/mIoU_per_epoch_{model_type}.png")  # Save the plot as an image
    plt.show()


def main():
    opts = get_argparser().parse_args()
    opts.num_classes = 2

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if not opts.crop_val:
        opts.val_batch_size = 1
    
    train_dst, val_dst = get_train_val_dataset(opts)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          ("VOC", len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](
        num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    metrics = StreamSegMetrics(opts.num_classes)

    learning_rate = opts.lr
    backbone_scaling_factor = 0.1
    optimizer = torch.optim.SGD(
        [
            {'params': model.backbone.parameters(), 'lr': learning_rate * backbone_scaling_factor},
            {'params': model.classifier.parameters(), 'lr': learning_rate},
        ],
        momentum=0.9
    )

    # Set up Learning Rate Policy
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=1000, 
                                                gamma=0.9)

    # Set up criterion
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(
        val_loader), opts.vis_num_samples, np.int32)  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[
                               0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    mIoU_per_epoch = []
  
    while True:  
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        iter = 0
       
        for images, labels in train_loader:
            # Ensuring binary labels
            labels = np.where(labels > 127, 1, 0).astype(np.uint8)
            labels = torch.from_numpy(labels).long() if isinstance(labels, np.ndarray) else labels

            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            optimizer.step()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                # Plot the graph of mean iou
                model_type = opts.model
                plot_mIoU_curve(mIoU_per_epoch, model_type)
                return

        # evaluation after each epoch
        save_ckpt('checkpoints/latest_%s_%s_os%d_lr%d_bs%d_dataset%s.pth' %
                  (opts.model, "VOC", opts.output_stride, opts.lr, opts.batch_size,
                  opts.dataset))

        print("validation...")
        torch.cuda.empty_cache()
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
            ret_samples_ids=vis_sample_id)

        print(metrics.to_str(val_score))
        mIoU_per_epoch.append(val_score['Mean IoU'])

        if val_score['Mean IoU'] > best_score:  # save best model
            best_score = val_score['Mean IoU']
            print("new best mIOU: ", best_score)
            save_ckpt('checkpoints/best_%s_%s_os%d_lr%d_bs%d_dataset%s.pth' %
                      (opts.model, "VOC", opts.output_stride, opts.lr, opts.batch_size,
                      opts.dataset))
                      

if __name__ == '__main__':
    main()
