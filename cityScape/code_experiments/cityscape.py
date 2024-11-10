import torch
import torch.nn as nn
from torch import nn
from torch.nn import Conv2d

def clip_weights_from_pretrain_of_cityscapes(f, out_file):
    """
    Adjusts pre-trained weights for Cityscapes dataset focusing on 'background' and 'traffic sign' labels.
    """
    # Cityscapes categories for the 'background' and 'traffic sign'
    CITYSCAPES_CATEGORIES = [
        "background",
        "traffic sign",  # Assuming traffic sign is your label for traffic signs
    ]
    
    cityscapes_cats = CITYSCAPES_CATEGORIES
    cityscapes_cats_to_inds = dict(zip(cityscapes_cats, range(len(cityscapes_cats))))

    checkpoint = torch.load(f)
    m = checkpoint['model']

    weight_names = {
        "cls_score": "module.roi_heads.box.predictor.cls_score.weight", 
        "bbox_pred": "module.roi_heads.box.predictor.bbox_pred.weight", 
        "mask_fcn_logits": "module.roi_heads.mask.predictor.mask_fcn_logits.weight", 
    }
    bias_names = {
        "cls_score": "module.roi_heads.box.predictor.cls_score.bias",
        "bbox_pred": "module.roi_heads.box.predictor.bbox_pred.bias", 
        "mask_fcn_logits": "module.roi_heads.mask.predictor.mask_fcn_logits.bias",
    }
    
    # Adjust the classifier for the two categories
    representation_size = m[weight_names["cls_score"]].size(1)
    cls_score = nn.Linear(representation_size, len(cityscapes_cats))
    nn.init.normal_(cls_score.weight, std=0.01)
    nn.init.constant_(cls_score.bias, 0)

    # Adjust the bbox prediction layer for the two categories
    representation_size = m[weight_names["bbox_pred"]].size(1)
    class_agnostic = m[weight_names["bbox_pred"]].size(0) != len(cityscapes_cats) * 4
    num_bbox_reg_classes = 2 if class_agnostic else len(cityscapes_cats)
    bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)
    nn.init.normal_(bbox_pred.weight, std=0.001)
    nn.init.constant_(bbox_pred.bias, 0)

    # Adjust the mask function logits layer for the two categories
    dim_reduced = m[weight_names["mask_fcn_logits"]].size(1)
    mask_fcn_logits = Conv2d(dim_reduced, len(cityscapes_cats), 1, 1, 0)
    nn.init.constant_(mask_fcn_logits.bias, 0)
    nn.init.kaiming_normal_(
        mask_fcn_logits.weight, mode="fan_out", nonlinearity="relu"
    )
    
    # Copy weights from the pre-trained model to the Cityscapes-specific layers
    def _copy_weight(src_weight, dst_weight):
        for ix, cat in enumerate(cityscapes_cats):
            dst_weight[ix] = src_weight[ix]
        return dst_weight

    def _copy_bias(src_bias, dst_bias, class_agnostic=False):
        if class_agnostic:
            return dst_bias
        return _copy_weight(src_bias, dst_bias)

    m[weight_names["cls_score"]] = _copy_weight(
        m[weight_names["cls_score"]], cls_score.weight
    )
    m[weight_names["bbox_pred"]] = _copy_weight(
        m[weight_names["bbox_pred"]], bbox_pred.weight
    )
    m[weight_names["mask_fcn_logits"]] = _copy_weight(
        m[weight_names["mask_fcn_logits"]], mask_fcn_logits.weight
    )

    m[bias_names["cls_score"]] = _copy_bias(
        m[bias_names["cls_score"]], cls_score.bias
    )
    m[bias_names["bbox_pred"]] = _copy_bias(
        m[bias_names["bbox_pred"]], bbox_pred.bias, class_agnostic
    )
    m[bias_names["mask_fcn_logits"]] = _copy_bias(
        m[bias_names["mask_fcn_logits"]], mask_fcn_logits.bias
    )

    print(f"f: {f}\nout_file: {out_file}")
    torch.save(m, out_file)
