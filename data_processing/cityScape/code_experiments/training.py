import os
import torch
from detectron2 import engine, config
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader

# Set up logging
setup_logger()

# Configuration setup
def setup_cfg():
    cfg = get_cfg()
    # Load the configuration file from the YAML
    cfg.merge_from_file("config.yaml")
    cfg.merge_from_list([])
    
    # Model weight path and training setup
    cfg.MODEL.WEIGHTS = "path_to_your_model_weights.pth"  # Use the model weights you saved from the previous step
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # 2 classes: background and traffic sign
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.STEPS = (3000,)  # Learning rate decay steps
    cfg.SOLVER.MAX_ITER = 4000  # Maximum number of iterations
    cfg.INPUT.MIN_SIZE_TRAIN = (800, 832, 864, 896, 928, 960, 992, 1024, 1024)
    cfg.INPUT.MAX_SIZE_TRAIN = 2048
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 2048
    cfg.OUTPUT_DIR = "./output"  # Where to store checkpoints
    return cfg

# Training the model
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg)

if __name__ == "__main__":
    # Set up config
    cfg = setup_cfg()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    
    # Initialize the model
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)  # Load checkpoint if resuming, else start fresh
    trainer.train()  # Train the model
