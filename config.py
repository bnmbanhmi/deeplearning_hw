# config.py
import os

# Data paths
TRAIN_DATA_PATH = "bkai-igh-neopolyp/train"  # Update with the actual path
TRAIN_MASKS_PATH = "bkai-igh-neopolyp/train_gt"  # Update with the actual path
TEST_DATA_PATH = "bkai-igh-neopolyp/test"  # Update with the actual path
TEST_MASKS_PATH = "path/to/your/test/masks"  # Update with the actual path

# Image dimensions
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Batch size
BATCH_SIZE = 32

# Model selection
MODEL = "unet"  # Choose from: "unet", "mobilenetv2_unet", etc.

# Paths to save results
SAVE_RESULTS_PATH = "results"
SAVE_WEIGHTS_PATH = "weights"

# Create directories if they don't exist
os.makedirs(SAVE_RESULTS_PATH, exist_ok=True)
os.makedirs(SAVE_WEIGHTS_PATH, exist_ok=True)