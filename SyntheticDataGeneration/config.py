"""
Configuration for Synthetic Data Generation
Models: CycleGAN and StyleGAN
"""
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "generated_images"

# Model paths
CYCLEGAN_MODEL_PATH = MODELS_DIR / "cyclegan_generator_model.pth"
# Note: StyleGAN model is in torch_utils subdirectory
STYLEGAN_MODEL_PATH = MODELS_DIR / "torch_utils/stylegan_network-snapshot-001000.pkl"

# Generation settings
IMAGE_SIZE = 128
CHANNELS = 1  # Grayscale
Z_DIM = 100
DEFAULT_NUM_IMAGES = 500
DEFAULT_BATCH_SIZE = 64
