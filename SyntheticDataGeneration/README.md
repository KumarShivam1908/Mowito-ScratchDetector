# Synthetic Data Generation - DCGAN Pipeline

A Deep Convolutional GAN (DCGAN) implementation for generating synthetic scratch images for anomaly detection and data augmentation.

## Overview

This module uses a Generative Adversarial Network (GAN) architecture to generate synthetic images of scratches. The GAN is trained on "good" (non-defective) samples from the anomaly detection dataset and can be used to:
- Generate additional training data for scratch detection models
- Perform data augmentation
- Create synthetic test cases
- Balance class distributions

## Project Structure

```
SyntheticDataGeneration/
├── src/
│   ├── DCGAN.py                # Main DCGAN implementation
│   └── __init__.py
├── datasets/
│   └── anomaly_detection_dataset/
│       ├── good/               # Good samples for training
│       └── bad/                # Bad samples (for reference)
├── models/                     # Trained GAN models
│   ├── generator_model.pth
│   ├── discriminator_model.pth
│   └── network-snapshot-001000.pkl
├── notebooks/
│   ├── sdg-gan-inference.ipynb  # GAN inference notebook
│   └── sdg-scratch.ipynb        # Training and experimentation
└── README.md
```

## Model Architecture

### Generator
The generator network transforms random noise (latent vector) into realistic images:
- **Input**: 100-dimensional noise vector
- **Architecture**: 5 transposed convolutional layers with batch normalization
- **Output**: 128x128 grayscale images
- **Activation**: Tanh (outputs in range [-1, 1])

**Layer progression**:
- (100, 1, 1) → (512, 4, 4) → (256, 8, 8) → (128, 16, 16) → (64, 32, 32) → (32, 64, 64) → (1, 128, 128)

### Discriminator
The discriminator network classifies images as real or fake:
- **Input**: 128x128 grayscale images
- **Architecture**: 5 convolutional layers with batch normalization
- **Output**: Single probability value
- **Activation**: Sigmoid

**Layer progression**:
- (1, 128, 128) → (32, 64, 64) → (64, 32, 32) → (128, 16, 16) → (256, 8, 8) → (512, 4, 4) → (1, 1, 1)

## Installation

Make sure you have PyTorch and torchvision installed:

```bash
pip install torch torchvision pillow
```

## Configuration

The DCGAN implementation uses the following default hyperparameters (configurable in `src/DCGAN.py`):

```python
BATCH_SIZE = 64          # Training batch size
IMAGE_SIZE = 128         # Output image size (128x128)
CHANNELS = 1             # 1 for grayscale, 3 for RGB
Z_DIM = 100              # Latent vector dimension
LR = 0.0002              # Learning rate for Adam optimizer
BETA1 = 0.5              # Beta1 for Adam optimizer
EPOCHS = 250             # Number of training epochs
```

**Paths** are automatically configured relative to project root:
- Dataset: `datasets/anomaly_detection_dataset/`
- Model saves: `models/`

## Usage

### Training

To train the DCGAN model on your dataset:

```python
from src.DCGAN import main

# Run training
main()
```

Or directly run the script:

```bash
cd SyntheticDataGeneration
python src/DCGAN.py
```

The training will:
1. Load images from `datasets/anomaly_detection_dataset/good/`
2. Train the generator and discriminator
3. Print training statistics every 50 iterations
4. (Optional) Save generated images every 10 epochs

### Generating Synthetic Images

After training, you can generate new images:

```python
import torch
from src.DCGAN import Generator, Z_DIM, CHANNELS, DEVICE, MODEL_SAVE_DIR

# Load trained generator
netG = Generator(Z_DIM, CHANNELS).to(DEVICE)
netG.load_state_dict(torch.load(MODEL_SAVE_DIR / 'generator_model.pth'))
netG.eval()

# Generate images
with torch.no_grad():
    noise = torch.randn(64, Z_DIM, 1, 1, device=DEVICE)
    fake_images = netG(noise)
    
# Save or process generated images
import torchvision.utils as vutils
vutils.save_image(fake_images, 'generated_samples.png', normalize=True)
```

### Using Notebooks

For interactive experimentation, use the provided notebooks:

**sdg-gan-inference.ipynb**
- Load trained models
- Generate synthetic images
- Visualize results
- Export generated data

**sdg-scratch.ipynb**
- Exploratory data analysis
- Training experiments
- Hyperparameter tuning
- Quality assessment

## Dataset Structure

The dataset should be organized as follows:

```
datasets/anomaly_detection_dataset/
├── good/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── bad/
    ├── image1.png
    ├── image2.png
    └── ...
```

The DCGAN trains on images from the `good/` subdirectory.

## Training Details

### Loss Function
- **Binary Cross-Entropy (BCE)** loss for both generator and discriminator
- Discriminator objective: maximize log(D(x)) + log(1 - D(G(z)))
- Generator objective: maximize log(D(G(z)))

### Weight Initialization
- Convolutional layers: Normal(0.0, 0.02)
- Batch normalization: Normal(1.0, 0.02) for weights, 0 for biases

### Training Loop
1. **Update Discriminator**:
   - Train on real images (label = 1)
   - Train on fake images (label = 0)
   - Compute combined loss
   
2. **Update Generator**:
   - Generate fake images
   - Train to fool discriminator (label = 1)

### Monitoring Training
Training statistics printed every 50 iterations:
- `Loss_D`: Discriminator loss
- `Loss_G`: Generator loss
- `D(x)`: Discriminator output on real images (should be close to 1)
- `D(G(z))`: Discriminator output on fake images (first value before G update, second after)

## Model Outputs

Trained models are saved in the `models/` directory:
- `generator_model.pth`: Generator state dict
- `discriminator_model.pth`: Discriminator state dict
- `network-snapshot-001000.pkl`: Complete network snapshot

## Tips for Best Results

1. **Data Quality**: Ensure your training images are clean and consistent
2. **Batch Size**: Larger batch sizes (64-128) generally produce more stable training
3. **Training Time**: GANs typically need 100-300 epochs to produce good results
4. **Mode Collapse**: If generator produces similar images, try:
   - Reducing learning rate
   - Increasing discriminator training iterations
   - Adding noise to discriminator inputs
   
5. **Evaluation**: Periodically save and visually inspect generated images

## Common Issues

**No images found error**:
- Verify dataset path in configuration
- Ensure images are in `datasets/anomaly_detection_dataset/good/`

**Training instability**:
- Reduce learning rate (try 0.0001)
- Increase discriminator capacity
- Use gradient clipping

**Poor image quality**:
- Train for more epochs
- Adjust architecture depth
- Tune hyperparameters

## Examples

### Quick Start

```bash
# Navigate to directory
cd SyntheticDataGeneration

# Run training
python src/DCGAN.py
```

### Custom Training

Modify hyperparameters in `src/DCGAN.py`:

```python
BATCH_SIZE = 128        # Larger batch
EPOCHS = 500            # More training
LR = 0.0001             # Lower learning rate
```

### Batch Generation

```python
# Generate 1000 synthetic images
num_samples = 1000
batch_size = 64

for i in range(0, num_samples, batch_size):
    noise = torch.randn(batch_size, Z_DIM, 1, 1, device=DEVICE)
    with torch.no_grad():
        fake = netG(noise).detach().cpu()
    # Save to disk
    for j, img in enumerate(fake):
        vutils.save_image(img, f'synthetic_{i+j}.png', normalize=True)
```

## Integration with Scratch Detection

Generated synthetic images can be used to:
1. **Augment training data** for the scratch detection models
2. **Balance dataset** if one class is underrepresented
3. **Test robustness** of detection models
4. **Create edge cases** for model validation

## References

- DCGAN paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- PyTorch DCGAN tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

## License

This project is for educational and research purposes.
