"""
CycleGAN Image Generator Module
Handles loading and generating images using trained CycleGAN model.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image


class CycleGANGenerator(nn.Module):
    """
    CycleGAN Generator Network
    Generates 128x128 grayscale images from random noise
    """
    def __init__(self, z_dim=100, channels=1):
        super(CycleGANGenerator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 4x4

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 8x8

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 16x16

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 32x32
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 64x64

            nn.ConvTranspose2d(32, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 128x128 (Range: -1 to 1)
        )

    def forward(self, x):
        return self.net(x)


class CycleGANImageGenerator:
    """
    CycleGAN Image Generator Wrapper
    Provides easy interface for generating synthetic images
    """
    
    def __init__(self, model_path, z_dim=100, channels=1, device='cuda'):
        """
        Initialize CycleGAN generator
        
        Args:
            model_path: Path to generator model weights (.pth file)
            z_dim: Latent vector dimension (default: 100)
            channels: Number of image channels (1 for grayscale, 3 for RGB)
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.z_dim = z_dim
        self.channels = channels
        
        # Check if model exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"CycleGAN model not found at {model_path}")
        
        # Load model
        self.generator = CycleGANGenerator(z_dim, channels).to(self.device)
        self.generator.load_state_dict(torch.load(model_path, map_location=self.device))
        self.generator.eval()
        
        print(f"✓ CycleGAN model loaded from {model_path}")
    
    def generate(self, num_images, seed=None):
        """
        Generate synthetic images
        
        Args:
            num_images: Number of images to generate
            seed: Random seed for reproducibility (optional)
            
        Returns:
            List of PIL Images
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        images = []
        
        with torch.no_grad():
            # Generate noise
            noise = torch.randn(num_images, self.z_dim, 1, 1, device=self.device)
            
            # Generate images
            fake_images = self.generator(noise)
            
            # Convert to PIL Images
            for img_tensor in fake_images:
                # Denormalize from [-1, 1] to [0, 1]
                img_tensor = (img_tensor + 1) / 2.0
                img_tensor = torch.clamp(img_tensor, 0, 1)
                
                # Convert to numpy
                img_np = img_tensor.cpu().numpy()
                
                # Convert to PIL based on channels
                if self.channels == 1:
                    img_np = (img_np[0] * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_np, mode='L')
                else:
                    img_np = (np.transpose(img_np, (1, 2, 0)) * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_np, mode='RGB')
                
                images.append(img_pil)
        
        return images
    
    def generate_batch(self, num_images, batch_size=64, seed=None):
        """
        Generate images in batches (more memory efficient)
        
        Args:
            num_images: Total number of images to generate
            batch_size: Number of images per batch
            seed: Random seed for reproducibility
            
        Yields:
            List of PIL Images for each batch
        """
        total_generated = 0
        
        while total_generated < num_images:
            current_batch = min(batch_size, num_images - total_generated)
            batch_seed = seed + total_generated if seed is not None else None
            
            batch_images = self.generate(current_batch, seed=batch_seed)
            yield batch_images
            
            total_generated += current_batch
