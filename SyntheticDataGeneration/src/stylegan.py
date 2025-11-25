"""
StyleGAN2 Image Generator Module
Handles loading and generating images using trained StyleGAN2 model.
Automatically sets up the official StyleGAN2-ADA repository and applies patches.
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import pickle
import warnings
import subprocess
import urllib.request

class StyleGANImageGenerator:
    """
    StyleGAN2 Image Generator Wrapper
    Provides easy interface for generating high-quality synthetic images
    """
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize StyleGAN2 generator
        
        Args:
            model_path: Path to StyleGAN2 model (.pkl file)
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Check if model exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"StyleGAN model not found at {model_path}")
            
        # Setup StyleGAN2-ADA repository
        self._setup_repository()
        
        # Load StyleGAN2 model
        try:
            # Import dnnlib from the cloned repo
            import dnnlib
            import legacy
            
            print(f"Loading StyleGAN model from {model_path}...")
            with dnnlib.util.open_url(str(model_path)) as f:
                self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)
            
            print(f"✓ StyleGAN model loaded successfully")
            
            # Get model properties
            self.z_dim = self.G.z_dim
            self.img_resolution = self.G.img_resolution
            self.img_channels = self.G.img_channels
            
        except Exception as e:
            raise RuntimeError(f"Failed to load StyleGAN model: {e}")

    def _setup_repository(self):
        """
        Clones StyleGAN2-ADA repo and applies compatibility patches
        """
        repo_url = "https://github.com/NVlabs/stylegan2-ada-pytorch.git"
        # Clone into a subdirectory of src to keep it contained
        repo_dir = Path(__file__).parent / "stylegan2_ada_pytorch"
        
        # 1. Clone repository if needed
        if not repo_dir.exists():
            print(f"Cloning StyleGAN2-ADA repository to {repo_dir}...")
            try:
                subprocess.check_call(["git", "clone", repo_url, str(repo_dir)])
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to clone repository: {e}")
        
        # 2. Add to Python path
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
            
        # 3. Apply Patches (Critical for PyTorch compatibility)
        ops_dir = repo_dir / "torch_utils" / "ops"
        patches = {
            "grid_sample_gradfix.py": "https://raw.githubusercontent.com/NVlabs/stylegan3/main/torch_utils/ops/grid_sample_gradfix.py",
            "conv2d_gradfix.py": "https://raw.githubusercontent.com/NVlabs/stylegan3/main/torch_utils/ops/conv2d_gradfix.py"
        }
        
        print("Checking PyTorch compatibility patches...")
        for filename, url in patches.items():
            dest = ops_dir / filename
            if dest.exists():
                # Optional: Check if we need to update? For now, assume if it exists it's fine, 
                # OR overwrite to be safe as requested by user.
                # User said "Do download it while running...". Let's overwrite to ensure fix.
                pass
            
            try:
                # Always download to ensure we have the fixed version
                urllib.request.urlretrieve(url, dest)
                # print(f" - Patched: {filename}")
            except Exception as e:
                print(f" - Warning: Could not patch {filename}: {e}")

    def generate(self, num_images, seed=None, truncation_psi=1.0):
        """
        Generate synthetic images using StyleGAN2
        
        Args:
            num_images: Number of images to generate
            seed: Random seed for reproducibility (optional)
            truncation_psi: Truncation parameter for controlling diversity
            
        Returns:
            List of PIL Images
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        images = []
        
        # Generate class labels (None for unconditional)
        label = torch.zeros([1, self.G.c_dim], device=self.device)
        
        with torch.no_grad():
            for i in range(num_images):
                # Generate latent vector
                z = torch.randn(1, self.G.z_dim, device=self.device)
                
                # Generate image
                img = self.G(z, label, truncation_psi=truncation_psi)
                
                # Process image tensor
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img_np = img[0].cpu().numpy()
                
                # Convert to PIL Image
                if img_np.shape[2] == 1:
                    img_pil = Image.fromarray(img_np[:, :, 0], mode='L')
                else:
                    img_pil = Image.fromarray(img_np, mode='RGB')
                
                images.append(img_pil)
        
        return images
    
    def generate_from_latent(self, latent_vectors, truncation_psi=1.0):
        """Generate images from specific latent vectors"""
        images = []
        label = torch.zeros([1, self.G.c_dim], device=self.device)
        
        with torch.no_grad():
            for z in latent_vectors:
                z = z.unsqueeze(0).to(self.device)
                img = self.G(z, label, truncation_psi=truncation_psi)
                
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img_np = img[0].cpu().numpy()
                
                if img_np.shape[2] == 1:
                    img_pil = Image.fromarray(img_np[:, :, 0], mode='L')
                else:
                    img_pil = Image.fromarray(img_np, mode='RGB')
                
                images.append(img_pil)
        
        return images
    
    def interpolate(self, z1, z2, steps=10, truncation_psi=1.0):
        """Generate interpolation between two latent vectors"""
        alphas = np.linspace(0, 1, steps)
        latents = []
        for alpha in alphas:
            z = z1 * (1 - alpha) + z2 * alpha
            latents.append(z)
        latents = torch.stack(latents)
        return self.generate_from_latent(latents, truncation_psi)
