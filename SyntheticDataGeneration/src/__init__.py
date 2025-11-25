"""
Synthetic Data Generation Package
Provides CycleGAN and StyleGAN2 image generators
"""
from .cyclegan import CycleGANGenerator, CycleGANImageGenerator
from .stylegan import StyleGANImageGenerator

__all__ = [
    'CycleGANGenerator',
    'CycleGANImageGenerator',
    'StyleGANImageGenerator',
]
