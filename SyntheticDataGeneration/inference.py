"""
Synthetic Image Generation - Inference Script
Generate synthetic images using CycleGAN and StyleGAN models.

USAGE:
------
# Generate 500 images using CycleGAN (default)
python inference.py --model cyclegan --num-images 500

# Generate 500 images using StyleGAN
python inference.py --model stylegan --num-images 500

# Generate images using both models
python inference.py --model both --num-images 500

# Generate with custom output directory
python inference.py --model cyclegan --num-images 1000 --output-dir ./my_images

# Generate with specific seed for reproducibility
python inference.py --model cyclegan --num-images 100 --seed 42

# Generate on CPU
python inference.py --model cyclegan --num-images 200 --device cpu

# StyleGAN with truncation for higher quality
python inference.py --model stylegan --num-images 100 --truncation 0.7

AVAILABLE MODELS:
-----------------
- cyclegan    : CycleGAN (fast generation)
- stylegan    : StyleGAN2 (high quality)
- both        : Generate using both models

KEY ARGUMENTS:
--------------
--model         : Model to use (cyclegan/stylegan/both, default: cyclegan)
--num-images    : Number of images to generate (default: 500)
--output-dir    : Output directory (default: generated_images/)
--batch-size    : Batch size for CycleGAN generation (default: 64)
--seed          : Random seed for reproducibility
--device        : Device to use (cuda/cpu, default: cuda)
--truncation    : Truncation psi for StyleGAN (0.5-1.0, default: 1.0)

OUTPUT:
-------
generated_images/
├── cyclegan/
│   ├── synthetic_00000.png
│   ├── synthetic_00001.png
│   └── ...
└── stylegan/
    ├── synthetic_00000.png
    ├── synthetic_00001.png
    └── ...
"""
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import time

from config import (
    CYCLEGAN_MODEL_PATH,
    STYLEGAN_MODEL_PATH,
    OUTPUT_DIR,
    IMAGE_SIZE,
    CHANNELS,
    Z_DIM,
    DEFAULT_NUM_IMAGES,
    DEFAULT_BATCH_SIZE
)
from src.cyclegan import CycleGANImageGenerator
from src.stylegan import StyleGANImageGenerator


def save_images(images, output_dir, prefix='synthetic', start_idx=0):
    """
    Save generated images to disk
    
    Args:
        images: List of PIL Images
        output_dir: Directory to save images
        prefix: Filename prefix
        start_idx: Starting index for numbering
        
    Returns:
        Number of images saved
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, img in enumerate(images):
        filename = f"{prefix}_{start_idx + i:05d}.png"
        filepath = output_dir / filename
        img.save(filepath)
    
    return len(images)


def generate_cyclegan_images(num_images, output_dir, batch_size=64, seed=None, device='cuda'):
    """Generate images using CycleGAN"""
    print(f"\n{'='*80}")
    print("CYCLEGAN IMAGE GENERATION")
    print(f"{'='*80}\n")
    
    # Initialize generator
    try:
        generator = CycleGANImageGenerator(
            model_path=CYCLEGAN_MODEL_PATH,
            z_dim=Z_DIM,
            channels=CHANNELS,
            device=device
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please ensure cyclegan_generator_model.pth is in the models/ directory")
        return 0
    
    # Create output directory
    cyclegan_output_dir = Path(output_dir) / "cyclegan"
    cyclegan_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_images} images...")
    print(f"Output: {cyclegan_output_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device.upper()}\n")
    
    # Generate images in batches
    total_generated = 0
    start_time = time.time()
    
    with tqdm(total=num_images, desc="Generating") as pbar:
        for batch_images in generator.generate_batch(num_images, batch_size, seed):
            # Save batch
            count = save_images(batch_images, cyclegan_output_dir, 
                              prefix='synthetic', start_idx=total_generated)
            total_generated += count
            pbar.update(count)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✓ Generation Complete!")
    print(f"  Images: {total_generated}")
    print(f"  Time: {elapsed_time:.2f}s")
    print(f"  Speed: {total_generated/elapsed_time:.2f} images/sec")
    print(f"  Location: {cyclegan_output_dir}\n")
    
    return total_generated


def generate_stylegan_images(num_images, output_dir, seed=None, device='cuda', truncation_psi=1.0):
    """Generate images using StyleGAN"""
    print(f"\n{'='*80}")
    print("STYLEGAN IMAGE GENERATION")
    print(f"{'='*80}\n")
    
    # Initialize generator
    try:
        generator = StyleGANImageGenerator(
            model_path=STYLEGAN_MODEL_PATH,
            device=device
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please ensure stylegan_network-snapshot-001000.pkl is in the models/ directory")
        return 0
    except Exception as e:
        print(f"ERROR loading StyleGAN: {e}")
        return 0
    
    # Create output directory
    stylegan_output_dir = Path(output_dir) / "stylegan"
    stylegan_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_images} images...")
    print(f"Output: {stylegan_output_dir}")
    print(f"Device: {device.upper()}")
    print(f"Truncation: {truncation_psi}\n")
    
    # Generate images
    total_generated = 0
    start_time = time.time()
    
    # Generate in small batches for memory efficiency
    batch_size = 10
    
    with tqdm(total=num_images, desc="Generating") as pbar:
        while total_generated < num_images:
            current_batch = min(batch_size, num_images - total_generated)
            batch_seed = seed + total_generated if seed is not None else None
            
            # Generate batch
            images = generator.generate(current_batch, seed=batch_seed, 
                                      truncation_psi=truncation_psi)
            
            # Save batch
            count = save_images(images, stylegan_output_dir,
                              prefix='synthetic', start_idx=total_generated)
            total_generated += count
            pbar.update(count)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✓ Generation Complete!")
    print(f"  Images: {total_generated}")
    print(f"  Time: {elapsed_time:.2f}s")
    print(f"  Speed: {total_generated/elapsed_time:.2f} images/sec")
    print(f"  Location: {stylegan_output_dir}\n")
    
    return total_generated


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic images using CycleGAN or StyleGAN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 500 CycleGAN images
  python inference.py --model cyclegan --num-images 500
  
  # Generate 500 high-quality StyleGAN images
  python inference.py --model stylegan --num-images 500 --truncation 0.7
  
  # Generate from both models
  python inference.py --model both --num-images 500
  
  # Reproducible generation
  python inference.py --model cyclegan --num-images 100 --seed 42
        """
    )
    
    parser.add_argument('--model', type=str, default='cyclegan',
                       choices=['cyclegan', 'stylegan', 'both'],
                       help='Model to use (default: cyclegan)')
    parser.add_argument('--num-images', type=int, default=DEFAULT_NUM_IMAGES,
                       help=f'Number of images to generate (default: {DEFAULT_NUM_IMAGES})')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR),
                       help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                       help=f'Batch size for CycleGAN (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--truncation', type=float, default=1.0,
                       help='Truncation psi for StyleGAN (0.5-1.0, default: 1.0)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("SYNTHETIC IMAGE GENERATION")
    print(f"{'='*80}\n")
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU\n")
        device = 'cpu'
    
    print(f"Configuration:")
    print(f"  Model: {args.model.upper()}")
    print(f"  Number of images: {args.num_images}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Device: {device.upper()}")
    if args.seed:
        print(f"  Seed: {args.seed}")
    if args.model in ['stylegan', 'both']:
        print(f"  Truncation (StyleGAN): {args.truncation}")
    
    # Generate images
    total_images = 0
    
    if args.model in ['cyclegan', 'both']:
        count = generate_cyclegan_images(
            num_images=args.num_images,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device
        )
        total_images += count
    
    if args.model in ['stylegan', 'both']:
        count = generate_stylegan_images(
            num_images=args.num_images,
            output_dir=args.output_dir,
            seed=args.seed,
            device=device,
            truncation_psi=args.truncation
        )
        total_images += count
    
    # Summary
    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total images generated: {total_images}")
    print(f"Output location: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
