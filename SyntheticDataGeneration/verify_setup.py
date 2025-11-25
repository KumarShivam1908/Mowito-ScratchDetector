"""
Verify Synthetic Data Generation Setup

USAGE:
------
python verify_setup.py

Checks CycleGAN and StyleGAN model availability and dependencies.
"""
import sys
from pathlib import Path

print("="*80)
print("SYNTHETIC DATA GENERATION - SETUP VERIFICATION")
print("="*80)

# 1. Python & PyTorch
print(f"\n1. Python Version: {sys.version}")

print(f"\n2. PyTorch:")
try:
    import torch
    print(f"   [OK] PyTorch {torch.__version__}")
    print(f"   [OK] CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   [OK] CUDA Version: {torch.version.cuda}")
        print(f"   [OK] GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("   [ERROR] PyTorch not installed")

# 2. Dependencies
print(f"\n3. Dependencies:")
for dep in ['torchvision', 'PIL', 'numpy', 'tqdm']:
    try:
        __import__(dep)
        print(f"   [OK] {dep}")
    except ImportError:
        print(f"   [MISSING] {dep}")

# 3. Project Structure
print(f"\n4. Project Structure:")
for dir_name in ['models', 'generated_images']:
    status = "[OK]" if Path(dir_name).exists() else "[MISSING]"
    print(f"   {status} {dir_name}/")

for file_name in ['inference.py', 'config.py', 'QUICK_REFERENCE.md']:
    status = "[OK]" if Path(file_name).exists() else "[MISSING]"
    print(f"   {status} {file_name}")

# 4. Model Files
print(f"\n5. Model Files:")
models = {
    'CycleGAN Generator': 'models/cyclegan_generator_model.pth',
    'StyleGAN2': 'models/torch_utils/stylegan_network-snapshot-001000.pkl'
}

for name, path in models.items():
    file_path = Path(path)
    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"   [OK] {name}: {size_mb:.1f} MB")
    else:
        print(f"   [MISSING] {name}")

print("\n" + "="*80)
print("SETUP VERIFICATION COMPLETE")
print("="*80)
print("\nNext Steps:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Ensure model files are in models/ directory")
print("3. Generate images: python inference.py --num-images 100")
print("4. See QUICK_REFERENCE.md for examples")
print("="*80 + "\n")
