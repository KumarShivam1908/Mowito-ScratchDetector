"""
Quick setup verification and demo script for the restructured ScratchDetection codebase.

PURPOSE:
--------
Verifies that your development environment is properly set up before training or
running inference. Checks Python version, dependencies, project structure, model
weights, datasets, and module imports.

USAGE:
------
# Run setup verification
python verify_setup.py

# This script takes no arguments - just run it!

WHAT IT CHECKS:
---------------
1. Python Version - Ensures compatible Python version
2. PyTorch Installation - Verifies PyTorch and CUDA availability
3. Project Structure - Checks all required directories and files
4. Src Subdirectories - Validates modular structure
5. Model Weights - Lists available trained model files
6. Datasets - Shows available dataset directories
7. Import Tests - Tests all module imports
8. Available Models - Lists all configured models

OUTPUT:
-------
- [OK] markers for properly set up components
- [MISSING] warnings for missing components
- [ERROR] messages for import failures
- Summary of next steps to get started

WHEN TO RUN:
------------
- After cloning the repository
- Before training models
- After restructuring the codebase
- When troubleshooting setup issues
- After installing new dependencies

EXPECTED OUTPUT:
----------------
All items should show [OK]. If you see [MISSING] or [ERROR], check:
- Install requirements: pip install -r requirements.txt
- Verify project structure matches documentation
- Check config/config.py for correct paths
- Ensure model weights are in models/ directory
"""
import sys
import torch
from pathlib import Path

print("="*80)
print("SCRATCH DETECTION - SETUP VERIFICATION")
print("="*80)

# Check Python version
print(f"\n1. Python Version: {sys.version}")

# Check PyTorch installation
print(f"\n2. PyTorch Version: {torch.__version__}")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU Device: {torch.cuda.get_device_name(0)}")

# Check project structure
print(f"\n3. Project Structure:")
required_dirs = ['config', 'src', 'models', 'datasets', 'results', 'notebooks']
required_files = ['train.py', 'inference.py', 'simple_inference.py', 'compare.py', 'pipeline.py', 'requirements.txt', 'README.md']

for dir_name in required_dirs:
    dir_path = Path(dir_name)
    status = "[OK]" if dir_path.exists() else "[MISSING]"
    print(f"   {status} {dir_name}/")

for file_name in required_files:
    file_path = Path(file_name)
    status = "[OK]" if file_path.exists() else "[MISSING]"
    print(f"   {status} {file_name}")

# Check src subdirectories
print(f"\n   SRC Subdirectories:")
src_subdirs = ['data', 'models', 'training', 'inference', 'evaluation']
for subdir_name in src_subdirs:
    subdir_path = Path('src') / subdir_name
    status = "[OK]" if subdir_path.exists() else "[MISSING]"
    print(f"      {status} src/{subdir_name}/")

# Check model weights
print(f"\n4. Model Weights:")
model_dir = Path('models')
if model_dir.exists():
    weight_files = list(model_dir.glob('*.pth')) + list(model_dir.glob('*.pt'))
    print(f"   Found {len(weight_files)} model weight files:")
    for weight_file in weight_files:
        size_mb = weight_file.stat().st_size / (1024 * 1024)
        print(f"   [OK] {weight_file.name} ({size_mb:.1f} MB)")
    
    if len(weight_files) == 0:
        print("   [WARNING] No model weight files found")
        print("   Expected files like: deit_binary.pth, vit_base_binary.pth, etc.")
else:
    print("   [MISSING] models directory not found")

# Check datasets
print(f"\n5. Datasets:")
dataset_dir = Path('datasets')
if dataset_dir.exists():
    dataset_subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    print(f"   Found {len(dataset_subdirs)} dataset subdirectories:")
    for dataset in dataset_subdirs:
        print(f"   [OK] {dataset.name}/")
else:
    print("   [MISSING] datasets directory not found")

# Test imports
print(f"\n6. Testing Imports:")
try:
    from config import config
    print("   [OK] config module")
except Exception as e:
    print(f"   [ERROR] config module: {e}")

try:
    from src.data.dataloader import get_dataloaders
    print("   [OK] data.dataloader module")
except Exception as e:
    print(f"   [ERROR] data.dataloader module: {e}")

try:
    from src.models.model_factory import get_model, get_all_model_names
    print("   [OK] models.model_factory module")
except Exception as e:
    print(f"   [ERROR] models.model_factory module: {e}")

try:
    from src.training.trainer import train_model
    print("   [OK] training.trainer module")
except Exception as e:
    print(f"   [ERROR] training.trainer module: {e}")

try:
    from src.inference.detector import ScratchDetector
    print("   [OK] inference.detector module")
except Exception as e:
    print(f"   [ERROR] inference.detector module: {e}")

try:
    from src.evaluation.evaluator import evaluate_model
    print("   [OK] evaluation.evaluator module")
except Exception as e:
    print(f"   [ERROR] evaluation.evaluator module: {e}")

# List available models
print(f"\n7. Available Models:")
try:
    from src.models.model_factory import get_all_model_names
    models = get_all_model_names()
    for model_name in models:
        print(f"   - {model_name}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*80)
print("SETUP VERIFICATION COMPLETE")
print("="*80)
print("\nNext Steps:")
print("1. Update config/config.py with your dataset paths")
print("2. Install dependencies: pip install -r requirements.txt")
print("3. Train models: python train.py --model all")
print("4. Run inference: python inference.py --model deit --mode dataset")
print("5. Quick inference: python simple_inference.py --image path/to/image.jpg")
print("6. Compare models: python compare.py --models all --plot")
print("7. Run full pipeline: python pipeline.py --model deit")
print("="*80 + "\n")
