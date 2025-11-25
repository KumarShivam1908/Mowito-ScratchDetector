"""
Model Conversion Script
Converts PyTorch models to ONNX and TensorRT formats for benchmarking.

USAGE EXAMPLES:
---------------

1. Convert PyTorch to both ONNX and TensorRT (saves to ./models folder by default):
   python convert_models.py --pytorch-model ../ScratchDetection/models/resnet50_binary.pth --model-name resnet50_binary --precision fp16

2. Convert with custom output directory:
   python convert_models.py --pytorch-model ../ScratchDetection/models/resnet50_binary.pth --output-dir ../ScratchDetection/models --model-name resnet50_binary --precision fp16

3. Convert only to ONNX (skip TensorRT):
   python convert_models.py --pytorch-model ../ScratchDetection/models/resnet50_binary.pth --model-name resnet50_binary --skip-tensorrt

4. Convert existing ONNX to TensorRT (skip ONNX conversion):
   python convert_models.py --pytorch-model ../ScratchDetection/models/resnet50_binary.pth --model-name resnet50_binary --skip-onnx

5. Use FP32 precision for TensorRT:
   python convert_models.py --pytorch-model ../ScratchDetection/models/resnet50_binary.pth --model-name resnet50_binary --precision fp32
"""

import torch
import torch.nn as nn
import timm
import onnx
import tensorrt as trt
from pathlib import Path
import argparse


def convert_pytorch_to_onnx(pytorch_model_path: str, onnx_output_path: str, model_arch: str = None, input_size=(1, 3, 224, 224)):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        pytorch_model_path: Path to PyTorch .pth model
        onnx_output_path: Path where ONNX model will be saved
        model_arch: Model architecture (e.g., 'resnet50', 'resnet18'). If None, auto-detect from checkpoint
        input_size: Input tensor size (default: (1, 3, 224, 224))
    """
    print(f"Converting PyTorch model to ONNX...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint to inspect
    checkpoint = torch.load(pytorch_model_path, map_location=device)
    
    # Auto-detect model architecture if not provided
    if model_arch is None:
        # Detect based on layer structure in checkpoint
        if 'layer1.2.conv3.weight' in checkpoint:
            # Has bottleneck blocks (ResNet50/101/152)
            if 'layer3.5.conv1.weight' in checkpoint:
                model_arch = 'resnet50'
            elif 'layer3.22.conv1.weight' in checkpoint:
                model_arch = 'resnet101'
            else:
                model_arch = 'resnet50'  # default to resnet50
        else:
            # Basic blocks (ResNet18/34)
            model_arch = 'resnet18'
        print(f"  Auto-detected architecture: {model_arch}")
    
    # Detect number of output classes from fc layer
    fc_weight_shape = checkpoint.get('fc.weight', checkpoint.get('head.weight'))
    if fc_weight_shape is not None:
        num_classes = fc_weight_shape.shape[0]
    else:
        num_classes = 2  # default
    print(f"  Detected output classes: {num_classes}")
    
    # Create model with correct architecture
    model = timm.create_model(model_arch, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_size).to(device)
    
    # Export to ONNX with fixed batch size for TensorRT compatibility
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
        # Note: No dynamic_axes - using fixed batch size for TensorRT compatibility
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"✓ ONNX model saved to: {onnx_output_path}")
    return model_arch, num_classes


def convert_onnx_to_tensorrt(onnx_model_path: str, trt_output_path: str, precision='fp16'):
    """
    Convert ONNX model to TensorRT engine.
    
    Args:
        onnx_model_path: Path to ONNX model
        trt_output_path: Path where TensorRT engine will be saved
        precision: Precision mode ('fp32', 'fp16', or 'int8')
    """
    print(f"Converting ONNX model to TensorRT ({precision})...")
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    with open(onnx_model_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return
    
    # Build engine configuration
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # Set precision
    if precision == 'fp16' and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  Using FP16 precision")
    elif precision == 'int8' and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("  Using INT8 precision")
    else:
        print("  Using FP32 precision")
    
    # Build engine
    print("  Building TensorRT engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Failed to build TensorRT engine")
        return
    
    # Save engine
    with open(trt_output_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"✓ TensorRT engine saved to: {trt_output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch models to ONNX and TensorRT')
    parser.add_argument('--pytorch-model', type=str, required=True,
                        help='Path to PyTorch model (.pth)')
    parser.add_argument('--output-dir', type=str, default='./models',
                        help='Directory to save converted models (default: ./models)')
    parser.add_argument('--model-name', type=str, default='model',
                        help='Base name for output files (default: model)')
    parser.add_argument('--precision', type=str, default='fp16', 
                        choices=['fp32', 'fp16', 'int8'],
                        help='TensorRT precision mode (default: fp16)')
    parser.add_argument('--skip-onnx', action='store_true',
                        help='Skip ONNX conversion (use existing ONNX file)')
    parser.add_argument('--skip-tensorrt', action='store_true',
                        help='Skip TensorRT conversion')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = output_dir / f"{args.model_name}.onnx"
    trt_path = output_dir / f"{args.model_name}.trt"
    
    # Convert to ONNX
    if not args.skip_onnx:
        convert_pytorch_to_onnx(args.pytorch_model, str(onnx_path))
    else:
        print(f"Skipping ONNX conversion, using: {onnx_path}")
    
    # Convert to TensorRT
    if not args.skip_tensorrt:
        if onnx_path.exists():
            convert_onnx_to_tensorrt(str(onnx_path), str(trt_path), args.precision)
        else:
            print(f"ERROR: ONNX file not found: {onnx_path}")
    else:
        print("Skipping TensorRT conversion")
    
    print("\n" + "="*60)
    print("Conversion Complete!")
    print("="*60)
    print(f"PyTorch model:  {args.pytorch_model}")
    if not args.skip_onnx or onnx_path.exists():
        print(f"ONNX model:     {onnx_path}")
    if not args.skip_tensorrt and trt_path.exists():
        print(f"TensorRT model: {trt_path}")


if __name__ == '__main__':
    main()
