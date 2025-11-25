# Model Inference Benchmark

This script benchmarks PyTorch, ONNX, and TensorRT models for inference performance.

## Features

- Benchmarks all three model formats in a single run
- Runs inference on a specified number of images (default: 100)
- Displays detailed statistics (avg, median, min, max, std, FPS)
- No results saving - only performance metrics
- CUDA acceleration support

## Requirements

```bash
pip install torch torchvision timm onnxruntime tensorrt pycuda opencv-python pillow tqdm numpy
```

## Model Conversion (Optional)

If you only have a PyTorch model and need to create ONNX and TensorRT versions:

```bash
python convert_models.py \
    --pytorch-model "path/to/model.pth" \
    --output-dir "output/directory" \
    --model-name "model_name" \
    --precision fp16
```

This will create:
- `model_name.onnx` - ONNX version
- `model_name.trt` - TensorRT engine

## Usage

```bash
python inference_benchmark.py \
    --test-dir "path/to/test/images" \
    --pytorch-model "path/to/model.pth" \
    --onnx-model "path/to/model.onnx" \
    --tensorrt-model "path/to/model.trt" \
    --num-images 100
```

### Example

```bash
python inference_benchmark.py \
    --test-dir "../ScratchDetection/datasets/good_bad_scratches/test" \
    --pytorch-model "../ScratchDetection/models/resnet50_binary.pth" \
    --onnx-model "../ScratchDetection/models/resnet50_binary.onnx" \
    --tensorrt-model "../ScratchDetection/models/resnet50_binary.trt" \
    --num-images 100
```

## Arguments

- `--test-dir`: Directory containing test images (supports jpg, jpeg, png, bmp)
- `--pytorch-model`: Path to PyTorch model file (.pth)
- `--onnx-model`: Path to ONNX model file (.onnx)
- `--tensorrt-model`: Path to TensorRT engine file (.trt)
- `--num-images`: Number of images to benchmark (default: 100)
- `--output-csv`: Output CSV file to save results (default: benchmark_results.csv)

## CSV Output

The benchmark results are automatically saved to a CSV file (default: `benchmark_results.csv`) with the following columns:

- **Timestamp**: When the benchmark was run
- **Model Type**: PyTorch, ONNX, or TensorRT
- **Num Images**: Number of images processed
- **Avg Time (ms)**: Average inference time in milliseconds
- **Median Time (ms)**: Median inference time
- **Std Dev (ms)**: Standard deviation of inference times
- **FPS**: Frames per second
- **Speedup vs PyTorch**: Relative speedup compared to PyTorch baseline

The CSV file is appended to on each run, allowing you to track performance over time.


## Output

The script will display:

1. **Per-model statistics:**
   - Number of images processed
   - Total time
   - Average inference time
   - Median inference time
   - Min/Max times
   - Standard deviation
   - FPS (Frames Per Second)

2. **Comparison summary:**
   - Average inference times for all models
   - Relative speedup compared to PyTorch baseline
   - FPS comparison

## Notes

- The script automatically detects CUDA availability
- First inference includes a warmup run to ensure fair benchmarking
- All times are reported in milliseconds
- Memory is properly managed for TensorRT (allocation/deallocation per image)
