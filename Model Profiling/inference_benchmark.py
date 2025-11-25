"""
Inference Benchmark Script
Benchmarks PyTorch, ONNX, and TensorRT models for inference performance.

USAGE EXAMPLES:
---------------

1. Run benchmark on 100 images (default, saves to benchmark_results.csv):
   python inference_benchmark.py --test-dir ../ScratchDetection/datasets/good_bad_scratches/test --pytorch-model ../ScratchDetection/models/resnet50_binary.pth --onnx-model ../ScratchDetection/models/resnet50_binary.onnx --tensorrt-model ../ScratchDetection/models/resnet50_binary.trt

2. Run benchmark with custom output CSV file:
   python inference_benchmark.py --test-dir ../ScratchDetection/datasets/good_bad_scratches/test --pytorch-model ../ScratchDetection/models/resnet50_binary.pth --onnx-model ../ScratchDetection/models/resnet50_binary.onnx --tensorrt-model ../ScratchDetection/models/resnet50_binary.trt --output-csv my_results.csv

3. Run benchmark on 50 images:
   python inference_benchmark.py --test-dir ../ScratchDetection/datasets/good_bad_scratches/test --pytorch-model ../ScratchDetection/models/resnet50_binary.pth --onnx-model ../ScratchDetection/models/resnet50_binary.onnx --tensorrt-model ../ScratchDetection/models/resnet50_binary.trt --num-images 50

4. Benchmark different model:
   python inference_benchmark.py --test-dir ../ScratchDetection/datasets/good_bad_scratches/test --pytorch-model ../ScratchDetection/models/mobilenetv3_binary.pth --onnx-model ../ScratchDetection/models/mobilenetv3_binary.onnx --tensorrt-model ../ScratchDetection/models/mobilenetv3_binary.trt --num-images 100

4. Use custom test images directory:
   python inference_benchmark.py --test-dir path/to/your/test/images --pytorch-model ../ScratchDetection/models/resnet50_binary.pth --onnx-model ../ScratchDetection/models/resnet50_binary.onnx --tensorrt-model ../ScratchDetection/models/resnet50_binary.trt --num-images 100

OUTPUT:
-------
- Detailed statistics for each model (PyTorch, ONNX, TensorRT)
- Comparison summary with speedup metrics
- No results saved to disk - only console output
"""

import os
import time
import torch
import numpy as np
import onnxruntime
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
from pathlib import Path
from PIL import Image
import glob
from tqdm import tqdm
import timm
import torch.nn as nn
from torchvision import transforms



class InferenceBenchmark:
    def __init__(self, test_images_dir: str, num_images: int = 100):
        """
        Initialize the benchmark with test images directory and number of images to test.
        
        Args:
            test_images_dir: Directory containing test images
            num_images: Number of images to run inference on (default: 100)
        """
        self.test_images_dir = Path(test_images_dir)
        self.num_images = num_images
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load test images
        self.image_paths = self._load_image_paths()
        print(f"Loaded {len(self.image_paths)} images from {test_images_dir}")
        
    def _load_image_paths(self):
        """Load image paths from the test directory"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(glob.glob(str(self.test_images_dir / '**' / ext), recursive=True))
        
        # Limit to num_images
        return image_paths[:self.num_images]
    
    def _load_image(self, image_path):
        """Load and preprocess a single image"""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)
    
    def benchmark_pytorch(self, model_path: str):
        """Benchmark PyTorch model"""
        print("\n" + "="*60)
        print("PYTORCH MODEL BENCHMARK")
        print("="*60)
        
        # Load checkpoint to inspect
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Auto-detect model architecture
        if 'layer1.2.conv3.weight' in checkpoint:
            # Has bottleneck blocks (ResNet50/101/152)
            if 'layer3.5.conv1.weight' in checkpoint:
                model_arch = 'resnet50'
            elif 'layer3.22.conv1.weight' in checkpoint:
                model_arch = 'resnet101'
            else:
                model_arch = 'resnet50'
        else:
            # Basic blocks (ResNet18/34)
            model_arch = 'resnet18'
        
        # Detect number of output classes
        fc_weight_shape = checkpoint.get('fc.weight', checkpoint.get('head.weight'))
        if fc_weight_shape is not None:
            num_classes = fc_weight_shape.shape[0]
        else:
            num_classes = 2
        
        print(f"  Detected: {model_arch} with {num_classes} output classes")
        
        # Load model
        model = timm.create_model(model_arch, pretrained=False, num_classes=num_classes)
        model.load_state_dict(checkpoint)
        model = model.to(self.device)
        model.eval()
        
        inference_times = []
        
        # Warmup
        warmup_img = self._load_image(self.image_paths[0]).to(self.device)
        with torch.no_grad():
            _ = model(warmup_img)
        
        # Benchmark
        with torch.no_grad():
            for img_path in tqdm(self.image_paths, desc="PyTorch Inference"):
                img_tensor = self._load_image(img_path).to(self.device)
                
                start_time = time.perf_counter()
                output = model(img_tensor)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.perf_counter()
                
                inference_times.append(end_time - start_time)
        
        self._print_stats("PyTorch", inference_times)
        return inference_times
    
    def benchmark_onnx(self, model_path: str):
        """Benchmark ONNX model"""
        print("\n" + "="*60)
        print("ONNX MODEL BENCHMARK")
        print("="*60)
        
        # Create ONNX session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name
        
        inference_times = []
        
        # Warmup
        warmup_img = self._load_image(self.image_paths[0]).numpy()
        _ = session.run(None, {input_name: warmup_img})
        
        # Benchmark
        for img_path in tqdm(self.image_paths, desc="ONNX Inference"):
            img_tensor = self._load_image(img_path).numpy()
            
            start_time = time.perf_counter()
            output = session.run(None, {input_name: img_tensor})
            end_time = time.perf_counter()
            
            inference_times.append(end_time - start_time)
        
        self._print_stats("ONNX", inference_times)
        return inference_times
    
    def benchmark_tensorrt(self, engine_path: str):
        """Benchmark TensorRT model"""
        print("\n" + "="*60)
        print("TENSORRT MODEL BENCHMARK")
        print("="*60)
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        context = engine.create_execution_context()
        
        inference_times = []
        
        # Warmup
        warmup_img = self._load_image(self.image_paths[0]).numpy()
        d_input = cuda.mem_alloc(warmup_img.nbytes)
        output = np.empty((1, 1), dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)
        bindings = [int(d_input), int(d_output)]
        cuda.memcpy_htod(d_input, warmup_img)
        context.execute_v2(bindings=bindings)
        cuda.memcpy_dtoh(output, d_output)
        d_input.free()
        d_output.free()
        
        # Benchmark
        for img_path in tqdm(self.image_paths, desc="TensorRT Inference"):
            img_tensor = self._load_image(img_path).numpy()
            
            # Allocate device memory
            d_input = cuda.mem_alloc(img_tensor.nbytes)
            output = np.empty((1, 1), dtype=np.float32)
            d_output = cuda.mem_alloc(output.nbytes)
            bindings = [int(d_input), int(d_output)]
            
            start_time = time.perf_counter()
            # Transfer input data to device
            cuda.memcpy_htod(d_input, img_tensor)
            # Execute model
            context.execute_v2(bindings=bindings)
            # Transfer predictions back
            cuda.memcpy_dtoh(output, d_output)
            end_time = time.perf_counter()
            
            inference_times.append(end_time - start_time)
            
            # Free memory
            d_input.free()
            d_output.free()
        
        self._print_stats("TensorRT", inference_times)
        return inference_times
    
    def _print_stats(self, model_type: str, inference_times: list):
        """Print benchmark statistics"""
        times_ms = np.array(inference_times) * 1000  # Convert to milliseconds
        
        print(f"\n{model_type} Results:")
        print(f"  Number of images:     {len(inference_times)}")
        print(f"  Total time:           {sum(times_ms):.2f} ms")
        print(f"  Average time:         {np.mean(times_ms):.2f} ms")
        print(f"  Median time:          {np.median(times_ms):.2f} ms")
        print(f"  Min time:             {np.min(times_ms):.2f} ms")
        print(f"  Max time:             {np.max(times_ms):.2f} ms")
        print(f"  Std deviation:        {np.std(times_ms):.2f} ms")
        print(f"  FPS:                  {1000 / np.mean(times_ms):.2f}")
    
    def run_all_benchmarks(self, pytorch_path: str, onnx_path: str, tensorrt_path: str, output_csv: str = None):
        """Run benchmarks for all three model types and compare"""
        print(f"\nRunning benchmarks on {len(self.image_paths)} images...")
        print(f"Device: {self.device}")
        
        # Run benchmarks
        pt_times = self.benchmark_pytorch(pytorch_path)
        onnx_times = self.benchmark_onnx(onnx_path)
        trt_times = self.benchmark_tensorrt(tensorrt_path)
        
        # Calculate statistics
        pt_avg = np.mean(pt_times) * 1000
        onnx_avg = np.mean(onnx_times) * 1000
        trt_avg = np.mean(trt_times) * 1000
        
        pt_median = np.median(pt_times) * 1000
        onnx_median = np.median(onnx_times) * 1000
        trt_median = np.median(trt_times) * 1000
        
        pt_std = np.std(pt_times) * 1000
        onnx_std = np.std(onnx_times) * 1000
        trt_std = np.std(trt_times) * 1000
        
        # Print comparison
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        print(f"\nAverage Inference Time:")
        print(f"  PyTorch:    {pt_avg:.2f} ms  (1.00x - baseline)")
        print(f"  ONNX:       {onnx_avg:.2f} ms  ({pt_avg/onnx_avg:.2f}x speedup)")
        print(f"  TensorRT:   {trt_avg:.2f} ms  ({pt_avg/trt_avg:.2f}x speedup)")
        
        print(f"\nFPS:")
        print(f"  PyTorch:    {1000/pt_avg:.2f}")
        print(f"  ONNX:       {1000/onnx_avg:.2f}")
        print(f"  TensorRT:   {1000/trt_avg:.2f}")
        
        print("\n" + "="*60)
        
        # Save results to CSV
        if output_csv:
            import pandas as pd
            from datetime import datetime
            
            results_df = pd.DataFrame({
                'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * 3,
                'Model Type': ['PyTorch', 'ONNX', 'TensorRT'],
                'Num Images': [len(self.image_paths)] * 3,
                'Avg Time (ms)': [pt_avg, onnx_avg, trt_avg],
                'Median Time (ms)': [pt_median, onnx_median, trt_median],
                'Std Dev (ms)': [pt_std, onnx_std, trt_std],
                'FPS': [1000/pt_avg, 1000/onnx_avg, 1000/trt_avg],
                'Speedup vs PyTorch': [1.0, pt_avg/onnx_avg, pt_avg/trt_avg]
            })
            
            # Append to CSV if it exists, otherwise create new
            try:
                existing_df = pd.read_csv(output_csv)
                combined_df = pd.concat([existing_df, results_df], ignore_index=True)
                combined_df.to_csv(output_csv, index=False)
            except FileNotFoundError:
                results_df.to_csv(output_csv, index=False)
            
            print(f"\n✓ Results saved to: {output_csv}")
        
        return {
            'pytorch': pt_times,
            'onnx': onnx_times,
            'tensorrt': trt_times
        }


def main():
    """Main function to run the benchmark"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark PyTorch, ONNX, and TensorRT models')
    parser.add_argument('--test-dir', type=str, required=True, 
                        help='Directory containing test images')
    parser.add_argument('--pytorch-model', type=str, required=True,
                        help='Path to PyTorch model (.pth)')
    parser.add_argument('--onnx-model', type=str, required=True,
                        help='Path to ONNX model (.onnx)')
    parser.add_argument('--tensorrt-model', type=str, required=True,
                        help='Path to TensorRT engine (.trt)')
    parser.add_argument('--num-images', type=int, default=100,
                        help='Number of images to benchmark (default: 100)')
    parser.add_argument('--output-csv', type=str, default='benchmark_results.csv',
                        help='Output CSV file to save results (default: benchmark_results.csv)')
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = InferenceBenchmark(args.test_dir, args.num_images)
    
    # Run all benchmarks
    benchmark.run_all_benchmarks(args.pytorch_model, args.onnx_model, args.tensorrt_model, args.output_csv)


if __name__ == '__main__':
    main()
