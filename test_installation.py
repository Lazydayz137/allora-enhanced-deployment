# Test script for Allora Enhanced Deployment
import torch
import psutil
import platform
from transformers import pipeline
import time

print("=== System Information ===")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Processor: {platform.processor()}")
print(f"CPU Cores: {psutil.cpu_count()}")
print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
print()

print("=== GPU Information ===")
if torch.cuda.is_available():
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Test tensor cores
    if torch.cuda.get_device_capability()[0] >= 7:
        print("✓ Tensor cores available (RTX series)")
else:
    print("✗ CUDA not available")
print()

print("=== Model Test ===")
print("Loading small model for RTX 3070...")

try:
    # Test with small model suitable for 8GB VRAM
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = classifier("Allora Network enables decentralized AI inference")
    print(f"✓ Model test successful: {result}")
except Exception as e:
    print(f"✗ Model test failed: {e}")

print()
print("=== Performance Test ===")
if torch.cuda.is_available():
    # Simple benchmark
    size = 1024
    a = torch.randn(size, size).cuda()
    b = torch.randn(size, size).cuda()
    
    # Warmup
    for _ in range(10):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    tflops = (2 * size**3 * 100) / (elapsed * 1e12)
    print(f"Matrix multiplication performance: {tflops:.2f} TFLOPS")

print()
print("✓ All tests completed!")
