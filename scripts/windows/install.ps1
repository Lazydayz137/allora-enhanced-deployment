# Allora Enhanced Deployment - Windows Installation Script
# Optimized for RTX 3070 (8GB VRAM) and Ryzen 3900X

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Allora Enhanced Deployment Suite" -ForegroundColor Cyan
Write-Host "Windows Installation Script" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "This script requires Administrator privileges. Please run PowerShell as Administrator." -ForegroundColor Red
    exit 1
}

# System Detection
Write-Host "Detecting System Configuration..." -ForegroundColor Yellow
$gpu = Get-WmiObject Win32_VideoController | Where-Object {$_.Name -like "*NVIDIA*"}
$cpu = Get-WmiObject Win32_Processor | Select-Object -First 1
$ram = [math]::Round((Get-WmiObject Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)

Write-Host "CPU: $($cpu.Name)" -ForegroundColor Green
Write-Host "GPU: $($gpu.Name)" -ForegroundColor Green
Write-Host "RAM: $ram GB" -ForegroundColor Green
Write-Host ""

# Check for RTX 3070
if ($gpu.Name -like "*3070*") {
    Write-Host "RTX 3070 detected! Optimizing for 8GB VRAM configuration..." -ForegroundColor Green
    $gpu_tier = "mid"
    $vram = 8
} else {
    Write-Host "GPU: $($gpu.Name) detected. Adjusting configuration..." -ForegroundColor Yellow
    $gpu_tier = "auto"
    $vram = "auto"
}

# Check Prerequisites
Write-Host "Checking Prerequisites..." -ForegroundColor Yellow

# Check Python
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python 3\.(9|10|11)") {
        Write-Host "✓ Python $pythonVersion found" -ForegroundColor Green
    } else {
        Write-Host "✗ Python 3.9+ required. Please install from python.org" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "✗ Python not found. Please install Python 3.9+ from python.org" -ForegroundColor Red
    exit 1
}

# Check CUDA
$cudaPath = $env:CUDA_PATH
if ($cudaPath) {
    Write-Host "✓ CUDA found at $cudaPath" -ForegroundColor Green
} else {
    Write-Host "⚠ CUDA not found. GPU acceleration may not work properly." -ForegroundColor Yellow
    Write-Host "  Install CUDA Toolkit 11.8 or 12.1 from NVIDIA website" -ForegroundColor Yellow
}

# Check Git
try {
    $gitVersion = git --version
    Write-Host "✓ Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Git not found. Please install Git for Windows" -ForegroundColor Red
    exit 1
}

# Check Docker Desktop
try {
    $dockerVersion = docker --version
    Write-Host "✓ Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "⚠ Docker not found. Some features may not work." -ForegroundColor Yellow
    Write-Host "  Install Docker Desktop from docker.com" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Creating Python Virtual Environment..." -ForegroundColor Yellow

# Create virtual environment
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists. Removing old environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force venv
}

python -m venv venv
Write-Host "✓ Virtual environment created" -ForegroundColor Green

# Activate virtual environment
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install PyTorch with CUDA support for RTX 3070
Write-Host ""
Write-Host "Installing PyTorch with CUDA support for RTX 3070..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Create requirements file
Write-Host ""
Write-Host "Creating requirements.txt..." -ForegroundColor Yellow

@"
# Core Dependencies
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0

# ML Framework (PyTorch installed separately with CUDA)
transformers>=4.30.0
datasets>=2.14.0
accelerate>=0.20.0
bitsandbytes>=0.41.0  # For 8-bit optimization on RTX 3070

# Allora Network
requests>=2.28.0
web3>=6.0.0
python-dotenv>=0.20.0

# Model Optimization for 8GB VRAM
peft>=0.4.0  # Parameter-Efficient Fine-Tuning
optimum>=1.13.0  # Model optimization

# Monitoring and Logging
tensorboard>=2.11.0
wandb>=0.15.0
tqdm>=4.65.0

# API and Server
fastapi>=0.95.0
uvicorn>=0.21.0
pydantic>=1.10.0

# Testing
pytest>=7.2.0
pytest-asyncio>=0.20.0
"@ | Out-File -FilePath requirements.txt -Encoding UTF8

Write-Host 'Requirements file created successfully' -ForegroundColor Green

# Install requirements
Write-Host ""
Write-Host "Installing Python packages..." -ForegroundColor Yellow
pip install -r requirements.txt

# Test GPU availability
Write-Host ""
Write-Host "Testing GPU availability..." -ForegroundColor Yellow

# Running GPU test script
python gpu_test.py

# Create configuration file
Write-Host ""
Write-Host "Creating configuration file..." -ForegroundColor Yellow

@"
# Allora Enhanced Deployment Configuration
# Auto-generated for RTX 3070 (8GB VRAM)

[system]
gpu_tier = "mid"
gpu_name = "RTX 3070"
vram_gb = 8
cpu_name = "AMD Ryzen 9 3900X"
cpu_cores = 12

[models]
# Recommended models for 8GB VRAM
default_model = "microsoft/DialoGPT-small"
available_models = [
    "distilbert-base-uncased",
    "microsoft/DialoGPT-small",
    "google/flan-t5-small",
    "microsoft/codebert-base"
]

[optimization]
# RTX 3070 optimizations
use_mixed_precision = true
use_tensor_cores = true
batch_size = 16
gradient_accumulation_steps = 2
max_sequence_length = 512

[allora]
# Network configuration
testnet_rpc = "https://allora-testnet-rpc.testnet.allora.network"
chain_id = "allora-testnet-1"
"@ | Out-File -FilePath config.ini -Encoding UTF8

Write-Host 'Configuration file created successfully' -ForegroundColor Green

# Create example worker script
Write-Host ""
Write-Host "Creating example worker script for RTX 3070..." -ForegroundColor Yellow

New-Item -ItemType Directory -Path workers/price_prediction -Force | Out-Null

@"
# Price Prediction Worker - Optimized for RTX 3070
# Example implementation using lightweight models

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PricePredictionWorker:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Load model optimized for 8GB VRAM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3  # bearish, neutral, bullish
        ).to(self.device)
        
        # Enable mixed precision for RTX 3070 tensor cores
        self.model.half()
        logger.info(f"Model loaded: {model_name}")
    
    def predict_sentiment(self, text):
        """Predict market sentiment from text"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        sentiment_map = {0: "bearish", 1: "neutral", 2: "bullish"}
        predicted_class = predictions.argmax().item()
        confidence = predictions.max().item()
        
        return {
            "sentiment": sentiment_map[predicted_class],
            "confidence": float(confidence),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def run(self):
        """Main worker loop"""
        logger.info("Price Prediction Worker started")
        
        # Example prediction
        test_texts = [
            "Bitcoin shows strong bullish momentum with increasing volume",
            "Market uncertainty continues as traders await Fed decision",
            "Massive sell-off triggered by regulatory concerns"
        ]
        
        for text in test_texts:
            result = self.predict_sentiment(text)
            logger.info(f"Text: {text[:50]}...")
            logger.info(f"Result: {result}")

if __name__ == "__main__":
    worker = PricePredictionWorker()
    worker.run()
"@ | Out-File -FilePath workers/price_prediction/worker.py -Encoding UTF8

Write-Host 'Example worker created successfully' -ForegroundColor Green

# Create test script
Write-Host ""
Write-Host "Creating test script..." -ForegroundColor Yellow

@"
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
"@ | Out-File -FilePath test_installation.py -Encoding UTF8

Write-Host 'Test script created successfully' -ForegroundColor Green

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Run test script: python test_installation.py" -ForegroundColor White
Write-Host "2. Check worker example: python workers/price_prediction/worker.py" -ForegroundColor White
Write-Host "3. Configure your Allora wallet in config.ini" -ForegroundColor White
Write-Host ""
Write-Host "Your RTX 3070 is optimized for:" -ForegroundColor Cyan
Write-Host "- Mixed precision training (FP16)" -ForegroundColor White
Write-Host "- Tensor core acceleration" -ForegroundColor White
Write-Host "- 8GB VRAM-optimized models" -ForegroundColor White
Write-Host ""
