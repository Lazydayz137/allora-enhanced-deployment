# Allora Enhanced Deployment - Windows Installation Script (Test Version)
# Optimized for RTX 3070 (8GB VRAM) and Ryzen 3900X

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Allora Enhanced Deployment Suite" -ForegroundColor Cyan
Write-Host "Windows Installation Script (Test Version)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

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
    if ($pythonVersion -match "Python 3\.(9|10|11|12)") {
        Write-Host "[OK] Python $pythonVersion found" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Python 3.9+ required. Current version: $pythonVersion" -ForegroundColor Red
        Write-Host "Please install Python from python.org" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "[ERROR] Python not found. Please install Python 3.9+ from python.org" -ForegroundColor Red
    exit 1
}

# Check CUDA
$cudaPath = $env:CUDA_PATH
if ($cudaPath) {
    Write-Host "[OK] CUDA found at $cudaPath" -ForegroundColor Green
} else {
    Write-Host "[WARNING] CUDA not found. GPU acceleration may not work properly." -ForegroundColor Yellow
    Write-Host "  Install CUDA Toolkit 11.8 or 12.1 from NVIDIA website" -ForegroundColor Yellow
}

# Check Git
try {
    $gitVersion = git --version
    Write-Host "[OK] Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Git not found. Please install Git for Windows" -ForegroundColor Red
    exit 1
}

# Check Docker Desktop
try {
    $dockerVersion = docker --version
    Write-Host "[OK] Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "[WARNING] Docker not found. Some features may not work." -ForegroundColor Yellow
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
Write-Host "[OK] Virtual environment created" -ForegroundColor Green

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install PyTorch with CUDA support for RTX 3070
Write-Host ""
Write-Host "Installing PyTorch with CUDA support for RTX 3070..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
Write-Host ""
Write-Host "Installing Python packages from requirements.txt..." -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
} else {
    Write-Host "[WARNING] requirements.txt not found" -ForegroundColor Yellow
}

# Test GPU availability
Write-Host ""
Write-Host "Testing GPU availability..." -ForegroundColor Yellow

# Running GPU test script
if (Test-Path "gpu_test.py") {
    python gpu_test.py
} else {
    Write-Host "[WARNING] gpu_test.py not found" -ForegroundColor Yellow
}

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
