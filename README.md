# Allora Enhanced Deployment Suite for GPU-Optimized ML Workers

A comprehensive enhancement suite for the Allora Network that includes worker node scripts optimized for various GPU configurations, from GTX 1080/1080Ti to RTX 3070/3080Ti, with popular open-source models from Hugging Face.

## Overview

Allora Network is a **self-improving decentralized AI network** that enables applications to leverage smarter, more secure AI through a network of machine learning models. This enhanced deployment suite transforms the basic Allora repository into a comprehensive, educational platform for learning machine learning while earning testnet tokens.

## Key Features

- **Hardware Optimization**: Scripts automatically detect and optimize for your specific GPU
- **Model Variety**: Support for lightweight to advanced models based on available resources
- **Cross-Platform**: Unified experience across macOS, Windows, and Linux
- **Learning Focus**: Educational examples for understanding ML concepts
- **Earnings Optimization**: Strategies for maximizing testnet token rewards

## GPU Requirements

### Low-Tier GPUs (8-11GB VRAM)
- **GTX 1080 (8GB)**: Lacks tensor cores but sufficient for lightweight models
- **GTX 1080 Ti (11GB)**: Better VRAM capacity, good for medium-size models

### Mid-Tier GPUs (8-12GB VRAM)
- **RTX 3070 (8GB)**: Modern architecture with tensor cores for accelerated training
- **RTX 3080 (10GB)**: Excellent performance for most ML workloads

### High-Tier GPUs (12GB+ VRAM)
- **RTX 3080 Ti (12GB)**: Ideal for larger models and complex inference tasks

## Recommended Models by GPU Tier

### For 8GB VRAM (GTX 1080, RTX 3070):
- **DistilBERT**: 40% smaller than BERT, 60% faster inference
- **Microsoft DialoGPT-small**: Conversational AI optimized for memory efficiency
- **Google FLAN-T5-small**: Instruction-tuned text-to-text transformer
- **CodeBERTa-small**: Code understanding and generation

### For 11-12GB VRAM (1080Ti, 3080, 3080Ti):
- **BERT-base-uncased**: Full-size BERT for comprehensive NLP tasks
- **GPT-Neo-1.3B**: Open-source alternative to GPT models
- **Facebook BART-large**: Advanced text generation and summarization
- **Time-series models**: LSTM/GRU models for price prediction

## Quick Start

### Windows Installation
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy Bypass -Scope Process -Force
.\scripts\windows\install.ps1
```

### macOS Installation
```bash
chmod +x scripts/macos/install.sh
./scripts/macos/install.sh
```

### Linux Installation
```bash
chmod +x scripts/linux/install.sh
./scripts/linux/install.sh
```

## Documentation

- [GPU Requirements Guide](docs/gpu_requirements.md)
- [Model Selection Guide](docs/model_selection.md)
- [Installation Guide](docs/installation_guide.md)
- [Troubleshooting](docs/troubleshooting.md)

## Worker Types

1. **Price Prediction Workers**: Basic linear regression to transformer-based models
2. **Text Classification Workers**: Market sentiment analysis
3. **Advanced Time-Series Workers**: Multi-modal prediction combining price data with sentiment

## Contributing

Feel free to submit issues and enhancement requests!

## License

Apache License 2.0
