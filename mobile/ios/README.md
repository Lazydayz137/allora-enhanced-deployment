# Allora MLX iOS App

A native iOS application for managing Allora worker nodes and running ML models on-device using Apple's MLX framework.

## Features

- 🚀 **On-Device ML Inference** - Run models directly on iPhone/iPad using MLX
- 📊 **Worker Management** - Deploy and monitor Allora workers remotely
- 💰 **Earnings Tracker** - Real-time testnet token monitoring
- 🔄 **Model Conversion** - Convert and optimize models for Core ML
- 📱 **Native Performance** - Optimized for Apple Silicon (A-series chips)

## Requirements

- iOS 16.0+ / iPadOS 16.0+
- Xcode 15.0+
- Apple Developer Account (for device testing)
- macOS 14.0+ (for development)

## Architecture

```
AlloraMLX/
├── AlloraMLX/              # Main app target
│   ├── App/                # App lifecycle
│   ├── Views/              # SwiftUI views
│   ├── ViewModels/         # MVVM view models
│   ├── Models/             # Data models
│   ├── Services/           # Business logic
│   │   ├── MLX/           # MLX integration
│   │   ├── Network/       # API services
│   │   └── Storage/       # Local storage
│   └── Resources/          # Assets, fonts, etc.
├── AlloraMLXTests/         # Unit tests
└── AlloraMLXUITests/       # UI tests
```

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/allora-enhanced-deployment.git
   cd allora-enhanced-deployment/mobile/ios
   ```

2. **Open in Xcode**
   ```bash
   open AlloraMLX.xcodeproj
   ```

3. **Configure signing**
   - Select the project in Xcode
   - Go to "Signing & Capabilities"
   - Select your development team
   - Xcode will create provisioning profiles automatically

4. **Build and run**
   - Select your target device (physical device recommended for MLX)
   - Press Cmd+R to build and run

## MLX Integration

The app uses MLX Swift for on-device machine learning:

- **Text Generation** - Using Mistral 7B quantized models
- **Price Prediction** - Custom ensemble models for Allora
- **Model Management** - Download, cache, and update models

## Features Overview

### 1. Dashboard
- Overview of active workers
- Earnings summary
- Model performance metrics

### 2. Worker Management
- Deploy new workers to remote servers
- Monitor worker status and logs
- Configure worker parameters

### 3. ML Models
- Browse available models
- Run inference on-device
- Compare model performance

### 4. Settings
- Server configuration
- Model storage management
- Performance tuning

## Development

### Code Style
We follow Swift's official style guide with SwiftLint enforcement.

### Testing
- Unit tests for business logic
- UI tests for critical user flows
- MLX model testing on physical devices

### Performance
- Optimized for 120Hz ProMotion displays
- Background processing for model downloads
- Efficient memory management for large models
