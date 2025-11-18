# ECG Digitization Project

A comprehensive, production-ready implementation of ECG image to digital signal conversion using multi-stage deep learning.

## 🚀 Project Overview

This project converts ECG images through a three-stage deep learning pipeline:

1. **Stage 0**: Image normalization and keypoint detection
2. **Stage 1**: Image rectification and grid alignment
3. **Stage 2**: Signal digitization and time series extraction

## 📁 Project Structure

```
ECG-Digitization-Project/
├── 📄 Quick Access Scripts
│   ├── train.py                      # Quick training access
│   ├── test.py                       # Quick testing access
│   ├── inference.py                  # Quick inference access
│   └── ablation.py                   # Quick ablation studies access
│
├── 📁 Core Directories
│   ├── configs/                      # Configuration files
│   ├── data/                         # Data handling (dataset, preprocessing)
│   ├── models/                       # Model definitions and heads
│   ├── engines/                      # Training/inference engines
│   ├── utils/                        # Utility functions (metrics, logging)
│   └── ablation_studies/             # Ablation study framework
│
├── 📁 Scripts & Tools
│   ├── scripts/                      # All training and utility scripts
│   │   ├── train_stage0.py           # Main training script
│   │   ├── load_model.py             # Model loading and inference
│   │   ├── main.py                   # Entry point script
│   │   └── *.py                      # Additional training scripts
│   └── tests/                        # Comprehensive test suite
│       ├── run_simple_tests.py       # Quick validation tests
│       ├── basic_test.py             # Basic functionality tests
│       ├── test_*.py                 # Specialized test files
│       └── README.md                 # Test documentation
│
├── 📁 Documentation
│   ├── README.md                     # Main documentation
│   ├── PROJECT_SUMMARY.md            # Complete project overview
│   ├── ABLATION_GUIDE.md             # Ablation study guide
│   └── docs/                         # Additional documentation
│       ├── TRAINING_GUIDE.md         # Training guide
│       ├── QUICK_START.md            # Quick start guide
│       └── PROJECT_STATUS.md         # Project status
│
└── 📁 Output & Build
    ├── outputs/                      # Training outputs and checkpoints
    ├── requirements.txt              # Python dependencies
    ├── Makefile                      # Build automation
    └── setup.py                      # Package setup
```

## 🚀 Quick Start

### 1. Quick Validation
```bash
# Run quick tests to validate installation
python test.py

# Or run comprehensive tests
python tests/run_simple_tests.py
```

### 2. Training
```bash
# Start training from project root
python train.py

# Or specify training script directly
python scripts/train_stage0.py
```

### 3. Ablation Studies
```bash
# Run all ablation studies
python ablation.py

# Or run specific studies
python ablation_studies/run_ablation_studies.py --studies backbone loss
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM (16GB+ recommended for training)

### Setup Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ChristinaXinny/ECG-Digitization-Project
   cd ECG-Digitization-Project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```


## 🚀 Quick Start

### 1. Training

**Train all stages**:
```bash
python main.py train --config configs/base.yaml --mode all
```

### 2. Evaluation

**Evaluate model**:
```bash
python main.py evaluate --config configs/inference_config.yaml --model outputs/checkpoints/stage0/best.pth
```

## 📈 Model Architecture

### Stage 0: Image Normalization
- **Backbone**: ResNet-18D encoder
- **Decoder**: Custom U-Net with skip connections
- **Heads**: Lead segmentation + Orientation classification

### Stage 1: Grid Detection
- **Backbone**: ResNet-34 encoder
- **Decoder**: Coordinate-aware U-Net
- **Heads**: Grid point detection + Grid line classification

### Stage 2: Signal Digitization
- **Backbone**: ResNet-34D encoder
- **Decoder**: Coordinate-enhanced U-Net
- **Heads**: Pixel segmentation + Signal regression

## 🎯 Training Pipeline

### Data Requirements

Training data should be organized as:
```
data/
├── train/
│   ├── images/
│   ├── annotations/
│   └── series/
├── val/
└── test/
```

## 🔍 Inference Pipeline

### Input Requirements

- **Format**: PNG, JPG, JPEG
- **Color**: RGB
- **Quality**: Clear ECG traces with visible grid

### Output Format

```csv
id,value
sample_001_0_I,0.123
sample_001_1_I,0.145
sample_001_0_II,-0.234
...
```

## 🧪 Testing

### Run Unit Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Tests
```bash
python -m pytest tests/test_models.py -v
python -m pytest tests/test_data.py -v
```

## 📊 Performance

### Benchmarks
- **Inference time**: ~0.5s per image (GPU)
- **Memory usage**: ~2GB GPU memory
- **Accuracy**: Competitor-level performance

### Optimization Tips
- Use mixed precision training
- Enable gradient accumulation
- Use appropriate batch sizes


---

**Note**: This is a research project. For clinical use, ensure proper validation and regulatory compliance.
