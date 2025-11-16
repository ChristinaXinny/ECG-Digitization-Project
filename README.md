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
├── configs/                          # Configuration files
│   ├── base.yaml                     # Base configuration
│   ├── stage0_config.yaml            # Stage0 model configuration
│   ├── stage1_config.yaml            # Stage1 model configuration
│   ├── stage2_config.yaml            # Stage2 model configuration
│   └── inference_config.yaml         # Inference configuration
├── data/                             # Data handling
│   ├── dataset.py                    # Dataset classes
│   ├── preprocessing.py              # Data preprocessing
│   └── transforms.py                 # Data augmentation
├── models/                           # Model definitions
│   ├── base_model.py                 # Base model class
│   ├── stage0_model.py               # Stage0 model
│   ├── stage1_model.py               # Stage1 model
│   ├── stage2_model.py               # Stage2 model
│   └── heads/                        # Model heads
│       ├── detection_head.py         # Detection heads
│       ├── regression_head.py        # Regression heads
│       └── segmentation_head.py      # Segmentation heads
├── engines/                          # Training/inference engines
│   ├── base_trainer.py               # Base trainer
│   ├── stage_trainer.py              # Stage-specific trainers
│   ├── inference.py                  # Inference engine
│   └── validation.py                 # Validation engine
├── utils/                            # Utility functions
│   ├── logger.py                     # Logging utilities
│   ├── metrics.py                    # Evaluation metrics
│   ├── visualization.py              # Visualization tools
│   ├── config_loader.py              # Configuration loader
│   └── checkpoint.py                 # Model checkpoint utilities
├── scripts/                          # Running scripts
│   ├── train_all.sh                  # Complete training script
│   ├── train_stage0.sh               # Stage0 training
│   ├── train_stage1.sh               # Stage1 training
│   ├── train_stage2.sh               # Stage2 training
│   ├── inference.sh                  # Inference script
│   └── setup_environment.sh          # Environment setup
├── tests/                            # Unit tests
├── outputs/                          # Output directory
│   ├── checkpoints/                  # Model weights
│   ├── logs/                         # Training logs
│   ├── predictions/                  # Prediction results
│   └── visualizations/               # Visualization results
├── docs/                             # Documentation
├── main.py                          # Main entry point
├── train.py                         # Training entry
├── inference.py                     # Inference entry
└── requirements.txt                 # Dependencies
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM (16GB+ recommended for training)

### Setup Environment

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ECG-Digitization-Project
   ```

2. **Create conda environment**:
   ```bash
   conda create -n ecg-digit python=3.9
   conda activate ecg-digit
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment**:
   ```bash
   python main.py setup --config configs/base.yaml
   ```

### Environment Setup Script

```bash
# Run complete setup
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh
```

## 🚀 Quick Start

### 1. Training

**Train all stages**:
```bash
python main.py train --config configs/base.yaml --mode all
```

**Train specific stage**:
```bash
python main.py train --config configs/stage0_config.yaml --mode stage0
python main.py train --config configs/stage1_config.yaml --mode stage1
python main.py train --config configs/stage2_config.yaml --mode stage2
```

**Resume training**:
```bash
python main.py train --config configs/stage0_config.yaml --resume outputs/checkpoints/stage0/latest.pth
```

### 2. Inference

**Single image inference**:
```bash
python main.py inference --config configs/inference_config.yaml --input path/to/ecg.png
```

**Batch inference**:
```bash
python main.py inference --config configs/inference_config.yaml --input /path/to/images/ --output /path/to/results/
```

**Complete pipeline**:
```bash
python main.py inference --config configs/inference_config.yaml --mode pipeline --input /path/to/images/
```

### 3. Evaluation

**Evaluate model**:
```bash
python main.py evaluate --config configs/inference_config.yaml --model outputs/checkpoints/stage0/best.pth
```

## 📊 Configuration

The project uses YAML configuration files for easy customization:

### Base Configuration (`configs/base.yaml`)
```yaml
# Project settings
PROJECT_NAME: "ECG Digitization"
SEED: 42

# Device settings
DEVICE:
  DEVICE: "cuda"
  GPU_IDS: [0]
  MIXED_PRECISION: True

# Model settings
MODEL:
  BACKBONE:
    NAME: "resnet18d"
    PRETRAINED: True

# Training settings
TRAIN:
  BATCH_SIZE: 4
  EPOCHS: 100
  LEARNING_RATE: 1e-4
```

### Stage-Specific Configuration
Each stage has its own configuration file that inherits from base.yaml:
- `stage0_config.yaml`: Image normalization settings
- `stage1_config.yaml`: Grid detection settings
- `stage2_config.yaml`: Signal digitization settings
- `inference_config.yaml`: Complete pipeline inference settings

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

### Training Scripts

**Complete training**:
```bash
chmod +x scripts/train_all.sh
./scripts/train_all.sh
```

**Individual stage training**:
```bash
chmod +x scripts/train_stage0.sh
./scripts/train_stage0.sh
```

### Monitoring Training

Training progress is logged to:
- Console output
- Log files (`outputs/logs/`)
- TensorBoard (if enabled)
- Weights & Biases (if configured)

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

### Processing Modes

- **Single stage**: Process individual stage
- **Complete pipeline**: All three stages
- **Batch processing**: Multiple images
- **Real-time**: Single image with minimal latency

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

### Test Coverage
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

## 📝 Documentation

- **[Architecture Guide](docs/architecture.md)**: Detailed system architecture
- **[API Reference](docs/api_reference.md)**: Complete API documentation
- **[Troubleshooting](docs/troubleshooting.md)**: Common issues and solutions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push branch: `git push origin feature-name`
5. Submit a pull request

### Code Style
```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size in config
   - Use CPU inference
   - Enable gradient checkpointing

2. **Model weights not found**:
   - Run setup script: `python main.py setup`
   - Download weights manually

3. **Poor quality results**:
   - Check input image quality
   - Verify correct configuration
   - Ensure proper model weights

### Debug Mode
```bash
python main.py train --config configs/base.yaml --debug --verbose
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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original Kaggle competition implementation
- Medical imaging research community
- PyTorch and deep learning ecosystem

## 📞 Contact

For questions and support:
- Create an issue in the repository
- Email: [your-email@example.com]

---

**Note**: This is a research project. For clinical use, ensure proper validation and regulatory compliance.