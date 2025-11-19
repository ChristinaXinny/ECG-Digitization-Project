# ECG Digitization Project / ECG 数字化项目

 **Project Source**: This project is based on the Kaggle competition [**PhysioNet - Digitization of ECG Images**](https://www.kaggle.com/competitions/physionet-ecg-image-digitization).  
 We present a comprehensive, production-ready implementation of ECG image to digital signal conversion using a multi-stage deep learning pipeline.

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
│   ├── README.md                     # Main documentation (English)
│   ├── README_CN.md                  # Main documentation (Chinese)
│   ├── PROJECT_SUMMARY.md            # Complete project overview
│   ├── ABLATION_GUIDE.md             # Ablation study guide
│   └── docs/                         # Additional documentation
│       ├── TRAINING_GUIDE.md         # Training guide
│       ├── QUICK_START.md            # Quick start guide
│       ├── PROJECT_STATUS.md         # Project status
│       └── GITIGNORE_GUIDE.md         # Git ignore guide
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

### 3. Inference
```bash
# Run inference with trained model
python inference.py --checkpoint path/to/model.pth --image path/to/ecg.png
eg: python inference.py --checkpoint outputs/stage0_checkpoints/stage0_best.pth --image data/ecg_data_simple/test/1053922973.png
```

### 4. Ablation Studies
```bash
# Run all ablation studies
python ablation.py

# Or run specific studies
python ablation_studies/run_ablation_studies.py --studies backbone loss
```

## 📚 Documentation (文档)

### Language Selection (语言选择)
- **[English Version](README.md)** - For international users
- **[中文版本](README_CN.md)** - 面向中文用户

Both versions contain the same information and are kept synchronized.

### Available Documentation
- [Project Summary](PROJECT_SUMMARY.md) - Complete project introduction
- [Ablation Study Guide](ABLATION_GUIDE.md) - Detailed ablation study instructions
- [Project Structure Guide](PROJECT_STRUCTURE.md) - File organization guide
- [Advanced Training Features](docs/TRAINING_FEATURES.md) - Learning rate scheduling, mixed precision, and more

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
python main.py train --config configs/stage0_config.yaml --resume outputs/stage0_checkpoints/stage0_best.pth 
```

### 2. Inference

**Single image inference**:
```bash
python main.py inference --config configs/inference_config.yaml --input data/ecg_data_simple/test/1053922973.png
```

**Batch inference**:
```bash
python main.py inference --config configs/inference_config.yaml --input data/ecg_data_simple/test/1053922973.png --output outputs/inference/result_batch/
```

**Complete pipeline**:
```bash
python main.py inference --config configs/inference_config.yaml --mode pipeline --input data/ecg_data_simple/test/1053922973.png
```

### 3. Evaluation

**Evaluate model**:
```bash
python main.py evaluate --config configs/inference_config.yaml --model outputs/checkpoints/stage0/best.pth
```

## ⚙️ Configuration

The project uses modular configuration for easy experimentation:

### Core Settings (`configs/base.yaml`)
```yaml
project:
  name: "ECG-Digitization"
  seed: 42

training:
  device: "cuda"
  batch_size: 4
  learning_rate: 1e-4

model:
  backbone: "resnet34"
  pretrained: true
```


### Stage-Specific Configuration
Each stage has its own configuration file that inherits from base.yaml:
- `stage0_config.yaml`: Image normalization settings
- `stage1_config.yaml`: Grid detection settings
- `stage2_config.yaml`: Signal digitization settings
- `inference_config.yaml`: Complete pipeline inference settings

## 📊 Dataset & Data Setup

### Data Download Sources

**Complete Dataset Download Links:**

1. **Official Kaggle Download**:
   - Link: https://www.kaggle.com/competitions/physionet-ecg-image-digitization/data
   - Note: Requires competition participation to download

2. **Baidu Cloud Download**:
   - Link: [Please fill in Baidu Cloud link here]
   - Extraction Code: [Please fill in extraction code here]
   - Description: Backup of competition official data

### Data Directory Structure

**Downloaded data should be placed in the following structure:**

Training data should be organized as:
```

 data/                        # Root data directory
    ├── ecg_data_simple/         # Simple test data (for quick verification)
    │   ├── train/
    │   │   └── images/         # ECG image files 
    │   └── test/
    │       └── images/         # Test images
    │
    └──physionet-ecg-image-digitization
         ├── train/                   # Complete training data (from downloaded dataset)
         │   ├── images/              # ECG image files (.png)
         │   └── series/              # Corresponding CSV time series data
         │
         ├── val/                     # Validation data
         │   ├── images/
         │   └── series/
         │
         └── test/                    # Test data (from downloaded dataset)
            ├── images/
            └── series/
```

**Data Placement Instructions:**
1. Extract the downloaded competition dataset
2. Copy/extract the files to the `data/` directory in the project root
3. Ensure the directory structure matches the format above
4. The `ecg_data_simple/` folder already exists and contains test samples


### Data Format Requirements

- **Images**: PNG format, RGB color
- **Series**: CSV format containing time and voltage values
- **File Naming**: Recommended to use consistent naming convention (such as ID numbers)

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

### Data Preparation
1. Download data from [Kaggle Competition](https://www.kaggle.com/competitions/physionet-ecg-image-digitization)
2. The training script will automatically handle the dataset structure

### Training Commands
```bash
# Train individual stages
python train.py --stage 0 --config configs/stage0_config.yaml
python train.py --stage 1 --config configs/stage1_config.yaml  
python train.py --stage 2 --config configs/stage2_config.yaml

# Or use provided script (if available)
./scripts/train_stage0.sh
```

### Training Monitoring
- Training logs: `outputs/logs/`
- Model checkpoints: outputs/checkpoints/
- Real-time metrics printing in console

## 🔍 Inference Pipeline

### Input Requirements

- **Format**: PNG
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

### Run Tests
```bash
python -m pytest tests/basic_test.py -v
```
The testing framework validates core functionality of the ECG digitization pipeline including data loading, preprocessing, model inference, and output generation. Tests ensure reproducibility and reliability across different components.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

We would like to express our gratitude to:

- **PhysioNet** and **Kaggle** for hosting the [ECG Image Digitization Competition](https://www.kaggle.com/competitions/physionet-ecg-image-digitization) and providing the dataset
- The **medical imaging research community** for foundational work in ECG analysis
- **PyTorch** team for the excellent deep learning framework
- Our institution and advisors for their support and guidance

---

**Note**: This is a research project. For clinical use, ensure proper validation and regulatory compliance.