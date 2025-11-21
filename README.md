# ECG Digitization Project / ECG 数字化项目

**Project Source**: This project is based on the Kaggle competition [**PhysioNet - Digitization of ECG Images**](https://www.kaggle.com/competitions/physionet-ecg-image-digitization).  
We present a comprehensive implementation of ECG image to digital signal conversion using a multi-stage deep learning pipeline.

## 🚀 Project Overview

This project converts ECG images through a three-stage deep learning pipeline:

1. **Stage 0**: Image normalization and orientation correction
2. **Stage 1**: Grid detection and image rectification
3. **Stage 2**: Signal extraction and digitization

**Key Results**: Achieved **16.12 dB SNR** on test data, demonstrating clinically viable signal quality.

## 🛠️ Installation 

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA-compatible GPU (recommended)

### Installation

```bash
git clone https://github.com/ChristinaXinny/ECG-Digitization-Project
cd ECG-Digitization-Project
pip install -r requirements.txt
```

## 🚀Quick Start

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
python main.py train --config configs/stage0_config.yaml --resume path/to/model.pth 
```

### 2. Inference

**Single image inference**:
```bash
python main.py inference --config configs/inference_config.yaml --input path/to/ecg.png
```

**Batch inference**:
```bash
python main.py inference --config configs/inference_config.yaml --input path/to/ecg.png --output outputs/inference/result_batch/
```

**Complete pipeline**:
```bash
python main.py inference --config configs/inference_config.yaml --mode pipeline --input path/to/ecg.png
```

### 3. Evaluation

**Evaluate model**:
```bash
python main.py evaluate --config configs/inference_config.yaml --model path/to/model.pth
```
### 4. Testing

**Run Tests**
```bash
python -m pytest tests/basic_test.py -v
```

## 📁 Project Structure

```
ECG-Digitization-Project/
├── configs/          # Configuration files
├── models/           # Model definitions (stage0,1,2)
├── engines/          # Training/inference engines  
├── utils/            # Metrics, logging, visualization
├── data/            # Data loading and preprocessing
├── tests/           # Test suite
└── outputs/         # Results and checkpoints
```

## 📊 Dataset & Data Setup

### Data Download Sources

**Official Kaggle Dataset Download**:
   - Link: https://www.kaggle.com/competitions/physionet-ecg-image-digitization/data
   - Note: Requires competition participation to download


### Data Directory Structure

**Downloaded data should be placed in the following structure:**

Training data should be organized as:
```
data/
   ├── train/        # Training data
   │ ├── images/     # ECG images (.png)
   │ └── *.csv       # ECG time series data
   ├── test/         # Test data
   │ └── images/     # Test images
   ├── train.csv     # Metadata file
   └── test.csv      # Metadata file
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


## 🏗️ Model Architecture

### Stage 0: Image Normalization & Orientation

- **Backbone**: ResNet-18D with custom U-Net decoder
- **Output**: 8-class orientation + lead marker segmentation
- **Purpose**: Standardize ECG image orientation for consistent processing

### Stage 1: Grid Detection & Rectification

- **Backbone**: ResNet-34 with coordinate-aware decoder
- **Output**: Grid point coordinates + line classification
- **Purpose**: Detect and correct image distortion

### Stage 2: Signal Digitization

- **Backbone**: ResNet-34 with coordinate-enhanced decoder
- **Output**: 4-channel pixel segmentation → time series
- **Purpose**: Extract digital signals from rectified images

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

## 📊 Results & Performance

### Evaluation Metrics

| Metric | Value | Description |
| --- | --- | --- |
| **SNR** | 16.12 dB | Signal-to-noise ratio |
| **MAE** | 0.065 mV | Mean absolute error |
| **Inference Time** | ~2s/image | End-to-end processing |

### Detailed Analysis

#### Signal Quality by Lead
![SNR by Lead](./docs/overall-data/summary-snr-by-lead.png)

*SNR performance across different ECG leads, showing consistent signal quality*


### Key Findings

- Successfully digitizes ECG images with preserved clinical features
- Robust to variations in image quality and orientation
- Computationally efficient for potential clinical deployment



## 📚 Documentation (文档)

### Language Selection (语言选择)
- **[English Version](README.md)** - For international users
- **[中文版本](README_CN.md)** - 面向中文用户

## 🔬 Ablation Studies

Comprehensive ablation studies demonstrate:

- **Backbone selection**: ResNet-34 optimal for accuracy/speed balance
- **Multi-stage design**: Each stage contributes significantly to final accuracy
- **Coordinate awareness**: Improves grid detection by 15% vs baseline

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **PhysioNet & Kaggle** for the ECG digitization competition and dataset
- **PyTorch team** for the deep learning framework
- Our institution and advisors for academic support

---

*This is an academic research project. Clinical applications require proper validation.*
