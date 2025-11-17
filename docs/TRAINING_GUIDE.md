# ECG Digitization - 训练指南

## 概述

这个项目使用三阶段架构来数字化ECG图像：
- **Stage 0**: 图像归一化和关键点检测
- **Stage 1**: 图像校正和网格检测
- **Stage 2**: 信号提取

## 快速开始

### 1. 环境准备

确保已安装必要的包：
```bash
pip install torch torchvision timm numpy pandas opencv-python loguru
```

### 2. 数据准备

确保ECG数据位于 `../ecg_data/physionet-ecg-image-digitization/` 目录下。

### 3. 开始训练

#### 方法1：使用简单训练脚本（推荐）
```bash
python train_stage0.py
```

#### 方法2：使用主训练脚本
```bash
python start_training.py
```

## 训练脚本功能

### `train_stage0.py`

直接训练脚本，不需要交互式输入，包含完整的检查点保存功能：

**特性：**
- ✅ 自动创建检查点目录
- ✅ 每个epoch后保存检查点
- ✅ 保存最佳模型（基于loss）
- ✅ 保存最终模型
- ✅ 详细的训练日志

**输出文件：**
```
./outputs/stage0_checkpoints/
├── stage0_epoch_1.pth      # 第1个epoch检查点
├── stage0_epoch_2.pth      # 第2个epoch检查点
├── ...
├── stage0_best.pth         # 最佳模型
└── stage0_final.pth        # 最终模型
```

### `start_training.py`

更完整的训练脚本，支持所有阶段：

```bash
# 训练Stage 0
python start_training.py --stage 0

# 训练Stage 1
python start_training.py --stage 1

# 训练Stage 2
python start_training.py --stage 2

# 训练所有阶段
python start_training.py --stage all
```

## 模型检查点管理

### 检查训练状态
```bash
python check_training.py
```

### 检查特定检查点
```bash
python check_training.py --test ./outputs/stage0_checkpoints/stage0_best.pth
```

### 列出所有可用模型
```bash
python check_training.py --list
```

## 模型推理

### 使用训练好的模型
```bash
python load_model.py --checkpoint ./outputs/stage0_checkpoints/stage0_best.pth --image path/to/ecg/image.jpg --output ./outputs/inference/
```

**输出：**
- `marker_prediction.jpg` - 关键点检测可视化
- 控制台输出：方向分类结果和置信度

### 在Python代码中使用
```python
from load_model import ECGModelLoader

# 加载模型
loader = ECGModelLoader("./outputs/stage0_checkpoints/stage0_best.pth", "cuda")
model, config = loader.load_model()

# 运行推理
results = loader.inference("path/to/ecg/image.jpg")
print(f"Orientation: {results['orientation_class']}")
print(f"Confidence: {results['orientation_confidence']}")
```

## 配置参数

### 默认配置
```python
config = {
    'TRAIN': {
        'BATCH_SIZE': 2,
        'EPOCHS': 5,
        'LEARNING_RATE': 1e-4
    },
    'MODEL': {
        'INPUT_SIZE': [1152, 1440],
        'BACKBONE': {
            'NAME': 'resnet18',
            'PRETRAINED': False
        }
    },
    'DEVICE': {
        'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
}
```

### 修改配置
编辑 `train_stage0.py` 或 `start_training.py` 中的配置参数来调整训练设置。

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少 `BATCH_SIZE`
   - 使用 `DEVICE': 'cpu'`

2. **数据加载慢**
   - 数据位于错误的目录
   - 检查 `../ecg_data/physionet-ecg-image-digitization/` 是否存在

3. **模型加载失败**
   - 检查检查点文件是否损坏
   - 确保模型架构匹配

### 调试模式
在训练脚本中设置：
```python
config['DEBUG'] = {'ENABLED': True}
```

## 输出说明

### 训练日志
```
[HH:MM:SS] Testing imports...
[HH:MM:SS] data.dataset imported successfully
[HH:MM:SS] models imported successfully
[HH:MM:SS] Basic imports successful
[HH:MM:SS] Using simple training loop (bypassing engines module)...
[HH:MM:SS] Created dataset...
[HH:MM:SS] Loaded 450 training samples
[HH:MM:SS] Creating model...
[HH:MM:SS] Model created with 14,392,662 parameters
[HH:MM:SS] Created checkpoint directory: ./outputs/stage0_checkpoints/
[HH:MM:SS] Epoch 1/5
[HH:MM:SS]   Batch 0/225, Loss: 4.7123
...
[HH:MM:SS] Epoch 1 completed, Average Loss: 4.5678
[HH:MM:SS] Checkpoint saved: ./outputs/stage0_checkpoints/stage0_epoch_1.pth
[HH:MM:SS] New best model saved with loss: 4.5678
```

### 检查点文件内容
每个 `.pth` 文件包含：
- `epoch`: 训练轮数
- `model_state_dict`: 模型权重
- `optimizer_state_dict`: 优化器状态
- `loss`: 该轮的平均损失
- `config`: 训练配置

## 下一步

1. **完成Stage 0训练**后，可以继续训练Stage 1和Stage 2
2. **调优超参数**以获得更好的性能
3. **使用验证集**评估模型性能
4. **部署模型**到生产环境

## 技术支持

如果遇到问题：
1. 检查错误日志
2. 使用 `check_training.py` 诊断检查点文件
3. 确保所有依赖包已正确安装
4. 检查数据路径和格式