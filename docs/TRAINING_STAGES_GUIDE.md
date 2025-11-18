# ECG 数字化项目 - 分阶段训练指南

本文档说明如何使用 ECG 数字化项目的三个训练脚本进行分阶段模型训练。

## 📋 目录

- [训练脚本概述](#训练脚本概述)
- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
- [配置说明](#配置说明)
- [输出和检查点](#输出和检查点)
- [常见问题](#常见问题)

---

## 🎯 训练脚本概述

本项目提供三个独立的训练脚本，对应 ECG 数字化的三个阶段：

| 阶段 | 脚本名称 | 功能描述 | 输入 | 输出 |
|------|----------|----------|------|------|
| **Stage 0** | `train_stage0.py` | 图像标准化和关键点检测 | 原始 ECG 图像 | 14类标记点 + 8类方向 |
| **Stage 1** | `train_stage1.py` | 图像矫正和网格对齐 | Stage 0 输出 + 原图 | 网格检测 + 矫正图像 |
| **Stage 2** | `train_stage2.py` | 信号数字化和时间序列提取 | Stage 1 输出 + 矫正图像 | 数字化 ECG 信号 |

### 脚本特点

- ✅ **独立运行**: 每个脚本可以独立训练
- ✅ **配置驱动**: 通过配置文件控制训练参数
- ✅ **自动保存**: 定期保存训练检查点
- ✅ **错误处理**: 完善的异常处理和日志输出
- ✅ **进度监控**: 实时显示训练进度和损失

---

## 🚀 快速开始

### 1. 单阶段训练

```bash
# 训练 Stage 0
python scripts/train_stage0.py

# 训练 Stage 1
python scripts/train_stage1.py

# 训练 Stage 2
python scripts/train_stage2.py
```

### 2. 全阶段训练

```bash
# 训练所有三个阶段
python scripts/train_all_stages.py

# 训练指定阶段
python scripts/train_all_stages.py --stages 0 1

# 遇到错误时停止
python scripts/train_all_stages.py --stages all --stop-on-error
```

### 3. 测试脚本

```bash
# 测试所有训练脚本是否正常工作
python scripts/test_stages.py
```

---

## 📖 详细使用说明

### Stage 0 - 图像标准化和关键点检测

**功能**: 检测 ECG 图像中的关键解剖标记点和图像方向

**配置**:
```python
config = {
    'TRAIN': {
        'BATCH_SIZE': 2,
        'EPOCHS': 5,
        'LEARNING_RATE': 1e-4
    },
    'MODEL': {
        'BACKBONE': {
            'NAME': 'resnet18',  # 支持的模型
            'PRETRAINED': False
        }
    }
}
```

**输出**:
- 检查点: `./outputs/stage0_checkpoints/`
- 模型文件: `stage0_epoch_{epoch}.pth`
- 最终模型: `stage0_final.pth`

### Stage 1 - 图像矫正和网格检测

**功能**: 矫正倾斜的 ECG 图像并检测网格线

**配置**:
```python
config = {
    'TRAIN': {
        'BATCH_SIZE': 2,
        'EPOCHS': 5,
        'LEARNING_RATE': 1e-4
    },
    'MODEL': {
        'BACKBONE': {
            'NAME': 'resnet34',  # 使用支持的模型
            'PRETRAINED': False
        },
        'GRID_CONFIG': {
            'H_LINES': 44,
            'V_LINES': 57,
            'MV_TO_PIXEL': 79.0
        }
    }
}
```

**输出**:
- 检查点: `./outputs/stage1_checkpoints/`
- 模型文件: `stage1_epoch_{epoch}.pth`

### Stage 2 - 信号数字化和提取

**功能**: 从矫正后的图像中提取数字化 ECG 信号

**配置**:
```python
config = {
    'TRAIN': {
        'BATCH_SIZE': 1,  # 信号处理使用较小批次
        'EPOCHS': 5,
        'LEARNING_RATE': 1e-4
    },
    'MODEL': {
        'CROP_SIZE': [1696, 2176],  # 更大的裁剪尺寸
        'BACKBONE': {
            'NAME': 'resnet34',
            'PRETRAINED': False
        },
        'SIGNAL_CONFIG': {
            'NUM_LEADS': 12,
            'SAMPLING_RATE': 500,
            'TIME_WINDOW': 10
        }
    }
}
```

**输出**:
- 检查点: `./outputs/stage2_checkpoints/`
- 模型文件: `stage2_epoch_{epoch}.pth`

---

## ⚙️ 配置说明

### 设备配置

```python
'DEVICE': {
    'DEVICE': 'cuda',  # 或 'cpu'
    'NUM_WORKERS': 0   # 数据加载进程数
}
```

### 训练配置

```python
'TRAIN': {
    'BATCH_SIZE': 2,        # 批次大小
    'EPOCHS': 5,           # 训练轮数
    'LEARNING_RATE': 1e-4, # 学习率
    'GRADIENT_CLIP': 1.0   # 梯度裁剪
}
```

### 模型配置

#### Stage 0 和 Stage 1 支持的主干网络:
- `resnet34`
- `resnet50`
- `convnext_tiny_in22k`
- `convnext_small.fb_in22k`
- `convnext_base.fb_in22k`

### 检查点配置

```python
'CHECKPOINT': {
    'SAVE_DIR': './outputs/stage{X}_checkpoints',
    'SAVE_INTERVAL': 1  # 每 N 个轮次保存一次
}
```

---

## 📁 输出和检查点

### 检查点结构

每个训练脚本保存的检查点包含:

```python
checkpoint = {
    'epoch': current_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': average_loss,
    'config': training_config
}
```

### 输出目录结构

```
outputs/
├── stage0_checkpoints/
│   ├── stage0_epoch_1.pth
│   ├── stage0_epoch_2.pth
│   └── stage0_final.pth
├── stage1_checkpoints/
│   ├── stage1_epoch_1.pth
│   └── stage1_final.pth
└── stage2_checkpoints/
    ├── stage2_epoch_1.pth
    └── stage2_final.pth
```

### 模型加载

```python
# 加载训练好的模型
checkpoint = torch.load('outputs/stage0_checkpoints/stage0_final.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 🔧 常见问题

### Q1: 训练脚本报错 "No module named 'xxx'"

**A**: 确保在项目根目录运行脚本，并且已安装所有依赖:

```bash
cd ECG-Digitization-Project
pip install -r requirements.txt
```

### Q2: CUDA 内存不足

**A**: 减少批次大小或使用 CPU 训练:

```python
# 修改配置中的批次大小
'TRAIN': {
    'BATCH_SIZE': 1  # 减小批次大小
}

# 或使用 CPU
'DEVICE': {
    'DEVICE': 'cpu'
}
```

### Q3: 数据目录未找到

**A**: 确保数据目录存在:

```bash
# 检查数据目录
ls ../ecg_data/physionet-ecg-image-digitization

# 或修改数据路径
'COMPETITION': {
    'KAGGLE_DIR': '/path/to/your/data'
}
```

### Q4: 模型不支持的架构

**A**: 检查每个阶段支持的模型名称:

- **Stage 0**: 支持常见的 ResNet 系列
- **Stage 1/2**: 使用 TIMM 支持的模型名称，如 `resnet34`

### Q5: 训练速度很慢

**A**: 优化建议:

1. 使用 GPU 训练
2. 启用混合精度训练
3. 增加 `NUM_WORKERS`
4. 使用预训练权重

```python
'DEVICE': {
    'DEVICE': 'cuda',
    'MIXED_PRECISION': True,
    'NUM_WORKERS': 4
}
```

---

## 💡 最佳实践

### 1. 训练顺序

建议按顺序训练各个阶段:
1. 先训练 Stage 0 (关键点检测)
2. 再训练 Stage 1 (图像矫正)
3. 最后训练 Stage 2 (信号提取)

### 2. 参数调优

```python
# 初期训练
'EPOCHS': 5,
'BATCH_SIZE': 2,
'LEARNING_RATE': 1e-4

# 深度训练
'EPOCHS': 50,
'BATCH_SIZE': 4,
'LEARNING_RATE': 5e-5
```

### 3. 监控训练

```bash
# 使用 screen 或 tmux 运行长时间训练
screen -S ecg_stage0
python scripts/train_stage0.py

# 分离会话
Ctrl+A, D

# 重新连接
screen -r ecg_stage0
```

---

## 📚 相关文档

- [项目概述](../PROJECT_SUMMARY.md)
- [高级训练功能](TRAINING_FEATURES.md)
- [消融研究指南](ABLATION_GUIDE.md)

---

*如有其他问题，请查看项目的 GitHub Issues 或联系开发团队。*