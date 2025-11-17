# 🚀 ECG数字化项目快速开始指南

## 📋 前置要求

1. **Python 3.7+** - 确保已安装Python 3.7或更高版本
2. **CUDA支持** - 如果有GPU，安装对应版本的CUDA (可选但推荐)
3. **数据准备** - 确保 `ecg_data/physionet-ecg-image-digitization` 目录存在

## 🎯 快速开始步骤

### 步骤 1: 环境设置

```bash
cd ECG-Digitization-Project

# 方法1: 使用自动安装脚本
python setup.py

# 方法2: 手动安装
pip install torch torchvision timm numpy pandas opencv-python loguru tqdm PyYAML
```

### 步骤 2: 验证数据加载

```bash
python simple_data_test.py
```

如果看到 `SUCCESS` 消息，说明数据加载正常！

### 步骤 3: 开始训练

#### 方法 A: 交互式训练 (推荐新手)
```bash
python simple_train.py
```
然后选择要训练的阶段：
- 1 = Stage 0 (图像标准化和关键点检测)
- 2 = Stage 1 (图像校正和网格检测)
- 3 = Stage 2 (信号数字化)

#### 方法 B: 命令行训练 (推荐高级用户)
```bash
# 训练Stage 0
python train.py --stage stage0 --epochs 10 --batch-size 4

# 训练Stage 1
python train.py --stage stage1 --epochs 10 --batch-size 2

# 训练Stage 2
python train.py --stage stage2 --epochs 10 --batch-size 1
```

## 📊 训练配置

### 默认快速训练配置
- **Epochs**: 3-5个 (快速测试)
- **Batch Size**: 2-4 (根据GPU内存调整)
- **Learning Rate**: 1e-4
- **预训练权重**: False (更快加载)

### 生产环境配置建议
- **Epochs**: 50-100个
- **Batch Size**: 8-16 (根据GPU内存)
- **Learning Rate**: 2e-4
- **预训练权重**: True

## 🎛️ 自定义配置

修改 `configs/` 目录下的YAML文件来调整训练参数：

```yaml
# configs/stage0_config.yaml
TRAIN:
  BATCH_SIZE: 8
  EPOCHS: 50
  LEARNING_RATE: 2e-4

MODEL:
  BACKBONE:
    PRETRAINED: true
```

## 📁 输出文件

训练过程中会自动创建以下目录：
```
outputs/
├── checkpoints/     # 模型权重
├── logs/           # 训练日志
└── visualizations/ # 可视化结果
```

## 🔧 故障排除

### 常见问题

1. **内存不足错误**
   ```bash
   # 减少batch size
   python simple_train.py  # 选择1，然后在代码中修改batch_size=1
   ```

2. **导入错误**
   ```bash
   # 确保安装了所有依赖
   pip install -r requirements_minimal.txt
   ```

3. **数据路径错误**
   ```bash
   # 检查数据目录是否存在
   ls ../ecg_data/physionet-ecg-image-digitization
   ```

4. **CUDA错误**
   ```bash
   # 使用CPU训练 (速度较慢)
   export CUDA_VISIBLE_DEVICES=""
   ```

### 性能优化建议

1. **GPU使用**
   - 使用 NVIDIA GPU 加速训练
   - 安装对应版本的CUDA和cuDNN

2. **数据加载优化**
   ```bash
   # 增加数据加载进程数
   # 在配置中设置 NUM_WORKERS: 4
   ```

3. **混合精度训练**
   ```yaml
   DEVICE:
     MIXED_PRECISION: true
     AMP_ENABLED: true
   ```

## 🎯 训练建议

### 训练顺序
1. **先训练Stage 0** - 图像标准化是基础
2. **再训练Stage 1** - 网格检测和校正
3. **最后训练Stage 2** - 信号提取

### 快速验证流程
```bash
# 1. 测试数据加载
python simple_data_test.py

# 2. 快速训练测试 (3 epochs)
python simple_train.py
# 选择 1 进行Stage 0快速训练

# 3. 检查结果
ls outputs/stage0_checkpoints/
```

## 📈 监控训练

训练过程中会显示：
- 每个epoch的损失值
- 训练进度条
- 模型保存状态
- 错误信息(如有)

## 🎉 训练完成后

1. **检查模型权重**
   ```bash
   ls outputs/stage0_checkpoints/
   ```

2. **运行推理测试**
   ```bash
   python inference.py --model outputs/stage0_checkpoints/best_checkpoint.pth
   ```

3. **查看训练日志**
   ```bash
   tail -f outputs/logs/training.log
   ```

## 📚 更多信息

- **详细文档**: `docs/` 目录
- **API参考**: `docs/api_reference.md`
- **架构说明**: `docs/architecture.md`

---

🚀 **现在就开始训练您的ECG模型吧！**