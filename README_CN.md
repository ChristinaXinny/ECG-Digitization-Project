# ECG 数字化项目

一个全面的、生产就绪的 ECG 图像到数字信号转换的深度学习实现。

## 🚀 项目概述

本项目通过三阶段深度学习网络将 ECG 图像转换为数字信号：

1. **阶段 0**: 图像标准化和关键点检测
2. **阶段 1**: 图像矫正和网格对齐
3. **阶段 2**: 信号数字化和时间序列提取

## 📁 项目结构

```
ECG-Digitization-Project/
├── 📄 快速访问脚本
│   ├── train.py                      # 快速训练访问
│   ├── test.py                       # 快速测试访问
│   ├── inference.py                  # 快速推理访问
│   └── ablation.py                   # 快速消融研究访问
│
├── 📁 核心目录
│   ├── configs/                      # 配置文件
│   ├── data/                         # 数据处理（数据集、预处理、变换）
│   ├── models/                       # 模型定义和头
│   ├── engines/                      # 训练/推理引擎
│   ├── utils/                        # 工具函数（指标、日志）
│   └── ablation_studies/             # 消融研究框架
│
├── 📁 脚本和工具
│   ├── scripts/                      # 所有训练和工具脚本
│   │   ├── train_stage0.py           # 主训练脚本
│   │   ├── load_model.py             # 模型加载和推理
│   │   ├── main.py                   # 入口点脚本
│   │   └── *.py                      # 其他训练脚本
│   └── tests/                        # 全面测试套件
│       ├── run_simple_tests.py       # 快速验证测试
│       ├── basic_test.py             # 基础功能测试
│       └── test_*.py                 # 专业测试文件
│
├── 📁 文档
│   ├── README.md                     # 主文档（英文）
│   ├── README_CN.md                  # 主文档（中文）
│   ├── PROJECT_SUMMARY.md            # 项目概述
│   ├── ABLATION_GUIDE.md             # 消融研究指南
│   └── docs/                         # 其他文档
│
└── 📁 输出和构建
    ├── outputs/                      # 训练输出和检查点
    ├── requirements.txt              # Python依赖
    ├── Makefile                      # 构建自动化
    └── setup.py                      # 包设置
```

## 🛠️ 安装

### 先决条件

- Python 3.8 或更高版本
- CUDA 兼容的 GPU（推荐用于训练）
- 8GB+ RAM（推荐 16GB+ 用于训练）

### 环境设置

1. **克隆仓库**：
   ```bash
   git clone <repository-url>
   cd ECG-Digitization-Project
   ```

2. **创建虚拟环境**：
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 快速开始

### 1. 快速验证
```bash
# 运行快速测试验证安装
python test.py

# 或运行完整测试
python tests/run_simple_tests.py
```

### 2. 训练
```bash
# 从项目根目录开始训练
python train.py

# 分阶段训练 - 推荐
python scripts/train_stage0.py    # 阶段0: 图像标准化和关键点检测
python scripts/train_stage1.py    # 阶段1: 图像矫正和网格检测
python scripts/train_stage2.py    # 阶段2: 信号数字化和提取

# 全阶段训练
python scripts/train_all_stages.py

# 测试训练脚本
python scripts/test_stages.py
```

### 3. 推理
```bash
# 使用训练好的模型运行推理（推荐）
python main.py inference --config configs/inference_config.yaml --input ecg.jpg

# 或使用简化版本
python inference.py --checkpoint outputs/stage0_final.pth --image ecg.jpg

# 使用分阶段推理脚本
python scripts/main_simple.py inference --config configs/inference_config.yaml --input ecg.jpg
```

### 4. 消融研究
```bash
# 运行所有消融研究
python ablation.py

# 或运行特定研究
python ablation_studies/run_ablation_studies.py --studies backbone loss
```

## 📋 详细使用说明

### 数据准备

准备你的 ECG 数据：

```
ecg_data/
├── train/                           # 训练数据
│   ├── ecg_001.jpg                  # ECG图像
│   ├── ecg_001_npy.png             # 对应标签
│   └── ...
└── val/                            # 验证数据
    ├── ecg_101.jpg
    ├── ecg_101_npy.png
    └── ...
```

### 训练配置

编辑配置文件 `configs/stage0_config.yaml`：

```yaml
MODEL:
  BACKBONE:
    NAME: resnet18
    PRETRAINED: false
  NUM_MARKER_CLASSES: 14
  NUM_ORIENTATION_CLASSES: 8

TRAIN:
  BATCH_SIZE: 4
  LEARNING_RATE: 1e-4
  EPOCHS: 100
  CHECKPOINT_DIR: ./outputs/checkpoints

DATA:
  DATA_ROOT: ./ecg_data
  TRAIN_SPLIT: 0.8
  BATCH_SIZE: 4
```

### 模型推理

```bash
# 基本推理
python inference.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --image path/to/ecg.jpg \
    --output outputs/inference_results/

# 指定设备
python inference.py \
    --checkpoint model.pth \
    --image ecg.jpg \
    --device cuda
```

### 消融研究

运行系统性的消融研究来证明每个组件的必要性：

```bash
# 运行所有研究
python ablation.py

# 运行特定研究
python ablation.py --studies backbone loss

# 直接运行研究框架
python ablation_studies/run_ablation_studies.py \
    --studies backbone module \
    --verbose
```

## 🔬 消融研究

本项目包含全面的消融研究框架，用于评估每个组件的贡献：

### 支持的研究类型

1. **主干网络消融** (`backbone_ablation.py`)
   - ResNet 系列比较
   - EfficientNet vs MobileNet
   - Vision Transformer 测试

2. **损失函数消融** (`loss_ablation.py`)
   - 不同权重组合
   - 高级损失函数
   - 标签平滑效果

3. **模块消融** (`module_ablation.py`)
   - 解码器重要性
   - 注意力机制影响
   - 归一化层选择

4. **数据增强消融** (`data_augmentation_ablation.py`)
   - 医学图像特定增强
   - 几何变换策略
   - 增强强度优化

### 结果分析

消融研究将生成：
- 📊 性能对比表格
- 📈 组件贡献度可视化
- 📋 详细分析报告
- 🔬 科学证据支持

## 📊 数据集与数据配置

### 数据下载来源

**完整数据下载链接：**

1. **Kaggle 官方下载**:
   - 链接：https://www.kaggle.com/competitions/physionet-ecg-image-digitization/data
   - 注意：需要参加比赛才可下载

2. **百度网盘下载**:
   - 链接：[请在此处填入百度网盘链接]
   - 提取码：[请在此处填入提取码]
   - 说明：比赛官方下载的数据备份

### 数据目录结构

**下载的数据需要放置在以下结构中：**

训练数据应按以下结构组织：
```
ECG-Digitization-Project/
└── data/                        # 数据根目录
    ├── ecg_data_simple/         # 简单测试数据（用于快速验证）
    │   ├── train/
    │   │   ├── images/         # ECG图像文件
    │   │   └── series/         # 对应的时间序列数据
    │   └── test/
    │       ├── images/         # 测试图像
    │       └── series/         # 测试时间序列
    │
    ├── train/                   # 完整训练数据（从下载的数据集）
    │   ├── images/              # ECG图像文件 (.png, .jpg)
    │   ├── series/              # 对应的CSV时间序列数据
    │   └── annotations/         # 标注文件（可选）
    │
    ├── val/                     # 验证数据
    │   ├── images/
    │   └── series/
    │
    └── test/                    # 测试数据（从下载的数据集）
        ├── images/
        └── series/
```

**数据放置说明：**
1. 解压下载的比赛数据集
2. 将文件复制/解压到项目根目录的 `data/` 目录中
3. 确保目录结构与上述格式匹配
4. `ecg_data_simple/` 文件夹已存在，包含测试样本

### 快速测试数据

**简单测试数据路径**: `ECG-Digitization-Project\data\ecg_data_simple`

此目录包含少量测试样本，用于：
- 快速验证环境安装
- 测试推理功能
- 调试模型性能

**示例测试命令**:
```bash
# 使用简单测试数据进行推理
python main.py inference --config configs/inference_config.yaml --input data/ecg_data_simple/test/1053922973.png
```

### 数据格式要求

- **图像**: PNG、JPG、JPEG格式，RGB色彩
- **序列**: CSV格式，包含时间和电压值
- **文件命名**: 建议使用一致的命名规则（如ID编号）

## 📊 模型架构

### Stage0Net 模型特点

- **参数量**: 14,392,662
- **输入**: 3通道 ECG 图像
- **输出**:
  - 14类标记点检测
  - 8类方向分类
- **架构**: U-Net + ResNet 主干
- **特性**: 多任务学习 + 注意力机制

### 性能指标

| 指标 | 数值 |
|------|------|
| 模型参数 | 14.4M |
| 训练内存需求 | 2-4GB (CPU), 4-8GB (GPU) |
| 推理时间 | ~50-100ms (CPU), ~10-20ms (GPU) |
| 检查点大小 | ~58MB |

## 🧪 测试套件

项目包含全面的测试套件：

### 快速测试
```bash
python test.py
```

### 完整测试
```bash
# 运行所有测试
python tests/run_tests.py

# 运行特定测试
python tests/run_tests.py --tests TestStage0Model
```

### 测试覆盖范围
- ✅ 模型导入和创建 (100%)
- ✅ 前向传播 (100%)
- ✅ 检查点操作 (100%)
- ✅ 模型组件 (100%)
- ✅ 训练流水线 (100%)
- ✅ 推理引擎 (100%)

## 🎯 使用场景

### 1. 医疗机构
- **ECG 数字化**: 将纸质 ECG 转换为数字信号
- **自动化处理**: 批量处理 ECG 图像
- **质量保证**: 高精度检测和分类

### 2. 研究
- **学术研究**: 消融研究验证模型组件
- **算法开发**: 测试新架构和损失函数
- **基准测试**: 与现有方法比较

### 3. 生产部署
- **云服务**: 部署为推理服务
- **边缘计算**: 移动端优化
- **批处理**: 大规模 ECG 处理

## 🔧 高级功能

### 自定义训练

```python
from engines.trainer import Trainer
from utils.config import load_config

# 加载配置
config = load_config('configs/custom_config.yaml')

# 创建训练器
trainer = Trainer(config)

# 训练模型
trainer.train()
```

### 自定义推理

```python
from engines.inference import InferenceEngine

# 创建推理引擎
engine = InferenceEngine(config)
engine.load_checkpoint('model.pth')

# 批量推理
results = engine.predict_batch(image_list)
```

### 自定义消融研究

```python
from ablation_studies.base_ablation import BaseAblationStudy

class CustomAblation(BaseAblationStudy):
    def get_experiments(self):
        return [
            ('baseline', {'MODEL.BACKBONE.NAME': 'resnet18'}),
            ('large_model', {'MODEL.BACKBONE.NAME': 'resnet50'})
        ]

# 运行研究
ablation = CustomAblation('custom_study')
ablation.run_study()
```

## 📈 性能优化

### 训练优化
- **批量大小**: 根据 GPU 内存调整
- **学习率调度**: 使用余弦退火
- **数据并行**: 多GPU训练支持
- **梯度累积**: 模拟大批量训练

### 推理优化
- **模型量化**: 减少模型大小
- **ONNX导出**: 跨平台部署
- **批处理**: 提高吞吐量
- **缓存优化**: 减少重复计算

## 🤝 贡献指南

我们欢迎社区贡献！

### 贡献类型
1. **新功能**: 添加新的模型或算法
2. **性能优化**: 提高训练或推理速度
3. **文档改进**: 完善文档和示例
4. **Bug修复**: 解决已知问题
5. **测试增强**: 提高测试覆盖率

### 开发流程
1. Fork 仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 开发环境
```bash
# 安装开发依赖
pip install -r requirements.txt
pip install pytest black flake8 mypy

# 运行测试
python -m pytest tests/

# 代码格式化
black .
flake8 .
```

## 📄 文档

- [项目概述](PROJECT_SUMMARY.md) - 完整的项目介绍
- [分阶段训练指南](docs/TRAINING_STAGES_GUIDE.md) - Stage 0/1/2 训练脚本使用说明
- [消融研究指南](ABLATION_GUIDE.md) - 详细的消融研究说明
- [项目结构指南](PROJECT_STRUCTURE.md) - 文件组织说明
- [高级训练功能](docs/TRAINING_FEATURES.md) - 学习率调度、混合精度等高级功能
- [Git忽略指南](docs/GITIGNORE_GUIDE.md) - .gitignore 使用指南

## 🐛 故障排除

### 常见问题

1. **内存不足**
   ```bash
   # 减少批量大小
   python train.py --batch-size 2
   ```

2. **CUDA错误**
   ```bash
   # 强制使用CPU
   CUDA_VISIBLE_DEVICES="" python train.py
   ```

3. **依赖问题**
   ```bash
   # 重新安装依赖
   pip install -r requirements.txt --force-reinstall
   ```

4. **路径问题**
   ```bash
   # 检查项目结构
   python test.py
   ```

### 调试模式

```bash
# 启用详细日志
python train.py --verbose

# 运行单个测试
python tests/run_tests.py --tests TestStage0Model --verbose
```

## 📊 项目状态

- ✅ **模型实现**: 完整的三阶段架构
- ✅ **训练流水线**: 端到端训练支持
- ✅ **推理引擎**: 生产级推理实现
- ✅ **消融研究**: 系统性组件分析
- ✅ **测试套件**: 100% 测试覆盖率
- ✅ **文档完整**: 中英文文档齐全
- ✅ **结构优化**: 专业项目组织
- ✅ **Git配置**: 完善的版本控制

## 📞 获取帮助

如果你遇到问题或需要帮助：

1. **查看文档**: 阅读 [文档](docs/) 目录下的相关指南
2. **运行测试**: `python test.py` 验证安装
3. **检查状态**: `git status` 确认文件完整性
4. **提交问题**: 在仓库中创建 issue

## 🎉 致谢

感谢所有为这个项目做出贡献的开发者和研究人员！

---

**ECG 数字化项目** - 从图像到数字信号的专业深度学习解决方案 🚀