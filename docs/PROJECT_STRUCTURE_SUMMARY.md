# 项目结构总结

本文档总结 ECG 数字化项目的文件组织结构，特别是主入口脚本的位置。

## 📁 根目录文件

### 主要入口脚本

| 文件 | 描述 | 用法 |
|------|------|------|
| **[`main.py`](../main.py)** | **主入口脚本** | `python main.py inference --config configs/inference_config.yaml --input ecg.jpg` |
| [`train.py`](../train.py) | 训练入口 | `python train.py` |
| [`inference.py`](../inference.py) | 推理入口 | `python inference.py --checkpoint outputs/stage0_final.pth --image ecg.jpg` |
| [`test.py`](../test.py) | 测试脚本 | `python test.py` |
| [`ablation.py`](../ablation.py) | 消融研究 | `python ablation.py` |

### 配置文件

| 文件 | 描述 |
|------|------|
| [`configs/base.yaml`](../configs/base.yaml) | 基础配置 |
| [`configs/inference_config.yaml`](../configs/inference_config.yaml) | 推理配置 |
| [`configs/stage0_config.yaml`](../configs/stage0_config.yaml) | Stage 0 配置 |
| [`configs/stage1_config.yaml`](../configs/stage1_config.yaml) | Stage 1 配置 |
| [`configs/stage2_config.yaml`](../configs/stage2_config.yaml) | Stage 2 配置 |

## 📁 scripts 目录

### 分阶段训练脚本

| 文件 | 描述 | 用法 |
|------|------|------|
| [`train_stage0.py`](../scripts/train_stage0.py) | Stage 0 训练 | `python scripts/train_stage0.py` |
| [`train_stage1.py`](../scripts/train_stage1.py) | Stage 1 训练 | `python scripts/train_stage1.py` |
| [`train_stage2.py`](../scripts/train_stage2.py) | Stage 2 训练 | `python scripts/train_stage2.py` |
| [`train_all_stages.py`](../scripts/train_all_stages.py) | 全阶段训练 | `python scripts/train_all_stages.py` |

### 专用工具脚本

| 文件 | 描述 | 用法 |
|------|------|------|
| [`main.py`](../scripts/main.py) | 主入口脚本（原版） | `python scripts/main.py inference --config ...` |
| [`main_simple.py`](../scripts/main_simple.py) | 简化主入口 | `python scripts/main_simple.py inference --config ...` |
| [`load_model.py`](../scripts/load_model.py) | 模型加载和推理 | `python scripts/load_model.py --checkpoint ...` |
| [`test_stages.py`](../scripts/test_stages.py) | 测试训练脚本 | `python scripts/test_stages.py` |
| [`check_dependencies.py`](../scripts/check_dependencies.py) | 依赖检查 | `python scripts/check_dependencies.py` |
| [`debug_imports.py`](../scripts/debug_imports.py) | 导入调试 | `python scripts/debug_imports.py` |
| [`visualization_demo.py`](../scripts/visualization_demo.py) | 可视化演示 | `python scripts/visualization_demo.py` |

## 🔄 文件移动说明

### 原始结构
```
ECG-Digitization-Project/
├── scripts/
│   └── main.py          # 主入口脚本
├── train.py
├── inference.py
└── ...
```

### 现在的结构
```
ECG-Digitization-Project/
├── main.py              # 主入口脚本（移动到根目录）
├── scripts/
│   ├── main.py          # 原主入口脚本（保留）
│   ├── main_simple.py    # 简化版本
│   └── ...
├── train.py
├── inference.py
└── ...
```

## 🎯 使用建议

### 推荐的推理命令优先级：

1. **使用根目录的 main.py（最推荐）**：
   ```bash
   python main.py inference --config configs/inference_config.yaml --input ecg.jpg
   ```

2. **使用简化的推理脚本**：
   ```bash
   python scripts/main_simple.py inference --config configs/inference_config.yaml --input ecg.jpg
   ```

3. **直接使用 inference.py**：
   ```bash
   python inference.py --checkpoint outputs/stage0_final.pth --image ecg.jpg
   ```

### 训练命令优先级：

1. **使用统一的训练脚本**：
   ```bash
   python train.py
   ```

2. **使用分阶段训练脚本**：
   ```bash
   python scripts/train_all_stages.py
   ```

3. **单独训练特定阶段**：
   ```bash
   python scripts/train_stage0.py
   ```

## 🔧 脚本差异

### main.py vs scripts/main.py

- **位置**: `main.py` 在根目录，`scripts/main.py` 在 `scripts/` 目录
- **路径处理**: 两个版本都正确处理了 Python 路径
- **功能**: 两个版本提供相同的功能
- **依赖**: 两个版本都依赖相同的模块

### main.py vs main_simple.py

- **复杂度**: `main.py` 更完整，`main_simple.py` 更简化
- **参数**: `main_simple.py` 参数更简单直观
- **错误处理**: `main_simple.py` 有更好的错误处理
- **推荐**: 日常使用推荐 `main_simple.py`

## 📋 维护说明

### 文件同步

- 两个 `main.py` 文件应该保持功能同步
- 如果修改一个，应该考虑是否需要同步修改另一个
- `main_simple.py` 是独立开发的，不需要同步

### 删除选项

如果不需要重复的脚本，可以考虑：
- 保留根目录的 `main.py`
- 保留 `scripts/main_simple.py` 作为备用
- 删除 `scripts/main.py` 以减少混淆

---

*本文档随项目更新而维护，如有问题请提交 Issue。*