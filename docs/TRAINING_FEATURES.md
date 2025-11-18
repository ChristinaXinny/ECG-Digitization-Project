# 高级训练功能文档

本文档详细说明了 ECG 数字化项目中实现的高级训练功能，包括学习率调度器、训练检查点、混合精度训练等技术特性。

## 📋 目录

- [学习率调度器](#学习率调度器)
- [训练检查点](#训练检查点)
- [混合精度训练](#混合精度训练)
- [分布式训练](#分布式训练)
- [功能使用指南](#功能使用指南)
- [配置示例](#配置示例)
- [性能优化建议](#性能优化建议)

---

## 🔄 学习率调度器 (Learning Rate Scheduler)

### 概述

项目实现了多种主流的学习率调度策略，支持配置化的学习率调整，以提高模型训练的收敛性和性能。

### 支持的调度器

| 调度器类型 | 实现状态 | 描述 | 适用场景 |
|------------|----------|------|----------|
| **CosineAnnealingLR** | ✅ 已实现 | 余弦退火学习率调度 | 长期训练，平滑衰减 |
| **StepLR** | ✅ 已实现 | 阶梯式学习率衰减 | 分阶段训练 |
| **ReduceLROnPlateau** | ✅ 已实现 | 基于验证指标的自适应调整 | 防止过拟合，自动调整 |
| **MultiStepLR** | ✅ 已实现 | 多里程碑式衰减 | 复杂训练策略 |
| **None** | ✅ 已实现 | 固定学习率 | 简单训练场景 |

### 实现位置

- **核心实现**: [`engines/base_trainer.py:_setup_scheduler()`](../engines/base_trainer.py:132-174)
- **配置文件**: [`configs/base.yaml:SCHEDULER`](../configs/base.yaml:31-34)
- **训练调用**: [`engines/base_trainer.py:271-310`](../engines/base_trainer.py:271-310)

### 使用示例

```yaml
# configs/base.yaml
TRAIN:
  SCHEDULER:
    NAME: "CosineAnnealingLR"
    MIN_LR: 1e-6
    WARMUP_EPOCHS: 5
```

```python
# 代码中使用
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=min_lr
)

# 训练循环中
if scheduler is not None:
    scheduler.step()
```

---

## 💾 训练检查点 (Training Checkpoints)

### 概述

项目实现了完整的训练检查点管理系统，支持模型状态、优化器状态、调度器状态和训练进度的保存与恢复。

### 功能特性

| 功能 | 实现状态 | 描述 |
|------|----------|------|
| **模型状态保存** | ✅ 已实现 | 保存模型权重和偏置 |
| **优化器状态** | ✅ 已实现 | 保存优化器的内部状态 |
| **调度器状态** | ✅ 已实现 | 保存学习率调度器状态 |
| **梯度缩放器状态** | ✅ 已实现 | 保存混合精度训练状态 |
| **训练进度** | ✅ 已实现 | 保存当前epoch、global_step等 |
| **最佳模型监控** | ✅ 已实现 | 基于验证指标保存最佳模型 |
| **配置保存** | ✅ 已实现 | 保存训练配置以供复现 |

### 实现位置

- **核心实现**: [`engines/base_trainer.py:404-474`](../engines/base_trainer.py:404-474)
- **配置文件**: [`configs/base.yaml:CHECKPOINT`](../configs/base.yaml:108-114)
- **依赖模块**: `utils.checkpoint.CheckpointManager` (需要实现)

### 检查点结构

```python
checkpoint_data = {
    'epoch': current_epoch,
    'global_step': global_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'best_metric': best_metric,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'config': config
}
```

### 使用示例

```yaml
# configs/base.yaml
CHECKPOINT:
  SAVE_DIR: "outputs/checkpoints"
  SAVE_TOP_K: 3
  SAVE_LAST: True
  MONITOR: "val_loss"
  MODE: "min"
```

```python
# 保存检查点
trainer.save_checkpoint(epoch=100, metrics={'val_loss': 0.123})

# 恢复训练
trainer = BaseTrainer(model, config, train_loader, val_loader,
                      resume_from="outputs/checkpoints/best_model.pth")
```

---

## ⚡ 混合精度训练 (Mixed Precision Training)

### 概述

项目支持自动混合精度训练（AMP），使用FP16进行前向传播和梯度计算，使用FP32进行权重更新，显著提升训练速度并减少显存占用。

### 技术原理

- **FP16前向传播**: 减少计算量和显存使用
- **梯度缩放**: 防止FP16下溢问题
- **FP32权重更新**: 保持数值稳定性
- **动态损失缩放**: 自适应调整缩放因子

### 功能特性

| 功能 | 实现状态 | 描述 |
|------|----------|------|
| **自动混合精度** | ✅ 已实现 | PyTorch AMP支持 |
| **梯度缩放** | ✅ 已实现 | GradScaler防止下溢 |
| **配置控制** | ✅ 已实现 | 可通过配置启用/禁用 |
| **设备兼容** | ✅ 已实现 | 自动检测CUDA支持 |

### 实现位置

- **初始化**: [`engines/base_trainer.py:53`](../engines/base_trainer.py:53)
- **训练步骤**: [`engines/base_trainer.py:220-229`](../engines/base_trainer.py:220-229)
- **配置文件**: [`configs/base.yaml:DEVICE`](../configs/base.yaml:10-16)

### 使用示例

```yaml
# 启用混合精度训练
DEVICE:
  MIXED_PRECISION: True
  AMP_ENABLED: True
```

```python
# 代码实现
self.scaler = torch.cuda.amp.GradScaler() if config.get('MIXED_PRECISION', True) else None

# 训练循环
if self.scaler:
    with torch.cuda.amp.autocast():
        loss, metrics = self._train_step(batch)

    self.optimizer.zero_grad()
    self.scaler.scale(loss).backward()
    self.scaler.step(self.optimizer)
    self.scaler.update()
else:
    # 标准精度训练
    loss, metrics = self._train_step(batch)
    loss.backward()
    self.optimizer.step()
```

### 性能优势

| 指标 | 标准精度 | 混合精度 | 提升幅度 |
|------|----------|----------|----------|
| **训练速度** | 基准 | +1.5-2.5x | 显著提升 |
| **显存占用** | 基准 | -30-50% | 大幅减少 |
| **数值稳定性** | 高 | 高 | 无损失 |

---

## 🌐 分布式训练 (Distributed Training)

### 当前状态

❌ **未实现** - 项目目前不支持分布式训练功能。

### 缺失功能

| 功能 | 状态 | 优先级 |
|------|------|--------|
| **DistributedDataParallel (DDP)** | ❌ 未实现 | 高 |
| **多GPU数据并行** | ❌ 未实现 | 高 |
| **分布式采样器** | ❌ 未实现 | 中 |
| **进程组初始化** | ❌ 未实现 | 高 |
| **节点间通信** | ❌ 未实现 | 中 |

### 实现建议

如需要添加分布式训练支持，建议实现以下组件：

1. **分布式初始化**
   ```python
   import torch.distributed as dist

   def setup_distributed(rank, world_size):
       dist.init_process_group(
           backend='nccl',
           init_method='env://',
           world_size=world_size,
           rank=rank
       )
   ```

2. **DDP模型包装**
   ```python
   model = torch.nn.parallel.DistributedDataParallel(
       model, device_ids=[local_rank]
   )
   ```

3. **分布式采样器**
   ```python
   sampler = torch.utils.data.distributed.DistributedSampler(
       dataset, num_replicas=world_size, rank=rank
   )
   ```

---

## 🛠️ 功能使用指南

### 快速启用高级功能

1. **配置学习率调度器**
   ```yaml
   TRAIN:
     SCHEDULER:
       NAME: "CosineAnnealingLR"
       MIN_LR: 1e-6
   ```

2. **启用混合精度训练**
   ```yaml
   DEVICE:
     MIXED_PRECISION: True
   ```

3. **配置检查点保存**
   ```yaml
   CHECKPOINT:
     SAVE_TOP_K: 3
     MONITOR: "val_loss"
   ```

### 训练命令示例

```bash
# 标准训练
python train.py --config configs/base.yaml

# 启用混合精度的训练
python train.py --config configs/base.yaml --mixed-precision

# 从检查点恢复训练
python train.py --config configs/base.yaml --resume outputs/checkpoints/latest.pth

# 多GPU训练（需要先实现分布式功能）
# python -m torch.distributed.launch --nproc_per_node=4 train.py
```

---

## 📊 性能优化建议

### 内存优化

1. **启用混合精度训练**
   - 减少显存占用 30-50%
   - 提升训练速度 1.5-2.5x

2. **梯度累积**
   ```python
   # 在配置中设置
   TRAIN:
     GRADIENT_ACCUMULATION_STEPS: 4
   ```

3. **批次大小调优**
   - 根据 GPU 显存调整 batch_size
   - 使用梯度累积模拟大批次训练

### 训练速度优化

1. **学习率调度策略选择**
   - **CosineAnnealingLR**: 适合长期训练
   - **ReduceLROnPlateau**: 适合自动调优
   - **StepLR**: 适合分阶段训练

2. **数据加载优化**
   ```yaml
   DEVICE:
     NUM_WORKERS: 4
     PIN_MEMORY: True
   ```

3. **检查点频率优化**
   - 频繁保存会影响训练速度
   - 建议每 5-10 个 epoch 保存一次

### 数值稳定性

1. **梯度裁剪**
   ```yaml
   TRAIN:
     GRADIENT_CLIP: 1.0
   ```

2. **权重初始化**
   - 使用适当的权重初始化策略
   - 考虑使用预训练权重

3. **学习率范围**
   - 初始学习率: 1e-4 到 1e-3
   - 最小学习率: 1e-6 到 1e-7

---

## 🔧 故障排除

### 常见问题

1. **混合精度训练报错**
   ```bash
   # 确保CUDA版本支持AMP
   python -c "import torch; print(torch.cuda.amp.is_available())"
   ```

2. **检查点加载失败**
   ```python
   # 检查检查点文件完整性
   import torch
   checkpoint = torch.load('path/to/checkpoint.pth')
   print(checkpoint.keys())
   ```

3. **学习率调度器不工作**
   ```python
   # 确保在训练循环中调用scheduler.step()
   scheduler.step()  # 在每个epoch结束后调用
   ```

### 性能监控

1. **显存监控**
   ```python
   import torch
   print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
   print(f"Cached: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
   ```

2. **训练速度监控**
   ```python
   import time
   start_time = time.time()
   # ... training step ...
   step_time = time.time() - start_time
   print(f"Step time: {step_time:.3f}s")
   ```

---

## 📚 相关资源

- [PyTorch AMP 官方文档](https://pytorch.org/docs/stable/amp.html)
- [PyTorch 分布式训练指南](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [学习率调度器详解](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

---

## 📝 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| 1.0.0 | 2024-11-18 | 初始文档，包含现有功能说明 |
| 1.0.1 | 2024-11-18 | 添加性能优化建议和故障排除指南 |

---

*本文档随项目更新而维护，如有问题或建议，请提交 Issue 或 Pull Request。*