# ECG Digitization Project - Git Ignore Guide

## 📋 .gitignore 文件说明

本文档说明了项目 `.gitignore` 文件的使用方法和配置。

## 🎯 主要忽略内容

### 🔴 **必须忽略的文件类型**

#### **1. 模型文件 (太大)**
```
*.pth, *.pt, *.ckpt          # PyTorch模型检查点
*.h5, *.hdf5                 # HDF5模型文件
*.pkl, *.pickle              # Python序列化文件
```

#### **2. 训练输出**
```
outputs/                      # 训练输出目录
logs/                        # 训练日志
wandb/                       # Weights & Biases日志
tensorboard_logs/            # TensorBoard日志
*.log                        # 日志文件
```

#### **3. 数据文件 (太大)**
```
ecg_data/                    # ECG数据集
*.jpg, *.png, *.jpeg          # 图像文件
*.npy, *.npz                 # NumPy数组
*.csv, *.json                # 数据文件
```

#### **4. Python缓存**
```
__pycache__/                # Python字节码缓存
*.pyc                        # 编译的Python文件
```

### 🟡 **建议忽略的文件类型**

#### **1. IDE配置文件**
```
.vscode/                     # VS Code配置
.idea/                       # PyCharm配置
*.swp, *.swo                 # Vim临时文件
```

#### **2. 环境文件**
```
.env                         # 环境变量
venv/, env/                  # Python虚拟环境
```

#### **3. 系统文件**
```
.DS_Store                    # macOS系统文件
Thumbs.db                    # Windows缩略图
```

## 🟢 **不会忽略的文件**

### **必须保留的文件**
```
✅ README.md                   # 项目说明
✅ *.md                        # Markdown文档
✅ requirements.txt            # Python依赖
✅ setup.py                    # 安装脚本
✅ Makefile                    # 构建脚本
✅ .gitignore                  # Git忽略文件
```

### **代码文件**
```
✅ *.py                        # Python源代码
✅ *.yaml, *.yml               # 配置文件
```

## 🔧 使用指南

### 1. 检查被忽略的文件
```bash
# 查看哪些文件被Git忽略
git status --ignored

# 检查特定文件是否被忽略
git check-ignore path/to/file
```

### 2. 强制添加被忽略的文件
```bash
# 如果真的需要添加被忽略的文件
git add -f path/to/important_file.py
```

### 3. 忽略已跟踪的文件
```bash
# 从Git中删除文件但保留本地文件
git rm --cached path/to/file

# 忽略目录但不删除子目录内容
git rm -r --cached path/to/directory/
```

## 📁 项目特定配置

### **ECG项目特殊忽略**
```
# 消融研究结果
ablation_studies/results/
ablation_studies/plots/
ablation_studies/cache/

# 推理结果
predictions/
inference_results/
segmentation_masks/

# 临时实验文件
experiments/
temp/
cache/
```

## ⚠️ 注意事项

### **1. 重要文件保护**
以下文件永远不会被忽略：
- 所有Python源代码 (`.py`)
- 配置文件 (`.yaml`, `.yml`)
- 文档文件 (`.md`)
- 依赖文件 (`requirements.txt`)

### **2. 数据文件处理**
- **大型数据集**应该在 `.gitignore` 中
- **示例数据**可以保留在项目中
- 使用Git LFS处理必要的大文件

### **3. 模型文件**
- **训练检查点**应该在 `.gitignore` 中
- **预训练模型**应该在外部存储
- 使用模型仓库或云存储

## 🚀 最佳实践

### **1. 定期清理**
```bash
# 检查.gitignore是否按预期工作
git status

# 查看仓库大小
git count-objects -vH

# 清理不需要的大文件
git filter-branch --tree-filter 'rm -rf outputs/' --prune-empty HEAD
```

### **2. 团队协作**
```bash
# 团队成员应该使用相同的.gitignore
# 在项目根目录有统一的.gitignore文件
# 个人忽略规则使用 .git/info/exclude
```

### **3. 文档同步**
- 更新代码时同步更新`.gitignore`
- 在README中说明数据获取方式
- 提供示例数据下载脚本

## 📞 常见问题

### **Q: 为什么我的文件没有被忽略？**
A: 检查文件路径是否正确，确保`.gitignore`在项目根目录

### **Q: 如何忽略已提交的文件？**
A: 使用 `git rm --cached` 从Git中移除，然后添加到`.gitignore`

### **Q: 如何只忽略特定扩展名的文件？**
A: 使用模式如 `*.log` 或 `data/*.csv`

### **Q: 如何在忽略目录中保留特定文件？**
A: 使用 `!` 取消忽略，如：
```
outputs/
!outputs/README.md
```

## 🎉 总结

这个`.gitignore`配置确保了：
- ✅ **仓库大小合理** - 忽略大文件和缓存
- ✅ **团队协作友好** - 统一的忽略规则
- ✅ **隐私保护** - 忽略敏感的配置文件
- ✅ **版本控制清洁** - 只跟踪必要的源代码

**保持Git仓库干净和高效！**