# ViT LoRA Training Project

## 项目概述
本项目实现了基于LoRA（Low-Rank Adaptation）技术对Vision Transformer（ViT）模型进行高效微调的完整工作流程。

## 项目特点
- 🚀 高效的LoRA微调，显著减少训练参数
- 🔧 模块化设计，易于扩展和定制
- 📊 完整的实验跟踪（MLflow, TensorBoard, W&B）
- 🎯 支持多种图像分类任务
- ⚡ GPU加速训练支持

## 项目结构
```
├── src/                    # 源代码
│   ├── data/              # 数据处理模块
│   ├── models/            # 模型定义和配置
│   ├── training/          # 训练相关代码
│   └── inference/         # 推理和评估代码
├── configs/               # 配置文件
├── scripts/               # 运行脚本
├── notebooks/             # Jupyter笔记本
├── data/                  # 数据目录
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后的数据
├── checkpoints/           # 模型检查点
└── logs/                 # 训练日志
```

## 快速开始

### 1. 环境设置
```bash
pip install -r requirements.txt
```

### 2. 数据准备
将您的图像数据放在 `data/raw/` 目录下，支持以下格式：
- ImageFolder格式（推荐）
- 自定义CSV格式

### 3. 配置训练参数
编辑 `configs/training_config.yaml` 和 `configs/model_config.yaml`

### 4. 开始训练
```bash
python scripts/train.py --config configs/training_config.yaml
```

### 5. 模型评估
```bash
python scripts/evaluate.py --model_path checkpoints/best_model.pth
```

## 技术栈
- **深度学习框架**: PyTorch
- **预训练模型**: Hugging Face Transformers, timm
- **微调技术**: PEFT (LoRA)
- **实验跟踪**: MLflow, TensorBoard, W&B
- **数据处理**: Albumentations, OpenCV

## 核心功能

### LoRA配置
- 支持自定义rank和alpha参数
- 可选择目标模块（attention, mlp等）
- 灵活的dropout配置

### 数据增强
- 现代化的图像增强策略
- 支持MixUp和CutMix
- 自适应数据增强

### 训练优化
- 混合精度训练
- 梯度检查点
- 学习率调度器
- 早停机制

## 贡献
欢迎提交Issue和Pull Request！

## 许可证
本项目基于MIT许可证开源。
