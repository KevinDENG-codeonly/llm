#!/usr/bin/env python3
"""
模型评估脚本
"""

import os
import sys
import argparse
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import create_data_loaders, get_val_transforms
from src.models import create_vit_lora_model
from src.inference import ViTLoRAPredictor
from src.training import calculate_metrics, plot_confusion_matrix


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ViT LoRA Model Evaluation')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型路径')
    parser.add_argument('--config_path', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='测试数据目录')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='评估结果保存目录')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备: auto, cpu, cuda')
    
    return parser.parse_args()


def load_model_and_config(model_path, config_path):
    """加载模型和配置"""
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 重建模型
    model = create_vit_lora_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        lora_r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        target_modules=config['lora']['target_modules']
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config


def evaluate_model(model, data_loader, device, class_names=None):
    """评估模型"""
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("开始评估...")
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算指标
    metrics = calculate_metrics(all_labels, all_preds, class_names)
    
    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs),
        'metrics': metrics
    }


def save_evaluation_results(results, output_dir, class_names=None):
    """保存评估结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存指标
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        # 转换numpy数组为列表以便JSON序列化
        metrics_json = {}
        for key, value in results['metrics'].items():
            if isinstance(value, np.ndarray):
                metrics_json[key] = value.tolist()
            else:
                metrics_json[key] = value
        json.dump(metrics_json, f, ensure_ascii=False, indent=2)
    
    # 保存预测结果
    predictions_path = os.path.join(output_dir, 'predictions.npz')
    np.savez(predictions_path,
             predictions=results['predictions'],
             labels=results['labels'],
             probabilities=results['probabilities'])
    
    # 生成分类报告
    report = classification_report(
        results['labels'], 
        results['predictions'],
        target_names=class_names,
        output_dict=True
    )
    
    report_path = os.path.join(output_dir, 'classification_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 打印分类报告
    print("\n=== 分类报告 ===")
    print(classification_report(
        results['labels'], 
        results['predictions'],
        target_names=class_names
    ))
    
    # 绘制混淆矩阵
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        results['labels'], 
        results['predictions'],
        class_names=class_names,
        save_path=cm_path
    )
    
    print(f"评估结果已保存到: {output_dir}")


def analyze_errors(results, class_names=None, top_k=5):
    """分析预测错误"""
    predictions = results['predictions']
    labels = results['labels']
    probabilities = results['probabilities']
    
    # 找到错误预测
    errors = predictions != labels
    error_indices = np.where(errors)[0]
    
    print(f"\n=== 错误分析 ===")
    print(f"总样本数: {len(labels)}")
    print(f"错误预测数: {len(error_indices)}")
    print(f"错误率: {len(error_indices) / len(labels) * 100:.2f}%")
    
    if len(error_indices) == 0:
        print("所有预测都是正确的！")
        return
    
    # 分析每个类别的错误
    if class_names:
        print("\n各类别错误统计:")
        for i, class_name in enumerate(class_names):
            class_indices = labels == i
            class_errors = errors[class_indices]
            if np.sum(class_indices) > 0:
                error_rate = np.sum(class_errors) / np.sum(class_indices)
                print(f"{class_name}: {error_rate * 100:.2f}% ({np.sum(class_errors)}/{np.sum(class_indices)})")
    
    # 分析置信度最低的错误
    error_probs = probabilities[error_indices]
    error_confidence = np.max(error_probs, axis=1)
    
    # 按置信度排序
    sorted_indices = np.argsort(error_confidence)[::-1]  # 降序
    
    print(f"\n置信度最高的 {min(top_k, len(sorted_indices))} 个错误预测:")
    for i in range(min(top_k, len(sorted_indices))):
        idx = error_indices[sorted_indices[i]]
        true_label = labels[idx]
        pred_label = predictions[idx]
        confidence = error_confidence[sorted_indices[i]]
        
        true_name = class_names[true_label] if class_names else str(true_label)
        pred_name = class_names[pred_label] if class_names else str(pred_label)
        
        print(f"样本 {idx}: 真实={true_name}, 预测={pred_name}, 置信度={confidence:.4f}")


def main():
    args = parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 加载模型和配置
    print("加载模型和配置...")
    model, config = load_model_and_config(args.model_path, args.config_path)
    
    # 加载类别名称
    class_names_path = os.path.join(os.path.dirname(args.config_path), 'class_names.json')
    class_names = None
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        print(f"类别名称: {class_names}")
    
    # 创建数据加载器
    print("创建数据加载器...")
    transform = get_val_transforms(config['data']['image_size'])
    
    # 这里简化处理，假设测试数据在指定目录下
    from src.data import ImageClassificationDataset
    from torch.utils.data import DataLoader
    
    test_dataset = ImageClassificationDataset(
        args.data_dir, 
        transform=transform, 
        split='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"测试样本数: {len(test_dataset)}")
    
    # 评估模型
    results = evaluate_model(model, test_loader, device, class_names)
    
    # 保存结果
    save_evaluation_results(results, args.output_dir, class_names)
    
    # 错误分析
    analyze_errors(results, class_names)
    
    print("\n评估完成！")


if __name__ == '__main__':
    main()
