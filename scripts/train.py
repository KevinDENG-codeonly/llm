#!/usr/bin/env python3
"""
ViT LoRA训练脚本
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
import mlflow
import mlflow.pytorch
import torch

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import create_data_loaders
from src.models import create_vit_lora_model, print_trainable_parameters, get_lora_config
from src.training import ViTLoRATrainer, set_seed, get_device


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ViT LoRA Training')
    
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='训练配置文件路径')
    parser.add_argument('--model_config', type=str, default='configs/model_config.yaml',
                       help='模型配置文件路径')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='数据目录路径，覆盖配置文件中的设置')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='输出目录路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--gpu', type=int, default=None,
                       help='指定使用的GPU编号')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式，使用小数据集')
    
    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_mlflow(config):
    """设置MLflow"""
    if config.get('experiment', {}).get('use_mlflow', False):
        experiment_name = config['experiment']['experiment_name']
        mlflow.set_experiment(experiment_name)
        
        # 开始新的运行
        run_name = config['experiment'].get('run_name')
        mlflow.start_run(run_name=run_name)
        
        # 记录配置参数
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    mlflow.log_param(f"{key}.{sub_key}", sub_value)
            else:
                mlflow.log_param(key, value)


def create_output_dirs(output_dir):
    """创建输出目录"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_dir, 'checkpoints')).mkdir(exist_ok=True)
    Path(os.path.join(output_dir, 'logs')).mkdir(exist_ok=True)
    return output_dir


def validate_data_paths(config):
    """验证数据路径"""
    data_config = config['data']
    
    required_paths = ['train_dir', 'val_dir']
    for path_key in required_paths:
        path = data_config.get(path_key)
        if not path or not os.path.exists(path):
            raise ValueError(f"数据路径不存在: {path}")
    
    print(f"训练数据: {data_config['train_dir']}")
    print(f"验证数据: {data_config['val_dir']}")
    
    if 'test_dir' in data_config and data_config['test_dir']:
        if os.path.exists(data_config['test_dir']):
            print(f"测试数据: {data_config['test_dir']}")
        else:
            print("警告: 测试数据路径不存在，将跳过测试")


def main():
    args = parse_args()
    
    # 加载配置
    print("加载配置文件...")
    config = load_config(args.config)
    model_config = load_config(args.model_config)
    
    # 设置输出目录
    output_dir = create_output_dirs(args.output_dir)
    config['save']['checkpoint_dir'] = os.path.join(output_dir, 'checkpoints')
    config['save']['log_dir'] = os.path.join(output_dir, 'logs')
    
    # 覆盖数据目录设置
    if args.data_dir:
        config['data']['train_dir'] = os.path.join(args.data_dir, 'train')
        config['data']['val_dir'] = os.path.join(args.data_dir, 'val')
        config['data']['test_dir'] = os.path.join(args.data_dir, 'test')
    
    # 验证数据路径
    validate_data_paths(config)
    
    # 设置随机种子
    seed = config['environment']['seed']
    set_seed(seed)
    print(f"随机种子: {seed}")
    
    # 设置设备
    device = get_device()
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    
    # 设置MLflow
    setup_mlflow(config)
    
    try:
        # 创建数据加载器
        print("创建数据加载器...")
        data_loaders = create_data_loaders(
            train_dir=config['data']['train_dir'],
            val_dir=config['data']['val_dir'],
            test_dir=config['data'].get('test_dir'),
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            image_size=config['data']['image_size'],
            augment_level=config['data']['augment_level']
        )
        
        print(f"数据集信息:")
        print(f"- 类别数量: {data_loaders['num_classes']}")
        print(f"- 类别名称: {data_loaders['class_names']}")
        print(f"- 训练批次数: {len(data_loaders['train'])}")
        print(f"- 验证批次数: {len(data_loaders['val'])}")
        
        # 调试模式：使用小数据集
        if args.debug:
            print("调试模式：使用小数据集")
            config['training']['num_epochs'] = 2
            config['data']['batch_size'] = 8
        
        # 创建模型
        print("创建模型...")
        model = create_vit_lora_model(
            model_name=config['model']['name'],
            num_classes=data_loaders['num_classes'],
            lora_r=config['lora']['r'],
            lora_alpha=config['lora']['alpha'],
            lora_dropout=config['lora']['dropout'],
            target_modules=config['lora']['target_modules']
        )
        
        # 打印模型信息
        print_trainable_parameters(model)
        
        # 恢复训练
        start_epoch = 0
        if args.resume and os.path.exists(args.resume):
            print(f"从检查点恢复训练: {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"从第 {start_epoch} 个epoch开始继续训练")
        
        # 创建训练器
        print("创建训练器...")
        trainer = ViTLoRATrainer(
            model=model,
            train_loader=data_loaders['train'],
            val_loader=data_loaders['val'],
            config=config
        )
        
        # 保存配置文件
        config_save_path = os.path.join(output_dir, 'config.yaml')
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        model_config_save_path = os.path.join(output_dir, 'model_config.yaml')
        with open(model_config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(model_config, f, default_flow_style=False, allow_unicode=True)
        
        # 保存类别名称
        class_names_path = os.path.join(output_dir, 'class_names.json')
        with open(class_names_path, 'w', encoding='utf-8') as f:
            json.dump(data_loaders['class_names'], f, ensure_ascii=False, indent=2)
        
        print("开始训练...")
        print("=" * 80)
        
        # 开始训练
        best_acc = trainer.train()
        
        print("=" * 80)
        print(f"训练完成！最佳验证准确率: {best_acc:.2f}%")
        
        # 记录最终结果到MLflow
        if config.get('experiment', {}).get('use_mlflow', False):
            mlflow.log_metric('best_val_accuracy', best_acc)
            mlflow.log_artifact(config_save_path)
            mlflow.log_artifact(model_config_save_path)
            mlflow.log_artifact(class_names_path)
    
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        raise e
    
    finally:
        # 结束MLflow运行
        if config.get('experiment', {}).get('use_mlflow', False):
            mlflow.end_run()


if __name__ == '__main__':
    main()
