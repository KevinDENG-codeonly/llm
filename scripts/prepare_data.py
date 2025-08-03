#!/usr/bin/env python3
"""
数据准备脚本
"""

import os
import argparse
import shutil
from pathlib import Path
import random
from PIL import Image
import json


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='数据准备工具')
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='原始数据目录路径')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='处理后数据保存目录')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='测试集比例')
    parser.add_argument('--image_size', type=int, default=224,
                       help='图像尺寸')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--min_images', type=int, default=10,
                       help='每个类别最少图像数量')
    
    return parser.parse_args()


def validate_ratios(train_ratio, val_ratio, test_ratio):
    """验证比例参数"""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"比例总和必须为1.0，当前为: {total}")


def get_image_files(directory):
    """获取目录下的所有图像文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for file_path in Path(directory).rglob('*'):
        if file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    return image_files


def organize_by_class(input_dir):
    """按类别组织图像文件"""
    classes = {}
    
    for class_dir in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_dir)
        if os.path.isdir(class_path):
            image_files = get_image_files(class_path)
            if image_files:
                classes[class_dir] = image_files
    
    return classes


def split_data(image_files, train_ratio, val_ratio, test_ratio, seed):
    """分割数据集"""
    random.seed(seed)
    random.shuffle(image_files)
    
    total = len(image_files)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]
    
    return train_files, val_files, test_files


def copy_files(files, dest_dir, class_name):
    """复制文件到目标目录"""
    class_dest_dir = os.path.join(dest_dir, class_name)
    os.makedirs(class_dest_dir, exist_ok=True)
    
    copied_count = 0
    failed_count = 0
    
    for file_path in files:
        try:
            dest_path = os.path.join(class_dest_dir, file_path.name)
            
            # 如果目标文件已存在，添加序号
            counter = 1
            original_dest_path = dest_path
            while os.path.exists(dest_path):
                name_parts = original_dest_path.rsplit('.', 1)
                if len(name_parts) == 2:
                    dest_path = f"{name_parts[0]}_{counter}.{name_parts[1]}"
                else:
                    dest_path = f"{original_dest_path}_{counter}"
                counter += 1
            
            shutil.copy2(file_path, dest_path)
            copied_count += 1
        
        except Exception as e:
            print(f"复制文件失败 {file_path}: {e}")
            failed_count += 1
    
    return copied_count, failed_count


def validate_images(files, image_size=None):
    """验证图像文件"""
    valid_files = []
    invalid_files = []
    
    for file_path in files:
        try:
            with Image.open(file_path) as img:
                # 检查图像是否可以正常打开
                img.verify()
                
                # 重新打开图像进行进一步检查
                with Image.open(file_path) as img:
                    # 检查图像尺寸
                    if image_size and (img.width < image_size or img.height < image_size):
                        print(f"图像尺寸过小 {file_path}: {img.size}")
                        invalid_files.append(file_path)
                    else:
                        valid_files.append(file_path)
        
        except Exception as e:
            print(f"无效图像文件 {file_path}: {e}")
            invalid_files.append(file_path)
    
    return valid_files, invalid_files


def create_dataset_info(output_dir, classes_info):
    """创建数据集信息文件"""
    info = {
        'num_classes': len(classes_info),
        'classes': {}
    }
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for class_name, counts in classes_info.items():
        info['classes'][class_name] = {
            'train': counts['train'],
            'val': counts['val'],
            'test': counts['test'],
            'total': counts['train'] + counts['val'] + counts['test']
        }
        total_train += counts['train']
        total_val += counts['val']
        total_test += counts['test']
    
    info['total'] = {
        'train': total_train,
        'val': total_val,
        'test': total_test,
        'all': total_train + total_val + total_test
    }
    
    # 保存到JSON文件
    info_path = os.path.join(output_dir, 'dataset_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    
    return info


def main():
    args = parse_args()
    
    # 验证参数
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"数据分割比例 - 训练: {args.train_ratio}, 验证: {args.val_ratio}, 测试: {args.test_ratio}")
    
    # 创建输出目录
    train_dir = os.path.join(args.output_dir, 'train')
    val_dir = os.path.join(args.output_dir, 'val')
    test_dir = os.path.join(args.output_dir, 'test')
    
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 按类别组织文件
    print("扫描输入目录...")
    classes = organize_by_class(args.input_dir)
    
    if not classes:
        print("未找到任何类别目录或图像文件")
        return
    
    print(f"找到 {len(classes)} 个类别:")
    for class_name, files in classes.items():
        print(f"  {class_name}: {len(files)} 个文件")
    
    # 处理每个类别
    classes_info = {}
    
    for class_name, image_files in classes.items():
        print(f"\n处理类别: {class_name}")
        
        # 验证图像文件
        valid_files, invalid_files = validate_images(image_files, args.image_size)
        
        if len(invalid_files) > 0:
            print(f"  跳过 {len(invalid_files)} 个无效文件")
        
        if len(valid_files) < args.min_images:
            print(f"  警告: 类别 {class_name} 只有 {len(valid_files)} 个有效图像，少于最小要求 {args.min_images}")
            continue
        
        # 分割数据
        train_files, val_files, test_files = split_data(
            valid_files, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )
        
        print(f"  分割结果: 训练={len(train_files)}, 验证={len(val_files)}, 测试={len(test_files)}")
        
        # 复制文件
        train_copied, train_failed = copy_files(train_files, train_dir, class_name)
        val_copied, val_failed = copy_files(val_files, val_dir, class_name)
        test_copied, test_failed = copy_files(test_files, test_dir, class_name)
        
        # 记录统计信息
        classes_info[class_name] = {
            'train': train_copied,
            'val': val_copied,
            'test': test_copied,
            'failed': train_failed + val_failed + test_failed
        }
        
        if train_failed + val_failed + test_failed > 0:
            print(f"  复制失败: {train_failed + val_failed + test_failed} 个文件")
    
    # 创建数据集信息文件
    dataset_info = create_dataset_info(args.output_dir, classes_info)
    
    # 打印最终统计
    print("\n" + "="*50)
    print("数据集创建完成！")
    print(f"总类别数: {dataset_info['num_classes']}")
    print(f"训练集: {dataset_info['total']['train']} 个样本")
    print(f"验证集: {dataset_info['total']['val']} 个样本")
    print(f"测试集: {dataset_info['total']['test']} 个样本")
    print(f"总计: {dataset_info['total']['all']} 个样本")
    print(f"数据集信息已保存到: {os.path.join(args.output_dir, 'dataset_info.json')}")


if __name__ == '__main__':
    main()
