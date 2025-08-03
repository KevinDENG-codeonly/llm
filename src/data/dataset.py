"""
数据集处理模块
处理图像数据加载、预处理和数据增强
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch


class ImageClassificationDataset(Dataset):
    """图像分类数据集类"""
    
    def __init__(self, data_dir, csv_file=None, transform=None, split='train'):
        """
        Args:
            data_dir (str): 数据目录路径
            csv_file (str, optional): CSV文件路径，包含图像路径和标签
            transform (callable, optional): 图像变换
            split (str): 数据集划分 ('train', 'val', 'test')
        """
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        if csv_file:
            self.data_df = pd.read_csv(csv_file)
            self.image_paths = self.data_df['image_path'].tolist()
            self.labels = self.data_df['label'].tolist()
        else:
            # 使用ImageFolder格式
            self.image_paths, self.labels = self._load_imagefolder_data()
        
        # 创建标签到索引的映射
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
    
    def _load_imagefolder_data(self):
        """从ImageFolder格式加载数据"""
        image_paths = []
        labels = []
        
        for class_name in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        img_path = os.path.join(class_dir, img_name)
                        image_paths.append(img_path)
                        labels.append(class_name)
        
        return image_paths, labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 转换标签为索引
        label_idx = self.label_to_idx[label]
        
        return {
            'image': image,
            'label': torch.tensor(label_idx, dtype=torch.long),
            'image_path': img_path
        }


def get_train_transforms(image_size=224, augment_level='medium'):
    """获取训练时的数据变换"""
    
    if augment_level == 'light':
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    elif augment_level == 'medium':
        transform = A.Compose([
            A.Resize(image_size + 32, image_size + 32),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    elif augment_level == 'heavy':
        transform = A.Compose([
            A.Resize(image_size + 64, image_size + 64),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=30, p=0.7),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.7),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    return transform


def get_val_transforms(image_size=224):
    """获取验证时的数据变换"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def create_data_loaders(train_dir, val_dir, test_dir=None, 
                       batch_size=32, num_workers=4, 
                       image_size=224, augment_level='medium'):
    """创建数据加载器"""
    
    # 定义变换
    train_transform = get_train_transforms(image_size, augment_level)
    val_transform = get_val_transforms(image_size)
    
    # 创建数据集
    train_dataset = ImageClassificationDataset(
        train_dir, transform=train_transform, split='train'
    )
    val_dataset = ImageClassificationDataset(
        val_dir, transform=val_transform, split='val'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    data_loaders = {
        'train': train_loader,
        'val': val_loader,
        'num_classes': train_dataset.num_classes,
        'class_names': list(train_dataset.label_to_idx.keys())
    }
    
    # 可选的测试集
    if test_dir and os.path.exists(test_dir):
        test_dataset = ImageClassificationDataset(
            test_dir, transform=val_transform, split='test'
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        data_loaders['test'] = test_loader
    
    return data_loaders


def mixup_data(x, y, alpha=1.0):
    """MixUp数据增强"""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample()
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix数据增强"""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample()
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # 调整lambda以匹配像素比例
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    """为CutMix生成随机边界框"""
    W = size[2]
    H = size[3]
    cut_rat = (1. - lam).sqrt()
    cut_w = (W * cut_rat).int()
    cut_h = (H * cut_rat).int()
    
    # 随机中心
    cx = torch.randint(W, (1,))
    cy = torch.randint(H, (1,))
    
    bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
    bby1 = torch.clamp(cy - cut_h // 2, 0, H)
    bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
    bby2 = torch.clamp(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2
