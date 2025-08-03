"""
训练器和训练工具
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from transformers import get_linear_schedule_with_warmup
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ViTLoRATrainer:
    """ViT LoRA训练器"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.0))
        
        # 训练状态
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # 混合精度
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _create_optimizer(self):
        """创建优化器"""
        optimizer_name = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-2)
        
        if optimizer_name.lower() == 'adamw':
            return AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler_name = self.config.get('scheduler', 'cosine')
        num_epochs = self.config.get('num_epochs', 10)
        
        if scheduler_name.lower() == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        elif scheduler_name.lower() == 'onecycle':
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.get('learning_rate', 1e-4),
                epochs=num_epochs,
                steps_per_epoch=len(self.train_loader)
            )
        elif scheduler_name.lower() == 'warmup':
            num_training_steps = len(self.train_loader) * num_epochs
            num_warmup_steps = int(0.1 * num_training_steps)
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            return None
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs
                    loss = self.criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                loss = self.criterion(logits, labels)
                
                loss.backward()
                self.optimizer.step()
            
            # 更新学习率（如果使用OneCycleLR）
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # 更新学习率（非OneCycleLR）
        if self.scheduler and not isinstance(self.scheduler, OneCycleLR):
            self.scheduler.step()
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        if hasattr(outputs, 'logits'):
                            logits = outputs.logits
                        else:
                            logits = outputs
                        loss = self.criterion(logits, labels)
                else:
                    outputs = self.model(images)
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs
                    loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = total_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        self.val_losses.append(epoch_loss)
        self.val_accs.append(epoch_acc)
        
        # 计算详细指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # 保存最佳模型
        if epoch_acc > self.best_val_acc:
            self.best_val_acc = epoch_acc
            self.save_checkpoint(epoch, is_best=True)
        
        return epoch_loss, epoch_acc, precision, recall, f1
    
    def train(self):
        """完整训练流程"""
        num_epochs = self.config.get('num_epochs', 10)
        
        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"设备: {self.device}")
        print(f"混合精度: {self.use_amp}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc, precision, recall, f1 = self.validate(epoch)
            
            # 打印结果
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
            print('-' * 60)
            
            # 记录到MLflow
            if self.config.get('use_mlflow', False):
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }, step=epoch)
        
        total_time = time.time() - start_time
        print(f'训练完成！总时间: {total_time:.2f}秒')
        print(f'最佳验证准确率: {self.best_val_acc:.2f}%')
        
        # 生成训练曲线
        self.plot_training_curves()
        
        return self.best_val_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint_dir = self.config.get('checkpoint_dir', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest.pth'))
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best.pth'))
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        save_dir = self.config.get('log_dir', './logs')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        
        # 记录到MLflow
        if self.config.get('use_mlflow', False):
            mlflow.log_artifact(os.path.join(save_dir, 'training_curves.png'))
        
        plt.show()


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def calculate_metrics(y_true, y_pred, class_names=None):
    """计算详细的评估指标"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    # 平均指标
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f1)
    
    metrics = {
        'accuracy': accuracy,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'per_class_support': support
    }
    
    # 打印详细报告
    print("\n=== 分类报告 ===")
    print(f"总体准确率: {accuracy:.4f}")
    print(f"平均精确率: {avg_precision:.4f}")
    print(f"平均召回率: {avg_recall:.4f}")
    print(f"平均F1分数: {avg_f1:.4f}")
    
    if class_names:
        print("\n各类别详细指标:")
        for i, name in enumerate(class_names):
            print(f"{name}: P={precision[i]:.4f}, R={recall[i]:.4f}, "
                  f"F1={f1[i]:.4f}, Support={support[i]}")
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names or range(len(cm)),
                yticklabels=class_names or range(len(cm)))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return cm
