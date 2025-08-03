"""
推理和预测模块
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json
import os
from tqdm import tqdm


class ViTLoRAPredictor:
    """ViT LoRA预测器"""
    
    def __init__(self, model_path, config_path=None, device=None):
        """
        Args:
            model_path (str): 模型权重路径
            config_path (str): 配置文件路径
            device (str): 设备
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.config_path = config_path
        
        # 加载配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        # 加载模型
        self.model = None
        self.class_names = None
        self.transform = None
        
    def load_model(self, model_class, transform=None):
        """加载模型"""
        # 加载检查点
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 创建模型
        self.model = model_class
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 设置变换
        self.transform = transform
        
        print(f"模型已加载: {self.model_path}")
        print(f"设备: {self.device}")
    
    def predict_single(self, image_path, return_probs=False):
        """预测单张图片"""
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model方法")
        
        # 加载图片
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # 预处理
        if self.transform:
            image = self.transform(image)
        
        # 添加batch维度
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(image)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        result = {
            'predicted_class': pred_class,
            'confidence': confidence
        }
        
        if self.class_names:
            result['class_name'] = self.class_names[pred_class]
        
        if return_probs:
            result['probabilities'] = probs.cpu().numpy()[0]
        
        return result
    
    def predict_batch(self, image_paths, batch_size=32, return_probs=False):
        """批量预测"""
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model方法")
        
        results = []
        
        # 分批处理
        for i in tqdm(range(0, len(image_paths), batch_size), desc="预测中"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            # 加载批次图片
            for path in batch_paths:
                if isinstance(path, str):
                    image = Image.open(path).convert('RGB')
                else:
                    image = path
                
                if self.transform:
                    image = self.transform(image)
                batch_images.append(image)
            
            # 转换为tensor
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                probs = F.softmax(logits, dim=1)
                pred_classes = torch.argmax(probs, dim=1)
                confidences = torch.max(probs, dim=1)[0]
            
            # 处理结果
            for j, path in enumerate(batch_paths):
                result = {
                    'image_path': path if isinstance(path, str) else 'tensor',
                    'predicted_class': pred_classes[j].item(),
                    'confidence': confidences[j].item()
                }
                
                if self.class_names:
                    result['class_name'] = self.class_names[pred_classes[j].item()]
                
                if return_probs:
                    result['probabilities'] = probs[j].cpu().numpy()
                
                results.append(result)
        
        return results
    
    def predict_folder(self, folder_path, output_path=None, 
                      batch_size=32, return_probs=False):
        """预测文件夹中的所有图片"""
        # 获取所有图片路径
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []
        
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(folder_path, filename))
        
        if not image_paths:
            print(f"在 {folder_path} 中未找到图片文件")
            return []
        
        print(f"找到 {len(image_paths)} 张图片")
        
        # 批量预测
        results = self.predict_batch(image_paths, batch_size, return_probs)
        
        # 保存结果
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {output_path}")
        
        return results
    
    def set_class_names(self, class_names):
        """设置类别名称"""
        self.class_names = class_names
    
    def get_feature_maps(self, image, layer_name=None):
        """获取特征图"""
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model方法")
        
        # 注册hook获取中间层输出
        features = {}
        
        def hook_fn(module, input, output):
            features[layer_name or 'features'] = output.detach()
        
        # 为指定层注册hook
        if hasattr(self.model, 'vit'):
            handle = self.model.vit.encoder.register_forward_hook(hook_fn)
        else:
            handle = self.model.register_forward_hook(hook_fn)
        
        # 前向传播
        with torch.no_grad():
            image = image.to(self.device)
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            _ = self.model(image)
        
        # 移除hook
        handle.remove()
        
        return features
    
    def explain_prediction(self, image, method='grad_cam'):
        """解释预测结果"""
        # 这里可以实现各种可解释性方法
        # 如Grad-CAM, LIME, SHAP等
        # 现在先返回简单的梯度信息
        
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model方法")
        
        image = image.to(self.device)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        image.requires_grad_()
        
        # 前向传播
        outputs = self.model(image)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # 获取预测类别
        pred_class = torch.argmax(logits, dim=1)
        
        # 反向传播获取梯度
        self.model.zero_grad()
        logits[0, pred_class].backward()
        
        # 获取输入梯度
        gradients = image.grad.data.abs()
        
        return {
            'gradients': gradients.cpu().numpy(),
            'predicted_class': pred_class.item(),
            'confidence': F.softmax(logits, dim=1)[0, pred_class].item()
        }


def load_checkpoint_for_inference(checkpoint_path):
    """加载检查点用于推理"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"检查点信息:")
    print(f"- Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"- 最佳验证准确率: {checkpoint.get('best_val_acc', 'Unknown')}")
    
    return checkpoint


def ensemble_predict(models, image, transforms=None):
    """集成预测"""
    predictions = []
    
    for i, model in enumerate(models):
        model.eval()
        
        # 使用对应的变换
        if transforms and i < len(transforms):
            processed_image = transforms[i](image)
        else:
            processed_image = image
        
        if len(processed_image.shape) == 3:
            processed_image = processed_image.unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(processed_image)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            probs = F.softmax(logits, dim=1)
            predictions.append(probs.cpu().numpy())
    
    # 平均预测概率
    avg_probs = np.mean(predictions, axis=0)
    pred_class = np.argmax(avg_probs)
    confidence = avg_probs[0, pred_class]
    
    return {
        'predicted_class': pred_class,
        'confidence': confidence,
        'probabilities': avg_probs[0],
        'individual_predictions': predictions
    }
