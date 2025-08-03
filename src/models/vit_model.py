"""
ViT模型定义和LoRA配置
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig
from peft import LoraConfig, TaskType, get_peft_model
import timm


class ViTLoRAModel:
    """ViT LoRA模型包装类"""
    
    def __init__(self, model_name="google/vit-base-patch16-224", 
                 num_classes=10, lora_config=None):
        """
        Args:
            model_name (str): 预训练模型名称
            num_classes (int): 分类数量
            lora_config (dict): LoRA配置参数
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.lora_config = lora_config or self._get_default_lora_config()
        
    def _get_default_lora_config(self):
        """获取默认LoRA配置"""
        return {
            'task_type': TaskType.IMAGE_CLASSIFICATION,
            'r': 16,  # LoRA rank
            'lora_alpha': 32,  # LoRA scaling parameter
            'lora_dropout': 0.1,  # LoRA dropout
            'target_modules': ['query', 'key', 'value', 'dense'],  # 目标模块
            'inference_mode': False,
        }
    
    def create_model(self, use_timm=False):
        """创建ViT模型并应用LoRA"""
        if use_timm:
            return self._create_timm_model()
        else:
            return self._create_transformers_model()
    
    def _create_transformers_model(self):
        """使用Hugging Face transformers创建模型"""
        # 加载预训练模型
        model = ViTForImageClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True
        )
        
        # 应用LoRA
        lora_config = LoraConfig(**self.lora_config)
        model = get_peft_model(model, lora_config)
        
        return model
    
    def _create_timm_model(self):
        """使用timm创建模型"""
        # 加载预训练模型
        model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=self.num_classes
        )
        
        # timm模型需要手动应用LoRA到attention层
        self._apply_lora_to_timm_model(model)
        
        return model
    
    def _apply_lora_to_timm_model(self, model):
        """为timm模型手动应用LoRA"""
        # 这里需要更复杂的实现来处理timm模型
        # 暂时使用简化版本
        pass
    
    def get_target_modules_for_vit(self):
        """获取ViT模型的目标模块名称"""
        return [
            "query",
            "key", 
            "value",
            "dense"
        ]


def create_vit_lora_model(model_name="google/vit-base-patch16-224", 
                         num_classes=10, 
                         lora_r=16,
                         lora_alpha=32,
                         lora_dropout=0.1,
                         target_modules=None):
    """便捷函数：创建ViT LoRA模型"""
    
    if target_modules is None:
        target_modules = ["query", "key", "value", "dense"]
    
    # 加载预训练模型（使用镜像端点）
    import os
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        cache_dir=os.environ.get('HUGGINGFACE_HUB_CACHE', '/root/.cache/huggingface')
    )
    
    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.IMAGE_CLASSIFICATION,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        inference_mode=False,
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    return model


def print_trainable_parameters(model):
    """打印可训练参数数量"""
    trainable_params = 0
    all_param = 0
    
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"可训练参数: {trainable_params:,} || "
          f"总参数: {all_param:,} || "
          f"可训练参数比例: {100 * trainable_params / all_param:.2f}%")
    
    return trainable_params, all_param


class ViTFeatureExtractor:
    """ViT特征提取器"""
    
    def __init__(self, model, layer_name='last_hidden_state'):
        self.model = model
        self.layer_name = layer_name
        self.features = {}
        
        # 注册hook
        self._register_hooks()
    
    def _register_hooks(self):
        """注册特征提取hooks"""
        def hook_fn(module, input, output):
            self.features[self.layer_name] = output.detach()
        
        # 为特定层注册hook
        if hasattr(self.model, 'vit'):
            self.model.vit.encoder.register_forward_hook(hook_fn)
        elif hasattr(self.model, 'base_model'):
            self.model.base_model.encoder.register_forward_hook(hook_fn)
    
    def extract_features(self, x):
        """提取特征"""
        self.features.clear()
        with torch.no_grad():
            _ = self.model(x)
        return self.features.get(self.layer_name, None)


def freeze_backbone(model, freeze=True):
    """冻结/解冻主干网络"""
    if hasattr(model, 'vit'):
        backbone = model.vit
    elif hasattr(model, 'base_model'):
        backbone = model.base_model
    else:
        backbone = model
    
    for param in backbone.parameters():
        param.requires_grad = not freeze
    
    # 只训练分类头和LoRA参数
    if hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    print(f"主干网络已{'冻结' if freeze else '解冻'}")


def get_model_size(model):
    """获取模型大小"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f'模型大小: {size_all_mb:.2f} MB')
    
    return size_all_mb
