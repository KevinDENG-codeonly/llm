"""
LoRA配置定义
"""

from peft import LoraConfig, TaskType


class ViTLoRAConfig:
    """ViT LoRA配置类"""
    
    @staticmethod
    def get_base_config():
        """基础LoRA配置"""
        return LoraConfig(
            task_type=TaskType.IMAGE_CLASSIFICATION,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"],
            inference_mode=False,
        )
    
    @staticmethod
    def get_medium_config():
        """中等LoRA配置"""
        return LoraConfig(
            task_type=TaskType.IMAGE_CLASSIFICATION,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value", "dense"],
            inference_mode=False,
        )
    
    @staticmethod
    def get_large_config():
        """大型LoRA配置"""
        return LoraConfig(
            task_type=TaskType.IMAGE_CLASSIFICATION,
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["query", "key", "value", "dense", "intermediate.dense", "output.dense"],
            inference_mode=False,
        )
    
    @staticmethod
    def get_custom_config(r=16, alpha=32, dropout=0.1, target_modules=None):
        """自定义LoRA配置"""
        if target_modules is None:
            target_modules = ["query", "key", "value", "dense"]
        
        return LoraConfig(
            task_type=TaskType.IMAGE_CLASSIFICATION,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            inference_mode=False,
        )


# 预定义配置
LORA_CONFIGS = {
    'base': ViTLoRAConfig.get_base_config,
    'medium': ViTLoRAConfig.get_medium_config,
    'large': ViTLoRAConfig.get_large_config,
}


def get_lora_config(config_name='medium', **kwargs):
    """获取LoRA配置"""
    if config_name in LORA_CONFIGS:
        return LORA_CONFIGS[config_name]()
    else:
        return ViTLoRAConfig.get_custom_config(**kwargs)
