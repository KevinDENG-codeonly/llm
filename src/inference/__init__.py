"""
推理模块初始化文件
"""

from .predict import (
    ViTLoRAPredictor,
    load_checkpoint_for_inference,
    ensemble_predict
)

__all__ = [
    'ViTLoRAPredictor',
    'load_checkpoint_for_inference',
    'ensemble_predict'
]
