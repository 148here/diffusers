# -*- coding: utf-8 -*-
"""
YZApatch - ControlNet训练自定义数据集模块
=========================================

提供实时sketch生成、复杂mask生成和边缘缓存功能。

主要组件：
- InpaintingSketchDataset: 自定义数据集类
- ComplexMaskGenerator: 复杂mask生成器
- EdgeCacheManager: 边缘缓存管理器
- create_huggingface_dataset: HuggingFace Dataset包装器

使用示例：
---------
from YZApatch import InpaintingSketchDataset, create_huggingface_dataset

# 创建数据集
dataset = InpaintingSketchDataset(
    image_dir="path/to/images",
    resolution=512,
    enable_edge_cache=True
)

# 转换为HuggingFace格式
hf_dataset = create_huggingface_dataset(dataset)
"""

__version__ = "1.0.0"
__author__ = "YZApatch"

# 导入主要组件
from .custom_dataset import InpaintingSketchDataset, MultiDatasetWrapper
from .dataset_wrapper import create_huggingface_dataset, create_lazy_huggingface_dataset
from .mask_generator import ComplexMaskGenerator, generate_complex_mask
from .edge_cache import EdgeCacheManager, get_edge_cache_manager

# 导入配置（可选，让用户能访问配置）
from . import config

# 定义公开接口
__all__ = [
    # 主要类
    'InpaintingSketchDataset',
    'MultiDatasetWrapper',
    'ComplexMaskGenerator',
    'EdgeCacheManager',
    
    # 便捷函数
    'create_huggingface_dataset',
    'create_lazy_huggingface_dataset',
    'generate_complex_mask',
    'get_edge_cache_manager',
    
    # 配置模块
    'config',
]


def get_version():
    """获取版本号"""
    return __version__


def print_config():
    """打印当前配置"""
    print("="*60)
    print("YZApatch Configuration")
    print("="*60)
    print(f"Version: {__version__}")
    print()
    print("Path Configuration:")
    print(f"  DexiNed Code Dir: {config.DEXINED_CODE_DIR}")
    print(f"  DexiNed Checkpoint: {config.DEXINED_CHECKPOINT}")
    print(f"  Train Data Dir: {config.TRAIN_DATA_DIR}")
    print(f"  Edge Cache Dir: {config.EDGE_CACHE_DIR}")
    print(f"  Sketch Util Dir: {config.SKETCH_UTIL_DIR}")
    print()
    print("DexiNed Parameters:")
    print(f"  Threshold: {config.DEXINED_THRESHOLD}")
    print(f"  Device: {config.DEXINED_DEVICE}")
    print()
    print("Sketch Parameters:")
    for key, value in config.SKETCH_PARAMS.items():
        print(f"  {key}: {value}")
    print()
    print("Mask Parameters:")
    for key, value in config.MASK_PARAMS.items():
        print(f"  {key}: {value}")
    print()
    print("Data Loading Parameters:")
    print(f"  Image Extensions: {config.IMAGE_EXTENSIONS}")
    print(f"  Recursive Scan: {config.RECURSIVE_SCAN}")
    print("="*60)


if __name__ == "__main__":
    # 当作为脚本运行时，显示配置
    print_config()
