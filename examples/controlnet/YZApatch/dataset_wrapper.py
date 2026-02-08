# -*- coding: utf-8 -*-
"""
HuggingFace Dataset包装器
=========================

将PyTorch Dataset包装为HuggingFace Dataset格式，
确保与原训练脚本的兼容性。
"""

from typing import Optional
from datasets import Dataset as HFDataset
from tqdm import tqdm


def create_huggingface_dataset(
    custom_dataset,
    max_samples: Optional[int] = None,
    show_progress: bool = True
) -> HFDataset:
    """
    将自定义PyTorch Dataset转换为HuggingFace Dataset
    
    Args:
        custom_dataset: InpaintingSketchDataset实例
        max_samples: 最大样本数（None则使用全部）
        show_progress: 是否显示进度条
    
    Returns:
        hf_dataset: HuggingFace Dataset实例
    
    注意：
    - 这个函数会预加载所有数据到内存
    - 如果数据集很大，建议使用max_samples限制数量
    - 或者考虑使用HF的IterableDataset（流式加载）
    """
    # 确定样本数量
    num_samples = len(custom_dataset)
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)
    
    print(f"[DatasetWrapper] Converting {num_samples} samples to HuggingFace Dataset format...")
    
    # 预加载数据
    data_list = []
    
    iterator = range(num_samples)
    if show_progress:
        iterator = tqdm(iterator, desc="Loading samples")
    
    for i in iterator:
        try:
            sample = custom_dataset[i]
            data_list.append(sample)
        except Exception as e:
            print(f"\n[WARNING] Failed to load sample {i}: {e}")
            print(f"[WARNING] Skipping sample {i}")
            continue
    
    if len(data_list) == 0:
        raise ValueError("No valid samples loaded from custom dataset")
    
    print(f"[DatasetWrapper] Successfully loaded {len(data_list)} samples")
    
    # 转换为HuggingFace Dataset
    hf_dataset = HFDataset.from_list(data_list)
    
    print(f"[DatasetWrapper] HuggingFace Dataset created with columns: {hf_dataset.column_names}")
    
    return hf_dataset


def create_lazy_huggingface_dataset(custom_dataset):
    """
    创建惰性加载的HuggingFace Dataset（使用生成器）
    
    这个方法不会预加载所有数据，而是在访问时才加载。
    适合大数据集，但与某些HF Dataset操作可能不兼容。
    
    Args:
        custom_dataset: InpaintingSketchDataset实例
    
    Returns:
        hf_dataset: HuggingFace Dataset实例
    """
    def gen():
        """生成器函数"""
        for i in range(len(custom_dataset)):
            try:
                yield custom_dataset[i]
            except Exception as e:
                print(f"[WARNING] Failed to load sample {i}: {e}")
                continue
    
    # 从生成器创建Dataset
    # 需要指定features以便HF知道数据结构
    from datasets import Features, Image as HFImage, Value
    
    features = Features({
        'image': HFImage(),
        'text': Value('string'),
        'conditioning_image': HFImage(),
        'mask': HFImage(),
    })
    
    hf_dataset = HFDataset.from_generator(gen, features=features)
    
    print(f"[DatasetWrapper] Created lazy HuggingFace Dataset with {len(custom_dataset)} samples")
    
    return hf_dataset


if __name__ == "__main__":
    # 测试代码
    import argparse
    import sys
    from pathlib import Path
    
    # 添加当前目录到path以便导入
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    from custom_dataset import InpaintingSketchDataset
    
    parser = argparse.ArgumentParser(description="Test dataset wrapper")
    parser.add_argument("--image_dir", type=str, required=True, help="Image directory")
    parser.add_argument("--max_samples", type=int, default=10, help="Max samples to load")
    parser.add_argument("--lazy", action="store_true", help="Use lazy loading")
    args = parser.parse_args()
    
    # 创建自定义数据集
    print("Creating custom dataset...")
    custom_dataset = InpaintingSketchDataset(
        image_dir=args.image_dir,
        resolution=512,
        enable_edge_cache=False,  # 测试时禁用缓存
    )
    
    print(f"Custom dataset size: {len(custom_dataset)}")
    
    # 转换为HuggingFace Dataset
    if args.lazy:
        print("\nUsing lazy loading...")
        hf_dataset = create_lazy_huggingface_dataset(custom_dataset)
    else:
        print("\nUsing eager loading...")
        hf_dataset = create_huggingface_dataset(
            custom_dataset,
            max_samples=args.max_samples,
            show_progress=True
        )
    
    # 测试访问
    print("\nTesting dataset access...")
    print(f"HuggingFace Dataset columns: {hf_dataset.column_names}")
    print(f"HuggingFace Dataset size: {len(hf_dataset)}")
    
    # 访问第一个样本
    print("\nAccessing first sample...")
    sample = hf_dataset[0]
    print(f"  image: {type(sample['image'])}, {sample['image'].size if hasattr(sample['image'], 'size') else 'N/A'}")
    print(f"  text: '{sample['text']}'")
    print(f"  conditioning_image: {type(sample['conditioning_image'])}, {sample['conditioning_image'].size if hasattr(sample['conditioning_image'], 'size') else 'N/A'}")
    print(f"  mask: {type(sample['mask'])}, {sample['mask'].size if hasattr(sample['mask'], 'size') else 'N/A'}")
    
    # 测试shuffle
    print("\nTesting shuffle...")
    shuffled = hf_dataset.shuffle(seed=42)
    print(f"Shuffled dataset size: {len(shuffled)}")
    
    # 测试select
    print("\nTesting select...")
    selected = hf_dataset.select(range(min(3, len(hf_dataset))))
    print(f"Selected dataset size: {len(selected)}")
    
    print("\nAll tests passed!")
