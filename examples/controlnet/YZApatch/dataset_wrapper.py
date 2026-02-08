# -*- coding: utf-8 -*-
"""
HuggingFace Dataset包装器
=========================

将PyTorch Dataset包装为HuggingFace Dataset格式，
确保与原训练脚本的兼容性。
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional
from datasets import Dataset as HFDataset
from tqdm import tqdm


def _init_worker(gpu_id: int):
    """Worker 初始化：设置所有 worker 使用指定 GPU"""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def _load_sample(args):
    """Worker 加载单个样本：在 worker 内创建 dataset 并返回样本"""
    idx, config, base_seed = args
    import numpy as np
    from custom_dataset import create_dataset_from_config

    np.random.seed(base_seed + idx)
    dataset = create_dataset_from_config(config)
    return idx, dataset[idx]


def create_huggingface_dataset(
    custom_dataset,
    max_samples: Optional[int] = None,
    show_progress: bool = True
) -> HFDataset:
    """
    将自定义PyTorch Dataset转换为HuggingFace Dataset
    
    Args:
        custom_dataset: InpaintingSketchDataset或MultiDatasetWrapper实例
        max_samples: 最大样本数（None则使用全部）
        show_progress: 是否显示进度条
    
    Returns:
        hf_dataset: HuggingFace Dataset实例
    
    注意：
    - 这个函数会预加载所有数据到内存
    - 如果数据集很大，建议使用max_samples限制数量
    - 或者考虑使用HF的IterableDataset（流式加载）
    """
    # 检查是否是MultiDatasetWrapper
    if hasattr(custom_dataset, 'datasets'):
        # 多数据集模式
        print(f"[DatasetWrapper] Multi-dataset mode: {len(custom_dataset.datasets)} datasets")
        for i, (name, ds, w) in enumerate(zip(
            custom_dataset.dataset_names,
            custom_dataset.datasets,
            custom_dataset.weights
        )):
            print(f"  Dataset '{name}': {len(ds)} images, weight={w:.2%}")
    
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


def create_huggingface_dataset_parallel(
    custom_dataset,
    max_samples: Optional[int] = None,
    show_progress: bool = True,
    num_workers: int = 8,
    preload_gpu_id: int = 0,
    seed: Optional[int] = None,
) -> HFDataset:
    """
    使用多进程并行预加载，将自定义 PyTorch Dataset 转换为 HuggingFace Dataset。
    所有 worker 进程共享同一块 GPU（适合 72GB 单卡上 8 个 DexiNed 进程同时运行）。

    Args:
        custom_dataset: InpaintingSketchDataset 或 MultiDatasetWrapper 实例
        max_samples: 最大样本数（None 则使用全部）
        show_progress: 是否显示进度条
        num_workers: 并行 worker 数量（默认 8）
        preload_gpu_id: 预加载使用的 GPU ID（默认 0）
        seed: 随机种子，用于 MultiDatasetWrapper 采样可复现（None 则使用 42）

    Returns:
        hf_dataset: HuggingFace Dataset 实例
    """
    from custom_dataset import extract_dataset_config

    if hasattr(custom_dataset, 'datasets'):
        print(f"[DatasetWrapper] Multi-dataset mode (parallel): {len(custom_dataset.datasets)} datasets")
        for i, (name, ds, w) in enumerate(zip(
            custom_dataset.dataset_names,
            custom_dataset.datasets,
            custom_dataset.weights
        )):
            print(f"  Dataset '{name}': {len(ds)} images, weight={w:.2%}")
    else:
        print(f"[DatasetWrapper] Single-dataset mode (parallel)")

    num_samples = len(custom_dataset)
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)

    print(f"[DatasetWrapper] Parallel preloading {num_samples} samples with {num_workers} workers on GPU {preload_gpu_id}...")

    config = extract_dataset_config(custom_dataset)
    config["dexined_device"] = "cuda"
    base_seed = seed if seed is not None else 42

    results = [None] * num_samples
    failed_count = 0

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(preload_gpu_id,),
        mp_context=mp.get_context("spawn"),
    ) as pool:
        futures = {
            pool.submit(_load_sample, (i, config, base_seed)): i
            for i in range(num_samples)
        }
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=num_samples, desc="Loading samples (parallel)")

        for future in iterator:
            try:
                idx, sample = future.result()
                results[idx] = sample
            except Exception as e:
                orig_idx = futures[future]
                print(f"\n[WARNING] Failed to load sample {orig_idx}: {e}")
                failed_count += 1

    data_list = [r for r in results if r is not None]
    if len(data_list) == 0:
        raise ValueError("No valid samples loaded from custom dataset")

    if failed_count > 0:
        print(f"[DatasetWrapper] Skipped {failed_count} failed samples, loaded {len(data_list)} valid samples")
    else:
        print(f"[DatasetWrapper] Successfully loaded {len(data_list)} samples")

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
