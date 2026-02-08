# -*- coding: utf-8 -*-
"""
自定义Inpainting Sketch数据集
==============================

整合所有功能的核心Dataset类：
- 递归扫描图片文件
- 实时sketch生成
- 复杂mask生成
- 边缘缓存
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional, List
from PIL import Image
import torch
from torch.utils.data import Dataset

# 导入YZApatch模块
try:
    from .config import (
        SKETCH_UTIL_DIR, DEXINED_CODE_DIR, DEXINED_CHECKPOINT,
        DEXINED_THRESHOLD, DEXINED_DEVICE, EDGE_CACHE_DIR,
        SKETCH_PARAMS, MASK_PARAMS, IMAGE_EXTENSIONS, RECURSIVE_SCAN,
        EDGE_CACHE_VERSION
    )
    from .mask_generator import ComplexMaskGenerator
    from .edge_cache import get_edge_cache_manager
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from config import (
        SKETCH_UTIL_DIR, DEXINED_CODE_DIR, DEXINED_CHECKPOINT,
        DEXINED_THRESHOLD, DEXINED_DEVICE, EDGE_CACHE_DIR,
        SKETCH_PARAMS, MASK_PARAMS, IMAGE_EXTENSIONS, RECURSIVE_SCAN,
        EDGE_CACHE_VERSION
    )
    from mask_generator import ComplexMaskGenerator
    from edge_cache import get_edge_cache_manager

# 导入sketch_util
sys.path.insert(0, SKETCH_UTIL_DIR)
try:
    from dataset.sketch_util import make_sketch_from_image_or_edge, extract_edge
except ImportError as e:
    print(f"[WARNING] Failed to import sketch_util: {e}")
    print(f"[WARNING] Please check SKETCH_UTIL_DIR in config.py: {SKETCH_UTIL_DIR}")
    raise


class InpaintingSketchDataset(Dataset):
    """
    自定义Inpainting Sketch数据集
    
    功能：
    - 从图片目录加载原始图像
    - 实时生成sketch（使用DexiNed边缘提取 + 形变）
    - 生成基于边缘密度的复杂mask
    - 支持边缘缓存加速训练
    - 返回空字符串caption
    
    数据流程：
    1. 加载原图 -> resize到指定分辨率
    2. 提取/缓存边缘图（DexiNed）
    3. 生成sketch（参数化形变）
    4. 生成复杂mask（基于边缘密度）
    5. 返回PIL Image格式数据
    """
    
    def __init__(
        self,
        image_dir: str,
        resolution: int = 512,
        enable_edge_cache: bool = True,
        dexined_checkpoint: Optional[str] = None,
        dexined_threshold: Optional[int] = None,
        dexined_device: Optional[str] = None,
        sketch_params: Optional[dict] = None,
        mask_params: Optional[dict] = None,
        recursive_scan: Optional[bool] = None,
        debug_edge: bool = False,
        debug_edge_output_dir: Optional[str] = None,
    ):
        """
        初始化数据集
        
        Args:
            image_dir: 图片目录路径
            resolution: 图像分辨率（默认512）
            enable_edge_cache: 是否启用边缘缓存
            dexined_checkpoint: DexiNed checkpoint路径（None则使用config中的默认值）
            dexined_threshold: DexiNed阈值（None则使用config中的默认值）
            dexined_device: DexiNed推理设备（None则使用config中的默认值）
            sketch_params: sketch生成参数（None则使用config中的默认值）
            mask_params: mask生成参数（None则使用config中的默认值）
            recursive_scan: 是否递归扫描子目录（None则使用config中的默认值）
            debug_edge: 是否保存edge图到输出目录（用于排查edge/mask问题）
            debug_edge_output_dir: edge debug输出目录（debug_edge=True时有效）
        """
        self.image_dir = Path(image_dir)
        self.resolution = resolution
        self.enable_edge_cache = enable_edge_cache
        
        # 使用配置或参数
        self.dexined_checkpoint = dexined_checkpoint or DEXINED_CHECKPOINT
        self.dexined_threshold = dexined_threshold if dexined_threshold is not None else DEXINED_THRESHOLD
        self.dexined_device = dexined_device or DEXINED_DEVICE
        self.sketch_params = sketch_params or SKETCH_PARAMS
        self.mask_params = mask_params or MASK_PARAMS
        self.recursive_scan = recursive_scan if recursive_scan is not None else RECURSIVE_SCAN
        self.debug_edge = debug_edge
        self.debug_edge_output_dir = Path(debug_edge_output_dir) if debug_edge_output_dir else None
        
        # 验证目录存在
        if not self.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {self.image_dir}")
        
        # 扫描图片文件
        self.image_paths = self._scan_images()
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        
        print(f"[InpaintingSketchDataset] Found {len(self.image_paths)} images in {self.image_dir}")
        
        # 初始化边缘缓存管理器
        if self.enable_edge_cache:
            dexined_params = {
                "threshold": self.dexined_threshold,
                "version": EDGE_CACHE_VERSION,
            }
            self.edge_cache_manager = get_edge_cache_manager(EDGE_CACHE_DIR, dexined_params)
            print(f"[InpaintingSketchDataset] Edge cache enabled: {EDGE_CACHE_DIR}")
        else:
            self.edge_cache_manager = None
            print(f"[InpaintingSketchDataset] Edge cache disabled (will extract edges in real-time)")
        
        # 初始化mask生成器（debug_edge时传入输出目录以输出详细统计）
        mp = dict(self.mask_params)
        if self.debug_edge and self.debug_edge_output_dir:
            mp["_debug_output_dir"] = str(self.debug_edge_output_dir)
        self.mask_generator = ComplexMaskGenerator(mp)
        
        print(f"[InpaintingSketchDataset] Initialized with resolution={resolution}")
        print(f"[InpaintingSketchDataset] DexiNed checkpoint: {self.dexined_checkpoint}")
        print(f"[InpaintingSketchDataset] Sketch params: {self.sketch_params}")
        if self.debug_edge and self.debug_edge_output_dir:
            print(f"[InpaintingSketchDataset] Debug mode: edge + mask stats -> {self.debug_edge_output_dir}")
    
    def _scan_images(self) -> List[Path]:
        """
        扫描图片文件
        
        Returns:
            image_paths: 图片路径列表
        """
        image_paths = []
        
        if self.recursive_scan:
            # 递归扫描所有子目录
            for ext in IMAGE_EXTENSIONS:
                image_paths.extend(self.image_dir.rglob(f"*{ext}"))
        else:
            # 只扫描当前目录
            for ext in IMAGE_EXTENSIONS:
                image_paths.extend(self.image_dir.glob(f"*{ext}"))
        
        # 排序保证顺序一致
        image_paths = sorted(image_paths)
        
        return image_paths
    
    def _extract_edge_with_cache(self, image_path: Path, image_np: np.ndarray) -> np.ndarray:
        """
        提取边缘图（使用缓存）
        
        Args:
            image_path: 图片路径
            image_np: 图片numpy数组 [H,W,3] RGB
        
        Returns:
            edge_image: 边缘图 [H,W,3] RGB，白底黑线
        """
        def extract_fn(img_np):
            """边缘提取函数（供缓存管理器调用）"""
            return extract_edge(
                image=img_np,
                method="dexined",
                dexined_checkpoint=self.dexined_checkpoint,
                dexined_threshold=self.dexined_threshold,
                dexined_device=self.dexined_device
            )
        
        # 如果启用缓存，使用缓存管理器
        if self.enable_edge_cache and self.edge_cache_manager is not None:
            edge_image = self.edge_cache_manager.get_or_compute_edge(
                image_path=str(image_path),
                image_np=image_np,
                enable_cache=True,
                extract_fn=extract_fn
            )
        else:
            # 不使用缓存，直接提取
            edge_image = extract_fn(image_np)
        
        return edge_image
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> dict:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            sample: 数据样本字典，包含：
                - 'image': PIL.Image，原图
                - 'text': str，caption（空字符串）
                - 'conditioning_image': PIL.Image，sketch条件图
                - 'mask': PIL.Image，复杂mask（可选）
        """
        # 1. 加载原始图像
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"[ERROR] Failed to load image {image_path}: {e}")
            # 返回第一张图片作为fallback
            image = Image.open(self.image_paths[0]).convert('RGB')
        
        # 2. Resize到目标分辨率
        if image.size != (self.resolution, self.resolution):
            image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
        
        # 3. 转为numpy用于处理
        image_np = np.array(image)  # [H,W,3], RGB, uint8
        
        # 4. 提取边缘图（使用缓存）
        edge_image = self._extract_edge_with_cache(image_path, image_np)
        
        # 4.1 debug: 保存edge图供排查（判断是edge问题还是后续mask计算问题）
        if self.debug_edge and self.debug_edge_output_dir:
            out_dir = self.debug_edge_output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"edge_debug_{idx}_{image_path.stem}.png"
            Image.fromarray(edge_image).save(out_dir / fname)
        
        # 5. 生成sketch（使用随机种子确保每次不同）
        # 使用idx和当前epoch作为seed的一部分，确保不同epoch产生不同结果
        seed = np.random.randint(0, 2**31 - 1)
        
        sketch_np = make_sketch_from_image_or_edge(
            input_image=edge_image,
            seed=seed,
            is_edge=True,  # 输入已是边缘图
            enable_edge_extraction=False,  # 跳过边缘提取
            # Sketch参数化形变
            sigma_mean=self.sketch_params.get("sigma_mean", 13.0),
            sigma_std=self.sketch_params.get("sigma_std", 2.6),
            spatial_smooth_sigma=self.sketch_params.get("spatial_smooth_sigma", 2.0),
            cp_sigma_mean=self.sketch_params.get("cp_sigma_mean", 2.1),
            cp_sigma_std=self.sketch_params.get("cp_sigma_std", 0.4),
            cp_spatial_smooth=self.sketch_params.get("cp_spatial_smooth", 1.5),
        )
        
        # 6. 生成复杂mask（基于边缘密度）
        mask_seed = np.random.randint(0, 2**31 - 1)
        mask_np = self.mask_generator.generate(edge_image, seed=mask_seed)
        
        # 7. 转为PIL Image
        sketch_pil = Image.fromarray(sketch_np)
        mask_pil = Image.fromarray(mask_np)
        
        # 8. 返回数据样本
        sample = {
            'image': image,                    # 原图
            'text': '',                        # 空字符串caption
            'conditioning_image': sketch_pil,  # sketch条件图
            'mask': mask_pil                   # mask（可选）
        }
        
        return sample
    
    def get_cache_stats(self) -> dict:
        """
        获取缓存统计信息
        
        Returns:
            stats: 缓存统计字典
        """
        if self.enable_edge_cache and self.edge_cache_manager is not None:
            return self.edge_cache_manager.get_cache_stats()
        else:
            return {
                "cache_enabled": False,
                "message": "Edge cache is disabled"
            }


class MultiDatasetWrapper(Dataset):
    """
    多数据集包装器
    
    功能：
    - 加载多个数据集（每个数据集有独立配置）
    - 按权重动态采样（per-batch随机选择）
    - 统一返回格式
    
    使用场景：
    - 混合多个数据源训练
    - 控制各数据源的采样比例
    - 支持不同的目录结构（单层/递归）
    """
    
    def __init__(
        self,
        datasets_config: List[dict],
        resolution: int = 512,
        enable_edge_cache: bool = True,
        dexined_checkpoint: Optional[str] = None,
        dexined_threshold: Optional[int] = None,
        dexined_device: Optional[str] = None,
        sketch_params: Optional[dict] = None,
        mask_params: Optional[dict] = None,
        debug_edge: bool = False,
        debug_edge_output_dir: Optional[str] = None,
    ):
        """
        初始化多数据集包装器
        
        Args:
            datasets_config: 数据集配置列表，每个配置包含：
                - name: 数据集名称
                - path: 数据集路径
                - weight: 采样权重（默认1.0）
                - recursive_scan: 是否递归扫描（默认True）
            resolution: 图像分辨率
            enable_edge_cache: 是否启用边缘缓存
            debug_edge: 是否保存edge图到输出目录
            debug_edge_output_dir: edge debug输出目录
            其他参数同InpaintingSketchDataset
        """
        if not datasets_config or len(datasets_config) == 0:
            raise ValueError("datasets_config must contain at least one dataset")
        
        self.datasets = []
        self.weights = []
        self.dataset_names = []
        
        print(f"[MultiDatasetWrapper] Initializing {len(datasets_config)} datasets...")
        
        # 加载所有数据集
        for i, config in enumerate(datasets_config):
            name = config.get("name", f"dataset_{i}")
            path = config.get("path")
            weight = config.get("weight", 1.0)
            recursive_scan = config.get("recursive_scan", True)
            
            if not path:
                raise ValueError(f"Dataset '{name}' missing required 'path' field")
            
            print(f"\n[MultiDatasetWrapper] Loading dataset '{name}':")
            print(f"  Path: {path}")
            print(f"  Weight: {weight}")
            print(f"  Recursive scan: {recursive_scan}")
            
            # 创建单个数据集
            dataset = InpaintingSketchDataset(
                image_dir=path,
                resolution=resolution,
                enable_edge_cache=enable_edge_cache,
                dexined_checkpoint=dexined_checkpoint,
                dexined_threshold=dexined_threshold,
                dexined_device=dexined_device,
                sketch_params=sketch_params,
                mask_params=mask_params,
                recursive_scan=recursive_scan,
                debug_edge=debug_edge,
                debug_edge_output_dir=debug_edge_output_dir,
            )
            
            self.datasets.append(dataset)
            self.weights.append(weight)
            self.dataset_names.append(name)
            
            print(f"  → Loaded {len(dataset)} images")
        
        # 归一化权重为概率分布
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        # 计算总样本数（使用最大数据集的大小）
        self.total_samples = max(len(d) for d in self.datasets)
        
        print(f"\n[MultiDatasetWrapper] Summary:")
        print(f"  Total datasets: {len(self.datasets)}")
        print(f"  Total samples (max): {self.total_samples}")
        print(f"  Sampling weights:")
        for name, weight in zip(self.dataset_names, self.weights):
            print(f"    - {name}: {weight:.2%}")
    
    def __len__(self) -> int:
        """返回数据集大小（使用最大数据集的大小）"""
        return self.total_samples
    
    def __getitem__(self, idx: int) -> dict:
        """
        获取单个样本（按权重随机选择数据集）
        
        Args:
            idx: 样本索引
        
        Returns:
            sample: 数据样本字典
        """
        # 按权重随机选择数据集
        dataset_idx = np.random.choice(
            len(self.datasets),
            p=self.weights
        )
        
        # 从选中的数据集获取样本
        selected_dataset = self.datasets[dataset_idx]
        
        # 使用取模确保索引在范围内
        sample_idx = idx % len(selected_dataset)
        
        # 获取样本
        sample = selected_dataset[sample_idx]
        
        # 可选：添加数据集来源标记（用于调试）
        # sample['_source_dataset'] = self.dataset_names[dataset_idx]
        
        return sample
    
    def get_dataset_stats(self) -> dict:
        """
        获取所有数据集的统计信息
        
        Returns:
            stats: 统计信息字典
        """
        stats = {
            "num_datasets": len(self.datasets),
            "total_samples": self.total_samples,
            "datasets": []
        }
        
        for name, dataset, weight in zip(self.dataset_names, self.datasets, self.weights):
            dataset_info = {
                "name": name,
                "num_images": len(dataset),
                "sampling_weight": weight,
                "sampling_percentage": f"{weight:.2%}"
            }
            stats["datasets"].append(dataset_info)
        
        return stats


def extract_dataset_config(dataset) -> dict:
    """
    从 InpaintingSketchDataset 或 MultiDatasetWrapper 提取可序列化的配置字典。
    用于多进程预加载时传递配置，避免 pickle 整个 dataset 对象。

    Args:
        dataset: InpaintingSketchDataset 或 MultiDatasetWrapper 实例

    Returns:
        config: 配置字典，可供 create_dataset_from_config 重建
    """
    if hasattr(dataset, 'datasets'):
        # MultiDatasetWrapper: 从已有 datasets 反推 config
        datasets_config = []
        total_weight = sum(dataset.weights)
        for name, ds, w in zip(dataset.dataset_names, dataset.datasets, dataset.weights):
            single_cfg = extract_dataset_config(ds)
            datasets_config.append({
                "name": name,
                "path": str(single_cfg["image_dir"]),
                "weight": w * total_weight,  # 反归一化
                "recursive_scan": single_cfg.get("recursive_scan", True),
            })
        return {
            "type": "multi",
            "datasets_config": datasets_config,
            "resolution": dataset.datasets[0].resolution if dataset.datasets else 512,
            "enable_edge_cache": dataset.datasets[0].enable_edge_cache if dataset.datasets else True,
            "dexined_checkpoint": dataset.datasets[0].dexined_checkpoint if dataset.datasets else None,
            "dexined_threshold": dataset.datasets[0].dexined_threshold if dataset.datasets else None,
            "dexined_device": dataset.datasets[0].dexined_device if dataset.datasets else None,
            "sketch_params": dict(dataset.datasets[0].sketch_params) if dataset.datasets else None,
            "mask_params": dict(dataset.datasets[0].mask_params) if dataset.datasets else None,
            "debug_edge": dataset.datasets[0].debug_edge if dataset.datasets else False,
            "debug_edge_output_dir": str(dataset.datasets[0].debug_edge_output_dir) if dataset.datasets and dataset.datasets[0].debug_edge_output_dir else None,
        }
    else:
        # InpaintingSketchDataset
        return {
            "type": "single",
            "image_dir": str(dataset.image_dir),
            "resolution": dataset.resolution,
            "enable_edge_cache": dataset.enable_edge_cache,
            "dexined_checkpoint": dataset.dexined_checkpoint,
            "dexined_threshold": dataset.dexined_threshold,
            "dexined_device": dataset.dexined_device,
            "sketch_params": dict(dataset.sketch_params),
            "mask_params": dict(dataset.mask_params),
            "recursive_scan": dataset.recursive_scan,
            "debug_edge": dataset.debug_edge,
            "debug_edge_output_dir": str(dataset.debug_edge_output_dir) if dataset.debug_edge_output_dir else None,
        }


def create_dataset_from_config(config: dict):
    """
    从配置字典创建 InpaintingSketchDataset 或 MultiDatasetWrapper 实例。
    用于多进程 worker 内重建 dataset，避免跨进程传递大对象。

    Args:
        config: 由 extract_dataset_config 产生的配置字典

    Returns:
        dataset: InpaintingSketchDataset 或 MultiDatasetWrapper 实例
    """
    cfg = dict(config)
    dexined_device = cfg.pop("dexined_device", None) or DEXINED_DEVICE

    if cfg.get("type") == "multi":
        datasets_config = cfg.pop("datasets_config", [])
        return MultiDatasetWrapper(
            datasets_config=datasets_config,
            resolution=cfg.get("resolution", 512),
            enable_edge_cache=cfg.get("enable_edge_cache", True),
            dexined_checkpoint=cfg.get("dexined_checkpoint"),
            dexined_threshold=cfg.get("dexined_threshold"),
            dexined_device=dexined_device,
            sketch_params=cfg.get("sketch_params"),
            mask_params=cfg.get("mask_params"),
            debug_edge=cfg.get("debug_edge", False),
            debug_edge_output_dir=cfg.get("debug_edge_output_dir"),
        )
    else:
        return InpaintingSketchDataset(
            image_dir=cfg["image_dir"],
            resolution=cfg.get("resolution", 512),
            enable_edge_cache=cfg.get("enable_edge_cache", True),
            dexined_checkpoint=cfg.get("dexined_checkpoint"),
            dexined_threshold=cfg.get("dexined_threshold"),
            dexined_device=dexined_device,
            sketch_params=cfg.get("sketch_params"),
            mask_params=cfg.get("mask_params"),
            recursive_scan=cfg.get("recursive_scan", True),
            debug_edge=cfg.get("debug_edge", False),
            debug_edge_output_dir=cfg.get("debug_edge_output_dir"),
        )


if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description="Test InpaintingSketchDataset")
    parser.add_argument("--image_dir", type=str, required=True, help="Image directory")
    parser.add_argument("--output_dir", type=str, default="test_output", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to test")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--enable_cache", action="store_true", help="Enable edge cache")
    args = parser.parse_args()
    
    # 创建数据集
    print(f"Creating dataset from: {args.image_dir}")
    dataset = InpaintingSketchDataset(
        image_dir=args.image_dir,
        resolution=args.resolution,
        enable_edge_cache=args.enable_cache,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试加载样本
    print(f"\nTesting {args.num_samples} samples...")
    for i in range(min(args.num_samples, len(dataset))):
        print(f"\nLoading sample {i}...")
        sample = dataset[i]
        
        # 保存结果
        image_path = output_dir / f"sample_{i}_image.png"
        sketch_path = output_dir / f"sample_{i}_sketch.png"
        mask_path = output_dir / f"sample_{i}_mask.png"
        
        sample['image'].save(image_path)
        sample['conditioning_image'].save(sketch_path)
        sample['mask'].save(mask_path)
        
        print(f"  Saved to: {output_dir}/sample_{i}_*.png")
        print(f"  Caption: '{sample['text']}'")
    
    # 显示缓存统计
    if args.enable_cache:
        print("\n" + "="*50)
        print("Cache Statistics:")
        stats = dataset.get_cache_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    print("\nTest completed!")
