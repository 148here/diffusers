# -*- coding: utf-8 -*-
"""
边缘缓存管理器
==============

实现边缘提取结果的磁盘缓存，支持：
- 基于文件路径和参数的哈希缓存键
- 自动创建缓存目录
- 缓存失效检测（源文件修改时间）
- 支持启用/禁用缓存
- 线程安全（多worker加载）
"""

import os
import sys
import hashlib
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import threading


class EdgeCacheManager:
    """
    边缘缓存管理器（单例模式）
    
    缓存策略：
    - 缓存键：基于(文件路径 + 修改时间 + DexiNed参数)的哈希
    - 缓存格式：pickle序列化的numpy数组
    - 失效检测：比较源文件修改时间
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, cache_dir: str, dexined_params: dict):
        """
        单例模式：确保全局只有一个缓存管理器实例
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, cache_dir: str, dexined_params: dict):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
            dexined_params: DexiNed参数字典（用于生成缓存键）
        """
        # 避免重复初始化
        if self._initialized:
            return
        
        self.cache_dir = Path(cache_dir)
        self.dexined_params = dexined_params
        
        # 创建缓存目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 线程锁（用于文件读写）
        self.file_locks = {}
        self.file_locks_lock = threading.Lock()
        
        self._initialized = True
        
        print(f"[EdgeCache] Initialized cache directory: {self.cache_dir}")
    
    def _get_file_lock(self, cache_key: str) -> threading.Lock:
        """
        获取文件级别的锁（避免多线程同时写同一文件）
        
        Args:
            cache_key: 缓存键
        
        Returns:
            file_lock: 文件锁
        """
        with self.file_locks_lock:
            if cache_key not in self.file_locks:
                self.file_locks[cache_key] = threading.Lock()
            return self.file_locks[cache_key]
    
    def _compute_cache_key(self, image_path: str, file_mtime: float) -> str:
        """
        计算缓存键
        
        Args:
            image_path: 图片文件路径
            file_mtime: 文件修改时间戳
        
        Returns:
            cache_key: 缓存键字符串
        """
        # 组合所有影响边缘提取结果的因素
        key_components = [
            image_path,
            str(file_mtime),
            str(self.dexined_params.get("threshold", 45)),
            str(self.dexined_params.get("version", "v1")),
        ]
        
        # 计算MD5哈希
        key_string = "|".join(key_components)
        cache_key = hashlib.md5(key_string.encode()).hexdigest()
        
        return cache_key
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """
        获取缓存文件路径
        
        Args:
            cache_key: 缓存键
        
        Returns:
            cache_path: 缓存文件路径
        """
        # 使用两级目录结构（避免单目录文件过多）
        subdir = cache_key[:2]
        cache_path = self.cache_dir / subdir / f"{cache_key}.pkl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        return cache_path
    
    def load_from_cache(self, image_path: str) -> Optional[np.ndarray]:
        """
        从缓存加载边缘图
        
        Args:
            image_path: 图片文件路径
        
        Returns:
            edge_image: 边缘图 [H,W,3]，如果缓存不存在则返回None
        """
        # 检查文件是否存在
        if not os.path.exists(image_path):
            return None
        
        # 获取文件修改时间
        file_mtime = os.path.getmtime(image_path)
        
        # 计算缓存键
        cache_key = self._compute_cache_key(image_path, file_mtime)
        cache_path = self._get_cache_path(cache_key)
        
        # 检查缓存是否存在
        if not cache_path.exists():
            return None
        
        # 加载缓存（带文件锁）
        file_lock = self._get_file_lock(cache_key)
        with file_lock:
            try:
                with open(cache_path, 'rb') as f:
                    edge_image = pickle.load(f)
                return edge_image
            except Exception as e:
                print(f"[EdgeCache] Failed to load cache: {e}")
                return None
    
    def save_to_cache(self, image_path: str, edge_image: np.ndarray) -> None:
        """
        保存边缘图到缓存
        
        Args:
            image_path: 图片文件路径
            edge_image: 边缘图 [H,W,3]
        """
        # 获取文件修改时间
        file_mtime = os.path.getmtime(image_path)
        
        # 计算缓存键
        cache_key = self._compute_cache_key(image_path, file_mtime)
        cache_path = self._get_cache_path(cache_key)
        
        # 保存缓存（带文件锁）
        file_lock = self._get_file_lock(cache_key)
        with file_lock:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(edge_image, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(f"[EdgeCache] Failed to save cache: {e}")
    
    def get_or_compute_edge(self, image_path: str, image_np: np.ndarray,
                           enable_cache: bool, 
                           extract_fn: callable) -> np.ndarray:
        """
        获取或计算边缘图（核心方法）
        
        Args:
            image_path: 图片文件路径
            image_np: 图片numpy数组 [H,W,3]
            enable_cache: 是否启用缓存
            extract_fn: 边缘提取函数，签名为 fn(image_np) -> edge_np
        
        Returns:
            edge_image: 边缘图 [H,W,3]
        """
        # 如果启用缓存，先尝试加载
        if enable_cache:
            cached_edge = self.load_from_cache(image_path)
            if cached_edge is not None:
                return cached_edge
        
        # 缓存未命中，调用边缘提取函数
        edge_image = extract_fn(image_np)
        
        # 如果启用缓存，保存结果
        if enable_cache:
            self.save_to_cache(image_path, edge_image)
        
        return edge_image
    
    def clear_cache(self, image_path: Optional[str] = None) -> None:
        """
        清除缓存
        
        Args:
            image_path: 图片文件路径，如果为None则清除所有缓存
        """
        if image_path is None:
            # 清除所有缓存
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                print(f"[EdgeCache] Cleared all cache")
        else:
            # 清除指定文件的缓存
            file_mtime = os.path.getmtime(image_path)
            cache_key = self._compute_cache_key(image_path, file_mtime)
            cache_path = self._get_cache_path(cache_key)
            
            if cache_path.exists():
                cache_path.unlink()
                print(f"[EdgeCache] Cleared cache for: {image_path}")
    
    def get_cache_stats(self) -> dict:
        """
        获取缓存统计信息
        
        Returns:
            stats: 缓存统计字典
        """
        # 统计缓存文件数量和总大小
        cache_files = list(self.cache_dir.rglob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "num_files": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
        }


def get_edge_cache_manager(cache_dir: str, dexined_params: dict) -> EdgeCacheManager:
    """
    获取边缘缓存管理器实例（单例）
    
    Args:
        cache_dir: 缓存目录路径
        dexined_params: DexiNed参数字典
    
    Returns:
        manager: EdgeCacheManager实例
    """
    return EdgeCacheManager(cache_dir, dexined_params)


if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description="Test edge cache manager")
    parser.add_argument("--action", type=str, choices=["stats", "clear"], 
                       default="stats", help="Action to perform")
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache directory")
    args = parser.parse_args()
    
    # 创建管理器
    dexined_params = {"threshold": 45, "version": "v1"}
    manager = get_edge_cache_manager(args.cache_dir, dexined_params)
    
    if args.action == "stats":
        # 显示统计信息
        stats = manager.get_cache_stats()
        print(f"Cache directory: {stats['cache_dir']}")
        print(f"Number of cached files: {stats['num_files']}")
        print(f"Total cache size: {stats['total_size_mb']:.2f} MB")
    
    elif args.action == "clear":
        # 清除所有缓存
        manager.clear_cache()
        print("Cache cleared successfully")
