# -*- coding: utf-8 -*-
"""
多数据集功能测试脚本
====================

测试MultiDatasetWrapper的功能：
- 配置验证
- 数据集加载
- 采样权重验证
- 数据集混合采样测试
"""

import sys
from pathlib import Path

# 添加YZApatch到路径
yzapatch_dir = Path(__file__).parent
sys.path.insert(0, str(yzapatch_dir))

from config import DATASETS_CONFIG
from custom_dataset import MultiDatasetWrapper
from collections import Counter


def test_config():
    """测试配置是否正确"""
    print("="*70)
    print("测试1: 配置验证")
    print("="*70)
    
    if not DATASETS_CONFIG:
        print("[FAIL] DATASETS_CONFIG is empty")
        return False
    
    print(f"[PASS] 找到 {len(DATASETS_CONFIG)} 个数据集配置")
    
    for i, config in enumerate(DATASETS_CONFIG):
        print(f"\n数据集 {i+1}:")
        print(f"  名称: {config.get('name', 'N/A')}")
        print(f"  路径: {config.get('path', 'N/A')}")
        print(f"  权重: {config.get('weight', 1.0)}")
        print(f"  递归扫描: {config.get('recursive_scan', True)}")
        
        # 检查必需字段
        if not config.get('path'):
            print(f"  [WARN] 缺少path字段")
    
    return True


def test_dataset_loading():
    """测试数据集加载"""
    print("\n" + "="*70)
    print("测试2: 数据集加载")
    print("="*70)
    
    try:
        wrapper = MultiDatasetWrapper(
            datasets_config=DATASETS_CONFIG,
            resolution=512,
            enable_edge_cache=False,  # 测试时禁用缓存
        )
        
        print(f"\n[PASS] 多数据集包装器创建成功")
        print(f"  总样本数: {len(wrapper)}")
        print(f"  数据集数量: {len(wrapper.datasets)}")
        
        return wrapper
        
    except Exception as e:
        print(f"\n[FAIL] 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_sampling_weights(wrapper, num_samples=100):
    """测试采样权重分布"""
    print("\n" + "="*70)
    print(f"测试3: 采样权重分布（采样{num_samples}次）")
    print("="*70)
    
    if wrapper is None:
        print("[SKIP] 跳过测试（wrapper未创建）")
        return
    
    print(f"\n配置的权重:")
    for name, weight in zip(wrapper.dataset_names, wrapper.weights):
        print(f"  {name}: {weight:.2%}")
    
    # 采样并统计
    print(f"\n开始采样...")
    dataset_counts = {name: 0 for name in wrapper.dataset_names}
    
    # 临时标记样本来源（用于统计）
    import numpy as np
    
    # 模拟采样分布
    samples_per_dataset = []
    for _ in range(num_samples):
        dataset_idx = np.random.choice(
            len(wrapper.datasets),
            p=wrapper.weights
        )
        dataset_counts[wrapper.dataset_names[dataset_idx]] += 1
    
    print(f"\n实际采样分布:")
    for name, count in dataset_counts.items():
        percentage = count / num_samples
        print(f"  {name}: {count}/{num_samples} = {percentage:.2%}")
    
    print(f"\n[PASS] 采样权重测试完成")


def test_sample_loading(wrapper, num_samples=3):
    """测试样本加载"""
    print("\n" + "="*70)
    print(f"测试4: 样本加载（加载{num_samples}个样本）")
    print("="*70)
    
    if wrapper is None:
        print("[SKIP] 跳过测试（wrapper未创建）")
        return
    
    try:
        for i in range(min(num_samples, len(wrapper))):
            print(f"\n加载样本 {i}...")
            sample = wrapper[i]
            
            # 验证返回格式
            required_keys = ['image', 'text', 'conditioning_image', 'mask']
            missing_keys = [k for k in required_keys if k not in sample]
            
            if missing_keys:
                print(f"  [WARN] 缺少字段: {missing_keys}")
            else:
                print(f"  [PASS] 包含所有必需字段")
            
            # 显示信息
            print(f"  image: {sample['image'].size if hasattr(sample['image'], 'size') else 'N/A'}")
            print(f"  text: '{sample['text']}'")
            print(f"  conditioning_image: {sample['conditioning_image'].size if hasattr(sample['conditioning_image'], 'size') else 'N/A'}")
            print(f"  mask: {sample['mask'].size if hasattr(sample['mask'], 'size') else 'N/A'}")
        
        print(f"\n[PASS] 样本加载测试完成")
        
    except Exception as e:
        print(f"\n[FAIL] 样本加载失败: {e}")
        import traceback
        traceback.print_exc()


def test_dataset_stats(wrapper):
    """测试数据集统计信息"""
    print("\n" + "="*70)
    print("测试5: 数据集统计信息")
    print("="*70)
    
    if wrapper is None:
        print("[SKIP] 跳过测试（wrapper未创建）")
        return
    
    stats = wrapper.get_dataset_stats()
    
    print(f"\n总数据集数量: {stats['num_datasets']}")
    print(f"总样本数: {stats['total_samples']}")
    print(f"\n各数据集详情:")
    
    for ds_info in stats['datasets']:
        print(f"  {ds_info['name']}:")
        print(f"    图片数量: {ds_info['num_images']}")
        print(f"    采样权重: {ds_info['sampling_weight']:.4f}")
        print(f"    采样概率: {ds_info['sampling_percentage']}")
    
    print(f"\n[PASS] 统计信息测试完成")


def main():
    """主测试流程"""
    print("多数据集功能测试")
    print("="*70)
    
    # 测试1: 配置验证
    if not test_config():
        print("\n配置验证失败，终止测试")
        return
    
    # 测试2: 数据集加载
    wrapper = test_dataset_loading()
    
    if wrapper is None:
        print("\n数据集加载失败，终止后续测试")
        return
    
    # 测试3: 采样权重
    test_sampling_weights(wrapper, num_samples=100)
    
    # 测试4: 样本加载
    test_sample_loading(wrapper, num_samples=3)
    
    # 测试5: 统计信息
    test_dataset_stats(wrapper)
    
    print("\n" + "="*70)
    print("所有测试完成！")
    print("="*70)


if __name__ == "__main__":
    main()
