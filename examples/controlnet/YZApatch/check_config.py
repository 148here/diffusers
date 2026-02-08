# -*- coding: utf-8 -*-
"""
快速配置检查脚本
================

验证YZApatch多数据集配置是否正确
"""

import sys
from pathlib import Path

# 添加YZApatch到路径
yzapatch_dir = Path(__file__).parent
sys.path.insert(0, str(yzapatch_dir))

print("="*70)
print("YZApatch 多数据集配置检查")
print("="*70)
print()

# 1. 导入配置
try:
    from config import (
        DATASETS_CONFIG,
        DEXINED_CODE_DIR,
        DEXINED_CHECKPOINT,
        SKETCH_UTIL_DIR,
        EDGE_CACHE_DIR
    )
    print("✓ 配置文件导入成功")
except ImportError as e:
    print(f"✗ 配置文件导入失败: {e}")
    sys.exit(1)

print()

# 2. 检查数据集配置
print("-" * 70)
print("数据集配置")
print("-" * 70)

if not DATASETS_CONFIG:
    print("⚠ DATASETS_CONFIG 为空，将使用单数据集模式")
else:
    print(f"✓ 找到 {len(DATASETS_CONFIG)} 个数据集配置\n")
    
    for i, config in enumerate(DATASETS_CONFIG):
        print(f"数据集 {i+1}: {config.get('name', '未命名')}")
        print(f"  路径: {config.get('path', 'N/A')}")
        print(f"  权重: {config.get('weight', 1.0)}")
        print(f"  递归扫描: {config.get('recursive_scan', True)}")
        
        # 检查路径是否存在
        path = config.get('path')
        if path and Path(path).exists():
            print(f"  状态: ✓ 路径存在")
        else:
            print(f"  状态: ✗ 路径不存在或无法访问")
        print()

# 3. 计算权重归一化
if DATASETS_CONFIG:
    print("-" * 70)
    print("权重归一化")
    print("-" * 70)
    
    weights = [cfg.get('weight', 1.0) for cfg in DATASETS_CONFIG]
    total = sum(weights)
    normalized = [w / total for w in weights]
    
    print("归一化后的采样概率：")
    for config, prob in zip(DATASETS_CONFIG, normalized):
        print(f"  {config.get('name', '未命名')}: {prob:.2%}")
    print()

# 4. 检查DexiNed配置
print("-" * 70)
print("DexiNed配置")
print("-" * 70)

print(f"代码目录: {DEXINED_CODE_DIR}")
if Path(DEXINED_CODE_DIR).exists():
    print(f"  状态: ✓ 目录存在")
else:
    print(f"  状态: ✗ 目录不存在")

print(f"Checkpoint: {DEXINED_CHECKPOINT}")
if Path(DEXINED_CHECKPOINT).exists():
    print(f"  状态: ✓ 文件存在")
else:
    print(f"  状态: ✗ 文件不存在")

print()

# 5. 检查其他路径
print("-" * 70)
print("其他路径配置")
print("-" * 70)

print(f"Sketch Util目录: {SKETCH_UTIL_DIR}")
if Path(SKETCH_UTIL_DIR).exists():
    print(f"  状态: ✓ 目录存在")
else:
    print(f"  状态: ✗ 目录不存在")

print(f"边缘缓存目录: {EDGE_CACHE_DIR}")
if Path(EDGE_CACHE_DIR).exists():
    print(f"  状态: ✓ 目录存在")
else:
    print(f"  状态: ⓘ 目录不存在（训练时会自动创建）")

print()

# 6. 总结
print("="*70)
print("检查总结")
print("="*70)

errors = []
warnings = []

# 检查必需路径
if not Path(DEXINED_CODE_DIR).exists():
    errors.append("DexiNed代码目录不存在")

if not Path(DEXINED_CHECKPOINT).exists():
    errors.append("DexiNed checkpoint文件不存在")

if not Path(SKETCH_UTIL_DIR).exists():
    errors.append("Sketch Util目录不存在")

if DATASETS_CONFIG:
    for config in DATASETS_CONFIG:
        path = config.get('path')
        if not path or not Path(path).exists():
            errors.append(f"数据集 '{config.get('name')}' 路径不存在或无法访问")
else:
    warnings.append("DATASETS_CONFIG为空，将使用单数据集模式")

if errors:
    print("\n✗ 发现以下错误（必须修复）：")
    for err in errors:
        print(f"  - {err}")
    print("\n请修改 YZApatch/config.py 中的路径配置")
    sys.exit(1)

if warnings:
    print("\n⚠ 警告：")
    for warn in warnings:
        print(f"  - {warn}")

print("\n✓ 配置检查通过！")
print("\n下一步：")
print("  1. 如有错误，请修改 YZApatch/config.py")
print("  2. 运行测试: python test_multi_dataset.py")
print("  3. 启动训练: accelerate launch train_controlnet_sdxl.py --use_custom_dataset")
print()
