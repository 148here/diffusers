# 多数据集使用示例

本文档说明如何使用YZApatch的多数据集功能。

## 场景说明

假设你有两个数据集：

1. **mural1**：壁画数据集，图片直接放在一个文件夹中
   ```
   mural1/
   ├── mural_001.jpg
   ├── mural_002.jpg
   └── ...
   ```

2. **artbench**：艺术作品数据集，按类别组织
   ```
   artbench/
   ├── impressionism/
   │   ├── art_001.jpg
   │   └── art_002.jpg
   ├── cubism/
   │   ├── art_003.jpg
   │   └── art_004.jpg
   └── ...
   ```

你希望：
- 同时使用这两个数据集训练
- mural1和artbench各占50%的采样比例
- 保持数据混合的随机性

## 配置步骤

### 步骤1：编辑config.py

打开 `YZApatch/config.py`，找到 `DATASETS_CONFIG` 部分：

```python
# ============================================================
# 多数据集配置（支持配置多个数据集及其采样权重）
# ============================================================

DATASETS_CONFIG = [
    {
        "name": "mural1",
        "path": "D:/Coding/lab/TSA-inpainting/codes/data/mural1",
        "weight": 1.0,
        "recursive_scan": False,  # 单层目录，直接扫描图片
    },
    {
        "name": "artbench",
        "path": "D:/Coding/lab/TSA-inpainting/codes/data/artbench",
        "weight": 1.0,
        "recursive_scan": True,  # 两层目录，递归扫描子目录
    },
]
```

**重要说明**：
- `name`: 数据集名称（用于日志显示）
- `path`: 数据集实际路径（修改为你的路径）
- `weight`: 采样权重（1.0表示相等权重）
- `recursive_scan`: 
  - `False` = 单层目录结构（直接扫描path下的所有图片）
  - `True` = 两层目录结构（扫描path/子目录/*.jpg）

### 步骤2：验证配置

运行测试脚本验证配置：

```bash
cd D:\Coding\lab\TSA-inpainting\codes\diffusers\examples\controlnet\YZApatch
python test_multi_dataset.py
```

你应该看到类似输出：

```
==================================
测试1: 配置验证
==================================
[PASS] 找到 2 个数据集配置

数据集 1:
  名称: mural1
  路径: D:/Coding/lab/TSA-inpainting/codes/data/mural1
  权重: 1.0
  递归扫描: False

数据集 2:
  名称: artbench
  路径: D:/Coding/lab/TSA-inpainting/codes/data/artbench
  权重: 1.0
  递归扫描: True

==================================
测试2: 数据集加载
==================================
[MultiDatasetWrapper] Initializing 2 datasets...
...
[PASS] 多数据集包装器创建成功
  总样本数: 1000
  数据集数量: 2

==================================
测试3: 采样权重分布（采样100次）
==================================
配置的权重:
  mural1: 50.00%
  artbench: 50.00%

实际采样分布:
  mural1: 52/100 = 52.00%
  artbench: 48/100 = 48.00%
```

### 步骤3：启动训练

使用标准训练命令即可：

```bash
accelerate launch train_controlnet_sdxl.py \
  --pretrained_model_name_or_path="diffusers/stable-diffusion-xl-1.0-inpainting-0.1" \
  --controlnet_model_name_or_path="xinsir/controlnet-scribble-sdxl-1.0" \
  --use_custom_dataset \
  --enable_edge_cache \
  --resolution=512 \
  --learning_rate=1e-5 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=10000 \
  --checkpointing_steps=500 \
  --output_dir="output/controlnet-multi-dataset" \
  --proportion_empty_prompts=1.0 \
  --mixed_precision="fp16" \
  --enable_xformers_memory_efficient_attention \
  --dataloader_num_workers=4 \
  --report_to="tensorboard"
```

**注意**：不需要指定 `--train_data_dir`，因为使用了 `DATASETS_CONFIG`。

训练日志中会显示：

```
======================================================================
Using YZApatch custom dataset for real-time sketch generation
======================================================================
Multi-dataset mode: 2 datasets configured
  Dataset 1: mural1
    Path: D:/Coding/lab/TSA-inpainting/codes/data/mural1
    Weight: 1.0
    Recursive: False
  Dataset 2: artbench
    Path: D:/Coding/lab/TSA-inpainting/codes/data/artbench
    Weight: 1.0
    Recursive: True

[MultiDatasetWrapper] Initializing 2 datasets...
...
[MultiDatasetWrapper] Summary:
  Total datasets: 2
  Total samples (max): 1000
  Sampling weights:
    - mural1: 50.00%
    - artbench: 50.00%
```

## 调整采样权重

如果你希望mural1占更多比例，调整weight：

```python
DATASETS_CONFIG = [
    {
        "name": "mural1",
        "path": "...",
        "weight": 2.0,  # 增加到2.0
        "recursive_scan": False,
    },
    {
        "name": "artbench",
        "path": "...",
        "weight": 1.0,  # 保持1.0
        "recursive_scan": True,
    },
]
```

结果：
- mural1: 67% (2.0 / (2.0 + 1.0))
- artbench: 33% (1.0 / (2.0 + 1.0))

## 添加更多数据集

可以添加任意数量的数据集：

```python
DATASETS_CONFIG = [
    {"name": "mural1", "path": "...", "weight": 1.0, "recursive_scan": False},
    {"name": "artbench", "path": "...", "weight": 1.0, "recursive_scan": True},
    {"name": "paintings", "path": "...", "weight": 0.5, "recursive_scan": True},
    {"name": "sketches", "path": "...", "weight": 2.0, "recursive_scan": False},
]
```

权重会自动归一化：
- mural1: 22.2% (1.0 / 4.5)
- artbench: 22.2% (1.0 / 4.5)
- paintings: 11.1% (0.5 / 4.5)
- sketches: 44.4% (2.0 / 4.5)

## 单数据集模式（向后兼容）

如果只想使用一个数据集，有两种方式：

### 方式1：只配置一个数据集

```python
DATASETS_CONFIG = [
    {
        "name": "my_dataset",
        "path": "D:/path/to/images",
        "weight": 1.0,
        "recursive_scan": True,
    },
]
```

### 方式2：使用命令行参数

清空或注释掉 `DATASETS_CONFIG`：

```python
DATASETS_CONFIG = []
```

然后使用 `--train_data_dir` 参数：

```bash
accelerate launch train_controlnet_sdxl.py \
  --use_custom_dataset \
  --train_data_dir="D:/path/to/images" \
  ...
```

## 常见问题

### Q: 两个数据集大小差异很大怎么办？

A: 调整权重来平衡。例如：
- 数据集A：1000张图片，weight=1.0
- 数据集B：100张图片，weight=10.0

这样B会有更高的采样概率，平衡数据集大小差异。

### Q: 可以混合不同分辨率的图片吗？

A: 可以，所有图片会被自动resize到 `--resolution` 指定的大小（默认512）。

### Q: 缓存如何管理？

A: 所有数据集共享同一个边缘缓存目录（`EDGE_CACHE_DIR`），按文件路径哈希自动区分。

### Q: 如何查看训练时的实际采样分布？

A: 可以添加日志或使用tensorboard监控。采样是随机的，长期来看会收敛到配置的权重比例。

## 总结

多数据集功能让你能够：
- ✅ 同时使用多个数据源
- ✅ 灵活控制各数据源的采样比例
- ✅ 支持不同的目录结构
- ✅ 保持向后兼容性
- ✅ 自动权重归一化
- ✅ Per-batch动态混合采样

祝训练顺利！
