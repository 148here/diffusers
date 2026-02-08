# YZApatch - ControlNet训练自定义数据集模块

## 概述

YZApatch是一个用于ControlNet训练的自定义数据集模块，提供以下功能：

- **实时sketch生成**：使用DexiNed边缘提取 + 参数化形变
- **复杂mask生成**：5-10个不规则矩形块 + 随机笔刷描边
- **边缘缓存**：磁盘缓存加速训练（可选）
- **空字符串caption**：支持无文本条件的训练
- **灵活的数据结构**：支持单目录或多子目录

## 快速开始

### 1. 配置路径

#### 方式A：多数据集模式（推荐）

编辑 `YZApatch/config.py` 文件中的 `DATASETS_CONFIG` 配置：

```python
DATASETS_CONFIG = [
    {
        "name": "mural1",                                          # 数据集名称
        "path": "D:/Coding/lab/TSA-inpainting/codes/data/mural1", # 数据集路径
        "weight": 1.0,                                             # 采样权重
        "recursive_scan": False,                                   # False=单层目录
    },
    {
        "name": "artbench",                                            # 数据集名称
        "path": "D:/Coding/lab/TSA-inpainting/codes/data/artbench",   # 数据集根路径
        "weight": 1.0,                                                 # 采样权重
        "recursive_scan": True,                                        # True=递归扫描子目录
    },
]
```

**权重说明**：
- `weight: [1.0, 1.0]` → 各占50%采样概率
- `weight: [2.0, 1.0]` → mural1占67%，artbench占33%
- `weight: [1.0, 3.0]` → mural1占25%，artbench占75%

#### 方式B：单数据集模式（向后兼容）

如果只使用一个数据集，可以保持原来的配置方式：

```python
# DexiNed代码目录路径（包含model.py的目录）
DEXINED_CODE_DIR = "D:/Coding/lab/TSA-inpainting/codes/TSA-inpaint/DexiNed"

# DexiNed模型checkpoint路径
DEXINED_CHECKPOINT = "D:/Coding/lab/TSA-inpainting/codes/TSA-inpaint/DexiNed/checkpoints/BIPED/10/10_model.pth"

# 训练图片数据目录
TRAIN_DATA_DIR = "D:/Coding/lab/TSA-inpainting/codes/data/train_images"

# 边缘缓存目录（用于加速训练）
EDGE_CACHE_DIR = "D:/Coding/lab/TSA-inpainting/codes/cache/edges"

# sketch_util.py所在目录（用于导入）
SKETCH_UTIL_DIR = "D:/Coding/lab/TSA-inpainting/codes/TSA-inpaint"
```

### 2. 准备数据

将训练图片放在指定目录中：

```
train_images/
├── image1.jpg
├── image2.png
├── image3.jpg
└── ...
```

或者使用子目录结构（自动递归扫描）：

```
train_images/
├── category1/
│   ├── image1.jpg
│   └── image2.jpg
├── category2/
│   ├── image3.jpg
│   └── image4.jpg
└── ...
```

### 3. 启动训练

使用以下命令启动训练：

```bash
accelerate launch train_controlnet_sdxl.py \
  --pretrained_model_name_or_path="diffusers/stable-diffusion-xl-1.0-inpainting-0.1" \
  --controlnet_model_name_or_path="xinsir/controlnet-scribble-sdxl-1.0" \
  --use_custom_dataset \
  --enable_edge_cache \
  --train_data_dir="path/to/your/images" \
  --resolution=512 \
  --learning_rate=1e-5 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=10000 \
  --checkpointing_steps=500 \
  --output_dir="output/controlnet-sketch-inpaint" \
  --proportion_empty_prompts=1.0 \
  --mixed_precision="fp16" \
  --enable_xformers_memory_efficient_attention \
  --dataloader_num_workers=4 \
  --report_to="tensorboard"
```

## 配置说明

### 核心参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_custom_dataset` | 启用YZApatch自定义数据集 | False |
| `--enable_edge_cache` | 启用边缘缓存（加速训练） | False |
| `--train_data_dir` | 训练图片目录路径 | 必填 |
| `--resolution` | 图像分辨率 | 512 |

### DexiNed参数（config.py）

```python
DEXINED_THRESHOLD = 45           # 边缘提取阈值（0-255）
DEXINED_DEVICE = "cuda"          # 推理设备: "cuda" or "cpu"
```

### Sketch生成参数（config.py）

```python
SKETCH_PARAMS = {
    "sigma_mean": 13.0,          # FFD位移强度基准值
    "sigma_std": 2.6,            # FFD位移倍率标准差（20%变化）
    "spatial_smooth_sigma": 2.0, # FFD空间平滑半径
    "cp_sigma_mean": 2.1,        # 线段控制点偏移基准值
    "cp_sigma_std": 0.4,         # 线段控制点倍率标准差（19%变化）
    "cp_spatial_smooth": 1.5,    # 线段控制点平滑半径
}
```

### Mask生成参数（config.py）

```python
MASK_PARAMS = {
    "num_blocks_range": (5, 10),        # 矩形块数量范围
    "area_ratio_range": (0.2, 0.5),     # mask总面积占比范围
    "min_block_size": 32,                # 单个块最小尺寸（像素）
    "max_block_ratio": 0.3,              # 单个块最大占比
    "rotation_prob": 0.7,                # 块旋转概率
    "rotation_range": (-30, 30),         # 旋转角度范围（度）
    "brush_stroke_prob": 0.8,            # 添加笔刷描边概率
    "brush_width_range": (3, 10),        # 笔刷宽度范围（像素）
    "brush_segments_range": (3, 8),      # 每次描边的笔画段数
    "edge_density_weight": 2.0,          # 边缘密集区域权重
    "edge_density_threshold": 0.3,       # 边缘密度阈值（0-1）
    "morphology_prob": 0.5,              # 应用形态学操作概率
    "morphology_kernel_size": (3, 7),    # 形态学操作核大小范围
}
```

## 功能详解

### 1. 实时Sketch生成

每次训练迭代都会：

1. 从原图提取边缘（使用DexiNed深度学习模型）
2. 应用参数化形变生成sketch（FFD + 线段控制点）
3. 确保每次生成的sketch都不同（增加训练多样性）

**边缘缓存**：
- 首次运行：提取边缘并保存到缓存
- 后续运行：直接加载缓存（大幅加速）
- 使用 `--enable_edge_cache` 启用

### 2. 复杂Mask生成

生成的mask包含：

- **5-10个不规则块**：矩形或椭圆，支持旋转
- **随机笔刷描边**：1-3条笔刷线，模拟真实涂抹
- **形态学操作**：增加边缘不规则性
- **基于边缘密度**：优先在边缘密集区域放置mask

### 3. 多数据集混合采样（新功能）

支持配置多个数据集，每个数据集有独立的：

- **路径配置**：支持不同的目录结构
- **采样权重**：控制各数据集的采样比例
- **递归扫描**：单层目录或递归子目录

**采样策略**：
- Per-batch动态混合：每个batch按权重随机选择数据集
- 自动权重归一化：权重会自动转换为概率分布
- 充分混合：确保训练数据多样性

**配置示例**：
```python
DATASETS_CONFIG = [
    {
        "name": "mural1",
        "path": "D:/path/to/mural1",
        "weight": 1.0,              # 50%采样概率
        "recursive_scan": False,    # 单层目录
    },
    {
        "name": "artbench",
        "path": "D:/path/to/artbench",
        "weight": 1.0,              # 50%采样概率
        "recursive_scan": True,     # 两层目录结构
    },
]
```

### 4. 数据流程

```
原图(512x512) 
  ↓
边缘提取(DexiNed) → [可选: 缓存]
  ↓
Sketch生成(参数化形变)
  ↓
Mask生成(复杂随机mask)
  ↓
返回: {image, sketch, mask, text=''}
```

## 性能优化

### 边缘缓存

**首次运行**（无缓存）：
- 约 2-5秒/图（取决于GPU）
- 建议先用小数据集测试

**后续运行**（有缓存）：
- 约 0.1-0.2秒/图
- 训练速度提升 10-20倍

### 多进程加载

使用多个dataloader worker加速：

```bash
--dataloader_num_workers=4  # 根据CPU核心数调整
```

**注意**：
- Windows上可能需要设置 `num_workers=0` 避免multiprocessing问题
- Linux/Mac上推荐设置 `num_workers=4-8`

### 混合精度训练

```bash
--mixed_precision="fp16"  # 节省显存，加速训练
```

## 常见问题（FAQ）

### Q1: 如何验证配置是否正确？

运行配置检查脚本：

```bash
cd YZApatch
python __init__.py
```

会显示所有配置路径和参数。

### Q2: 训练速度很慢怎么办？

1. **启用边缘缓存**：`--enable_edge_cache`
2. **增加worker数量**：`--dataloader_num_workers=4`
3. **使用混合精度**：`--mixed_precision="fp16"`
4. **检查GPU利用率**：使用 `nvidia-smi` 监控

### Q3: 如何调整mask的复杂度？

修改 `config.py` 中的 `MASK_PARAMS`：

```python
# 增加复杂度
MASK_PARAMS = {
    "num_blocks_range": (8, 15),      # 增加块数量
    "brush_stroke_prob": 1.0,         # 总是添加笔刷
    "morphology_prob": 0.8,           # 增加形态学操作概率
}

# 降低复杂度
MASK_PARAMS = {
    "num_blocks_range": (3, 5),       # 减少块数量
    "brush_stroke_prob": 0.3,         # 减少笔刷概率
}
```

### Q4: 如何调整sketch的变化程度？

修改 `config.py` 中的 `SKETCH_PARAMS`：

```python
# 增加变化（更夸张的形变）
SKETCH_PARAMS = {
    "sigma_std": 5.0,        # 增加FFD变化（原来2.6）
    "cp_sigma_std": 0.8,     # 增加线段变化（原来0.4）
}

# 减少变化（更接近原始边缘）
SKETCH_PARAMS = {
    "sigma_std": 1.0,        # 减少FFD变化
    "cp_sigma_std": 0.2,     # 减少线段变化
}
```

### Q5: 支持哪些图片格式？

默认支持：`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

修改 `config.py` 中的 `IMAGE_EXTENSIONS` 可添加更多格式。

### Q6: 如何禁用边缘缓存？

不添加 `--enable_edge_cache` 参数即可，每次都会实时提取边缘。

### Q7: 如何禁用边缘缓存？

不添加 `--enable_edge_cache` 参数即可，每次都会实时提取边缘。

### Q8: 如何使用多个数据集训练？

编辑 `config.py` 中的 `DATASETS_CONFIG`：

```python
DATASETS_CONFIG = [
    {
        "name": "mural1",
        "path": "D:/path/to/mural1",
        "weight": 1.0,              # 采样权重
        "recursive_scan": False,    # False=单层目录
    },
    {
        "name": "artbench",
        "path": "D:/path/to/artbench",
        "weight": 1.0,
        "recursive_scan": True,     # True=递归扫描子目录
    },
]
```

权重控制各数据集的采样比例：
- `[1.0, 1.0]` → 各占50%
- `[2.0, 1.0]` → 第一个占67%，第二个占33%
- `[1.0, 3.0]` → 第一个占25%，第二个占75%

### Q9: 如何测试多数据集配置是否正确？

运行测试脚本：

```bash
cd YZApatch
python test_multi_dataset.py
```

这会验证：
- 配置是否正确
- 数据集是否能正常加载
- 采样权重是否符合预期
- 样本是否能正常生成

### Q10: DexiNed模型加载失败怎么办？

检查配置：

1. `DEXINED_CODE_DIR`：确保包含 `model.py` 文件
2. `DEXINED_CHECKPOINT`：确保 `.pth` 文件存在
3. `SKETCH_UTIL_DIR`：确保包含 `dataset/sketch_util.py`

## Troubleshooting

### 错误：ModuleNotFoundError: No module named 'dataset.sketch_util'

**原因**：`SKETCH_UTIL_DIR` 配置错误

**解决**：
1. 检查 `config.py` 中的 `SKETCH_UTIL_DIR` 路径
2. 确保该目录包含 `dataset/sketch_util.py` 文件

### 错误：FileNotFoundError: DexiNed checkpoint not found

**原因**：`DEXINED_CHECKPOINT` 路径错误

**解决**：
1. 检查 `.pth` 文件是否存在
2. 确保使用绝对路径或正确的相对路径

### 错误：CUDA out of memory

**原因**：GPU显存不足

**解决方案**：
1. 降低 `--train_batch_size`（例如从4降到2或1）
2. 使用 `--gradient_accumulation_steps=8` 保持有效batch size
3. 启用 `--enable_xformers_memory_efficient_attention`
4. 使用 `--mixed_precision="fp16"`

### 警告：YZApatch not found

**原因**：训练脚本找不到YZApatch目录

**解决**：
1. 确保 `YZApatch` 文件夹在 `train_controlnet_sdxl.py` 同一目录下
2. 检查文件夹名称是否正确（大小写敏感）

## 测试建议

### 1. 小数据集测试

先用10-20张图片测试pipeline：

```bash
--max_train_samples=20 \
--max_train_steps=100 \
--validation_steps=10
```

### 2. 检查生成结果

在训练目录运行测试脚本：

```bash
cd YZApatch
python custom_dataset.py --image_dir="path/to/images" --output_dir="test_output" --num_samples=5
```

会生成：
- `sample_0_image.png`：原图
- `sample_0_sketch.png`：生成的sketch
- `sample_0_mask.png`：生成的mask

### 3. 监控训练

使用tensorboard监控训练进度：

```bash
tensorboard --logdir=output/controlnet-sketch-inpaint/logs
```

### 4. 验证缓存

第一次运行启用缓存：

```bash
--enable_edge_cache --max_train_samples=100
```

记录时间，然后第二次运行，应该快很多。

## 扩展与定制

### 添加新的数据增强

编辑 `YZApatch/custom_dataset.py`，在 `__getitem__` 方法中添加：

```python
def __getitem__(self, idx: int) -> dict:
    # ... 原有代码 ...
    
    # 添加自定义数据增强
    if np.random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        sketch_pil = sketch_pil.transpose(Image.FLIP_LEFT_RIGHT)
        mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
    
    # ... 返回数据 ...
```

### 自定义Mask生成策略

编辑 `YZApatch/mask_generator.py`，修改 `generate` 方法。

### 自定义Sketch参数

编辑 `YZApatch/config.py`，调整 `SKETCH_PARAMS` 字典。

## 版本信息

- **版本**：1.0.0
- **兼容性**：diffusers >= 0.37.0
- **Python版本**：3.8+
- **PyTorch版本**：1.13+
- **CUDA版本**：11.7+（推荐）

## 许可证

与diffusers项目保持一致。

## 技术支持

遇到问题请检查：

1. 配置文件路径是否正确
2. DexiNed模型是否正确加载
3. 数据目录是否包含有效图片
4. GPU显存是否足够

## 更新日志

### v1.0.0 (2026-02)
- 初始版本
- 实时sketch生成
- 复杂mask生成（5-10个矩形块 + 笔刷描边）
- 边缘缓存机制
- 支持单目录/多子目录数据结构
