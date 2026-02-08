@echo off
REM 多数据集训练 - 快速启动脚本
REM ==========================================

echo ========================================
echo YZApatch 多数据集训练启动脚本
echo ========================================
echo.

REM 检查配置
echo [1/3] 检查配置...
cd /d "%~dp0YZApatch"
python -c "from config import DATASETS_CONFIG; print(f'找到 {len(DATASETS_CONFIG)} 个数据集配置'); [print(f\"  - {cfg['name']}: {cfg['path']}\") for cfg in DATASETS_CONFIG]"
if errorlevel 1 (
    echo [ERROR] 配置检查失败，请检查 YZApatch/config.py
    pause
    exit /b 1
)
echo.

REM 运行测试（可选，注释掉可跳过）
echo [2/3] 运行数据集测试（可选，按Ctrl+C跳过）...
timeout /t 3 /nobreak
python test_multi_dataset.py
if errorlevel 1 (
    echo [WARNING] 测试失败，但继续训练...
)
echo.

REM 启动训练
echo [3/3] 启动训练...
cd /d "%~dp0"
accelerate launch train_controlnet_sdxl.py ^
  --pretrained_model_name_or_path="diffusers/stable-diffusion-xl-1.0-inpainting-0.1" ^
  --controlnet_model_name_or_path="xinsir/controlnet-scribble-sdxl-1.0" ^
  --use_custom_dataset ^
  --enable_edge_cache ^
  --resolution=512 ^
  --learning_rate=1e-5 ^
  --train_batch_size=4 ^
  --gradient_accumulation_steps=4 ^
  --max_train_steps=10000 ^
  --checkpointing_steps=500 ^
  --output_dir="output/controlnet-multi-dataset" ^
  --proportion_empty_prompts=1.0 ^
  --mixed_precision="fp16" ^
  --enable_xformers_memory_efficient_attention ^
  --dataloader_num_workers=4 ^
  --report_to="tensorboard"

echo.
echo ========================================
echo 训练完成或中断
echo ========================================
pause
