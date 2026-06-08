# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RVC 实时语音转换工具 — 基于深度学习的实时变声器。PySide6 桌面GUI，CUDA-only，Python 3.13 + RTX 5060。

## Commands

```bash
# 启动推理 GUI
.venv/Scripts/python.exe app.py --infer

# 启动训练 GUI
.venv/Scripts/python.exe app.py --train

# 安装依赖
.venv/Scripts/pip.exe install -r requirements.txt

# 语法检查
.venv/Scripts/python.exe -m py_compile app.py
```

## Architecture

项目统一入口 `app.py`，通过 `--infer` / `--train` 参数选择模式。GUI 模块在 `app/infer/` 和 `app/train/` 下。

共享核心模块在 `rvc/` 目录下。

### 推理流程

用户选择模型 → 点击"开始" → `LoadThread` 加载模型 → `RealtimeEngine.setup()` 打开音频流 → `sounddevice` 回调 `_cb()` → `VCPipeline.infer()` → SOLA 交叉淡化 → 扬声器输出

### 训练流程

用户选择音频目录 → 预处理（切片/归一化/重采样）→ 提取 F0（RMVPE）→ 提取 HuBERT 特征（768维）→ GAN 训练（Generator + Discriminator）→ 导出推理 `.pth` 模型

GUI 支持分步执行（预处理 / 提取F0 / 提取特征 / 训练）和一键全流程。配置自动保存到 `configs/inuse/train_config.json`。

### 目录结构

```
app.py                    # 统一入口 (--infer / --train)
app/
  infer/
    window.py             # MainWindow — 推理主窗口
    widgets.py            # ModelCard, ModelListData, LoadThread
    tabs/
      settings_tab.py     # 音频设备 + 引擎参数
      models_tab.py       # 模型列表管理
      audio_tab.py        # 降噪 + 音效
      offline_tab.py      # 离线推理
  train/
    window.py             # TrainWindow — 训练主窗口
    widgets.py            # ToolThread, browse helpers
    tabs/
      settings_tab.py     # 数据设置 + 训练参数
      train_tab.py        # 训练步骤 + 进度 + 日志
      tools_tab.py        # 模型合并/查看
rvc/
  audio_io.py             # RealtimeEngine — sounddevice 流 + 音频回调 + 效果链
  vc_pipeline.py          # VCPipeline — HuBERT + 合成器 + FAISS + F0
  audio_loader.py         # 统一音频加载（librosa + ffmpeg fallback）
  offline_worker.py       # OfflineWorker — 离线推理
  hubert.py               # HuBERT 模型加载（带缓存）
  rmvpe.py                # RMVPE F0 提取器
  params.py               # 全局运行时参数单例
  denoise/                # TorchGate 降噪
  nn/                     # 注意力/通用模块/判别器
  synthesizer/            # 合成器模型
    encoder.py            # TextEncoder + PosteriorEncoder
    decoder.py            # Generator + GeneratorNSF + SineGen
    flow.py               # ResidualCouplingBlock
    model.py              # SynthesizerTrnMsNSFsid + _nono 变体
  train/                  # 训练管线
    trainer.py            # GAN 训练循环
    preprocess.py         # 音频预处理（Slicer + 归一化）
    extract_f0.py         # F0 提取
    extract_feature.py    # HuBERT 特征提取
    data_utils.py         # Dataset + BucketSampler
    ckpt_utils.py         # checkpoint 保存/恢复/导出/合并/查看
    losses.py             # GAN 损失函数
    mel_processing.py     # STFT / mel 频谱
```

### 核心组件

- **`RealtimeEngine`** (`rvc/audio_io.py`) — 管理 `sounddevice.Stream`、缓冲区、重采样器、SOLA。`_cb()` 是音频回调（实时线程），读取全局 `p` Params 对象获取参数。
- **`VCPipeline`** (`rvc/vc_pipeline.py`) — 推理管线。加载 HuBERT + 合成器 + 可选 FAISS 索引。`infer()` 接收 16kHz 滚动缓冲区，返回模型采样率音频。
- **`audio_loader`** (`rvc/audio_loader.py`) — 统一音频加载，支持任意格式，自动 fallback 到 ffmpeg。
- **`ModelCard`** (`app/infer/widgets.py`) — 可折叠的模型配置卡片。`load_requested` 信号通知 MainWindow 选中模型。
- **`ModelListData`** (`app/infer/widgets.py`) — `configs/models.json` 的读写。
- **`Params`** — 全局运行时参数单例（`p = Params()`），回调线程读、主线程写（依赖 GIL 保证原子性）。

### 判别器

`rvc/nn/discriminator.py` 中的 `MultiPeriodDiscriminatorV2`（periods `[2,3,5,7,11,17,23,37]`），仅训练时使用，不导出。

### 合成器 (`rvc/synthesizer/`)

- `encoder.py` — `TextEncoder`（文本/特征编码）+ `PosteriorEncoder`（频谱后验编码）
- `decoder.py` — `Generator` + `GeneratorNSF`（声码器）+ `SineGen` + `SourceModuleHnNSF`
- `flow.py` — `ResidualCouplingBlock`（可逆耦合流）
- `model.py` — `_SynthesizerTrnMsBase` / `SynthesizerTrnMsNSFsid`（F0）+ `_nono` 变体

### 模型缓存

模块级缓存，跨实例复用：
- `rvc/hubert.py` → `_hubert_cache` — HuBERT 模型（设备级缓存）
- `rvc/vc_pipeline.py` → `_rmvpe_cache` / `_fcpe_cache` — F0 提取器（设备级缓存）

合成器不缓存（每个 .pth 文件权重不同，每次重新加载）。

### 预训练权重 (`assets/pretrained_v2/`)

- `f0G48k.pth` / `f0D48k.pth` — 48k 带 F0 的 Generator/Discriminator 预训练权重
- `f0G32k.pth` / `f0D32k.pth` — 32k 带 F0 版本
- `G*.pth` / `D*.pth` — 无 F0 版本（本项目不用）
- 训练时通过"预训练 G/D"路径加载，加速收敛

### 配置持久化

- `configs/inuse/gui_config.json` — 推理 GUI 配置（设备、引擎参数、EQ、选中的模型路径）
- `configs/inuse/train_config.json` — 训练 GUI 配置（实验名、音频目录、采样率、训练参数）
- `configs/models.json` — 推理模型列表（name/pth/idx/pitch/index_rate/rms_mix/gender）
- `configs/v2/48k.json` / `configs/v2/32k.json` — 训练超参数（模型结构、学习率、batch size 等）

### 关键模式

- **滚动缓冲区**: 音频以 ~250ms 块流入，维护 2.5s 上下文窗口。HuBERT 在整个缓冲区上运行，但只解码最新块。
- **SOLA 交叉淡化**: 使用互相关找到最佳拼接点，正弦平方窗口淡入淡出。
- **F0 方法**: `fcpe`（快速）和 `rmvpe`（精确），全局选择，不绑定单个模型。
- **索引**: `index_rate > 0` 时才加载 FAISS 索引（惰性加载）。
- **声学效果**: EQ (低/中/高频 biquad) + tanh 饱和 + 动态压限 + 4-tap 延迟混响。5 个内置预设。
- **训练 checkpoint**: `G_*.pth` / `D_*.pth` 保存在 `logs/<exp>/`，支持中断恢复
- **模型导出**: 从 Generator checkpoint 移除 `enc_q.*` 权重，转 half，打包为推理 `.pth`
- **日志**: 统一中文格式，每个组件一行（`加载 HuBERT` / `加载 Synthesizer` / `加载完成`）。

## Environment

- **Python**: 3.13+
- **GPU**: NVIDIA only (CUDA required). Target: RTX 5060
- **fairseq-fixed**: 社区修复版 fairseq，用于加载 HuBERT 模型
- **OS**: Windows 11
