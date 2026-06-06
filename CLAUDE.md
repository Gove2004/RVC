# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RVC 实时语音转换工具 — 基于深度学习的实时变声器。PySide6 桌面GUI，CUDA-only，Python 3.13 + RTX 5060。

## Commands

```bash
# 启动推理 GUI
.venv/Scripts/python.exe app.py

# 启动训练 GUI
.venv/Scripts/python.exe train_app.py

# 安装依赖
.venv/Scripts/pip.exe install -r requirements.txt

# 语法检查（训练相关）
.venv/Scripts/python.exe -m py_compile train_app.py rvc/train/*.py rvc/nn/discriminator.py
```

## Architecture

项目包含两个独立程序：

- **推理程序** (`app.py`) — 实时/离线变声
- **训练程序** (`train_app.py`) — 从干净音频训练 RVC 模型

共享模块在 `rvc/` 目录下。

### 推理流程

用户选择模型 → 点击"开始" → `LoadThread` 加载模型 → `RealtimeEngine.setup()` 打开音频流 → `sounddevice` 回调 `_cb()` → `RealtimeVC.infer()` → SOLA 交叉淡化 → 扬声器输出

### 训练流程

用户选择音频目录 → 预处理（切片/归一化/重采样）→ 提取 F0（RMVPE）→ 提取 HuBERT 特征（768维）→ GAN 训练（Generator + Discriminator）→ 导出推理 `.pth` 模型

GUI 支持分步执行（预处理 / 提取F0 / 提取特征 / 训练）和一键全流程。配置自动保存到 `configs/inuse/train_config.json`。

### GUI 结构 (`app.py`)

三个 Tab + 底部控制栏，统一 Grid 布局（间距 6px、边距 10px、标签最小宽度 35px）：

- **设置** (`_build_settings_tab`) — 音频设备路由 + 引擎参数（阈值、采样长度、淡入淡出、额外上下文、F0 算法）
- **模型** (`_build_models_tab`) — 可滚动的 `ModelCard` 列表，单选选中，展开显示参数（音调/性别/索引/响度），删除按钮在展开底部
- **声学** (`_build_audio_tab`) — 降噪 + 音效（EQ/饱和/压限/混响 + 预设系统）

底部控制栏：`开始`（成功后显示"运行中"）| `停止` | `当前: 模型名` | `延迟: X` | `推理: X`

### 核心组件

- **`RealtimeEngine`** — 管理 `sounddevice.Stream`、缓冲区、重采样器、SOLA。`_cb()` 是音频回调（实时线程），读取全局 `p` Params 对象获取参数。
- **`RealtimeVC`** — 推理引擎。加载 HuBERT + 合成器 + 可选 FAISS 索引。`infer()` 接收 16kHz 滚动缓冲区，返回模型采样率音频。
- **`ModelCard`** — 可折叠的模型配置卡片。`load_requested` 信号通知 MainWindow 选中模型。
- **`ModelListData`** — `configs/models.json` 的读写。
- **`Params`** — 全局运行时参数单例（`p = Params()`），回调线程读、主线程写（依赖 GIL 保证原子性）。

### 训练组件 (`rvc/train/`)

- **`train_worker.py`** — `TrainWorker(QThread)` 后台线程，根据 `step` 参数选择性执行训练步骤
- **`trainer.py`** — `Trainer` 类，单 GPU GAN 训练循环。`TrainConfig` dataclass 配置。`setup()` + `train()` 分离
- **`preprocess.py`** — `PreProcessor` 音频预处理（Slicer 静音切分、高通滤波、归一化）
- **`extract_f0.py`** — `F0Extractor`，复用 `rvc/rmvpe.py` 的 RMVPE
- **`extract_feature.py`** — `HuBERTExtractor`，复用 `rvc/hubert.py` 的 `load_hubert()`
- **`data_utils.py`** — `TextAudioLoaderMultiNSFsid` Dataset + `BucketSampler`（按长度分桶）
- **`losses.py`** — feature_loss / discriminator_loss / generator_loss / kl_loss
- **`mel_processing.py`** — STFT / mel 频谱计算（全局缓存 hann_window 和 mel_basis）
- **`ckpt_utils.py`** — checkpoint 保存/恢复/导出（移除 enc_q，转 half）

### 判别器 (`rvc/nn/discriminator.py`)

- `MultiPeriodDiscriminatorV2` — v2 判别器，periods `[2,3,5,7,11,17,23,37]`
- 训练时与 `SynthesizerTrnMsNSFsid`（Generator）配合使用
- 训练结束后不导出，仅 Generator 保存为推理模型

### 模型架构 (`rvc/synthesizer.py`)

- `_SynthesizerTrnMsBase` — F0 模型基类，包含 `forward()`（训练用）和 `infer()`（推理用）
- `_SynthesizerTrnMsBase_nono` — 无 F0 模型基类，只有 `infer()`
- `SynthesizerTrnMsNSFsid` — v2 768 维 F0 模型（训练+推理），继承 `_SynthesizerTrnMsBase`
- `SynthesizerTrnMsNSFsid_nono` — v2 768 维无 F0 模型（仅推理）

### 模型缓存

模块级缓存，跨实例复用：
- `rvc/hubert.py` → `_hubert_cache` — HuBERT 模型（设备级缓存）
- `rvc/realtime_engine.py` → `_rmvpe_cache` / `_fcpe_cache` — F0 提取器（设备级缓存）

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
